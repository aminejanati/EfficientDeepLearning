"""Run sensitivity-guided mixed-precision (INT4/INT8) QAT on a step3 checkpoint."""

import copy
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import (  # noqa: E402
    ResNet18,
    ResNet18_Depthwise,
    ResNet18_GroupedG2,
    ResNet18_GroupedG4,
    ResNet18_GroupedG8,
)
from quantization.quantization_qat_4bit import fake_quantize  # noqa: E402


MODEL_BUILDERS = {
    "resnet18": ResNet18,
    "resnet18_depthwise": ResNet18_Depthwise,
    "resnet18_g2": ResNet18_GroupedG2,
    "resnet18_g4": ResNet18_GroupedG4,
    "resnet18_g8": ResNet18_GroupedG8,
}


# ----------------------
# Initial configuration
# ----------------------
CHECKPOINT_PATH = "Final Weights/step3/resnet18_depthwise_structured_0p7_finetuned_unstructured_0p7_finetuned.pth"
MODEL_ARCH = "resnet18_depthwise"
SENSITIVITY_THRESHOLD = 1  # Accuracy drop threshold in percentage points.
SENSITIVITY_THRESHOLDS_SWEEP = [0.3, 0.5, 1.0]
FORCE_FIRST_AND_LAST_INT8 = True
QAT_EPOCHS = 50
QAT_LR = 0.001
QAT_WEIGHT_DECAY = 5e-4 
QAT_MOMENTUM = 0.9
BATCH_SIZE_TRAIN = 128
BATCH_SIZE_TEST = 100
NUM_WORKERS = 2
DATA_DIR = "data"
RESULTS_JSON = "Final Weights/step4/qat_mixed_precision_results.json"
SAVE_DIR = "Final Weights/step4"


def _adapt_state_dict_for_model(model: nn.Module, state_dict):
    model_keys = model.state_dict().keys()
    has_module_in_model = any(k.startswith("module.") for k in model_keys)
    has_module_in_ckpt = any(k.startswith("module.") for k in state_dict.keys())

    if has_module_in_ckpt and not has_module_in_model:
        return {
            (k[len("module.") :] if k.startswith("module.") else k): v
            for k, v in state_dict.items()
        }
    if has_module_in_model and not has_module_in_ckpt:
        return {f"module.{k}": v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint(model: nn.Module, checkpoint_path: Path, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["net"] if isinstance(checkpoint, dict) and "net" in checkpoint else checkpoint
    state_dict = _adapt_state_dict_for_model(model, state_dict)
    model.load_state_dict(state_dict, strict=True)
    return model


def build_test_loader(data_dir: Path, batch_size: int, num_workers: int):
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    testset = torchvision.datasets.CIFAR10(
        root=str(data_dir), train=False, download=True, transform=transform_test
    )
    return torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )


def build_train_loader(data_dir: Path, batch_size: int, num_workers: int):
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    trainset = torchvision.datasets.CIFAR10(
        root=str(data_dir), train=True, download=True, transform=transform_train
    )
    return torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )


def evaluate(model: nn.Module, dataloader, device: str):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / max(1, len(dataloader))
    acc = 100.0 * correct / max(1, total)
    return avg_loss, acc


class MixedPrecisionQATConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        act_bits=8,
        weight_bits=8,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.act_bits = act_bits
        self.weight_bits = weight_bits

    def forward(self, input):
        quantized_input = fake_quantize(input, num_bits=self.act_bits, symmetric=False)
        quantized_weight = fake_quantize(self.weight, num_bits=self.weight_bits, symmetric=True)
        return F.conv2d(
            quantized_input,
            quantized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class MixedPrecisionQATLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, act_bits=8, weight_bits=8):
        super().__init__(in_features, out_features, bias)
        self.act_bits = act_bits
        self.weight_bits = weight_bits

    def forward(self, input):
        quantized_input = fake_quantize(input, num_bits=self.act_bits, symmetric=False)
        quantized_weight = fake_quantize(self.weight, num_bits=self.weight_bits, symmetric=True)
        return F.linear(quantized_input, quantized_weight, self.bias)


def get_quantizable_layer_names(model: nn.Module) -> List[str]:
    names = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            names.append(name)
    return names


def replace_layers_with_mixed_qat(model: nn.Module, layer_bits: Dict[str, int], prefix: str = ""):
    for name, module in list(model._modules.items()):
        full_name = f"{prefix}.{name}" if prefix else name

        if len(list(module.children())) > 0:
            replace_layers_with_mixed_qat(module, layer_bits, full_name)

        if isinstance(module, nn.Conv2d):
            bits = int(layer_bits.get(full_name, 8))
            qat_conv = MixedPrecisionQATConv2d(
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                module.stride,
                module.padding,
                module.dilation,
                module.groups,
                module.bias is not None,
                module.padding_mode,
                act_bits=bits,
                weight_bits=bits,
            )
            qat_conv.weight.data = module.weight.data.clone()
            if module.bias is not None:
                qat_conv.bias.data = module.bias.data.clone()
            model._modules[name] = qat_conv

        elif isinstance(module, nn.Linear):
            bits = int(layer_bits.get(full_name, 8))
            qat_linear = MixedPrecisionQATLinear(
                module.in_features,
                module.out_features,
                module.bias is not None,
                act_bits=bits,
                weight_bits=bits,
            )
            qat_linear.weight.data = module.weight.data.clone()
            if module.bias is not None:
                qat_linear.bias.data = module.bias.data.clone()
            model._modules[name] = qat_linear

    return model


def build_uniform_layer_bits(model: nn.Module, bits: int) -> Dict[str, int]:
    return {name: bits for name in get_quantizable_layer_names(model)}


def get_forced_int8_layers(model: nn.Module) -> List[str]:
    names = get_quantizable_layer_names(model)
    if not names:
        return []
    if not FORCE_FIRST_AND_LAST_INT8:
        return []
    return [names[0], names[-1]]


def analyze_layer_sensitivity(model: nn.Module, testloader, device: str):
    _, baseline_acc = evaluate(model, testloader, device)
    print(f"Sensitivity baseline accuracy: {baseline_acc:.2f}%")

    layer_names = get_quantizable_layer_names(model)
    forced_int8 = set(get_forced_int8_layers(model))
    candidates = [n for n in layer_names if n not in forced_int8]

    deltas = {}
    for idx, layer_name in enumerate(candidates, start=1):
        probe_model = copy.deepcopy(model)
        probe_bits = build_uniform_layer_bits(probe_model, bits=8)
        probe_bits[layer_name] = 4
        for fixed_name in forced_int8:
            probe_bits[fixed_name] = 8

        probe_model = replace_layers_with_mixed_qat(probe_model, probe_bits).to(device)
        _, probe_acc = evaluate(probe_model, testloader, device)
        delta = baseline_acc - probe_acc
        deltas[layer_name] = delta

        print(
            f"Sensitivity {idx}/{len(candidates)} | layer={layer_name} | "
            f"acc={probe_acc:.2f}% | delta={delta:.2f}%"
        )

    return baseline_acc, deltas, sorted(forced_int8)


def assign_bits_from_sensitivity(
    model: nn.Module,
    deltas: Dict[str, float],
    threshold: float,
    forced_int8_layers: List[str],
):
    layer_bits = build_uniform_layer_bits(model, bits=8)

    for name in layer_bits.keys():
        if name in forced_int8_layers:
            layer_bits[name] = 8
            continue

        delta = deltas.get(name, 0.0)
        layer_bits[name] = 8 if delta > threshold else 4

    return layer_bits


def summarize_bit_allocation(layer_bits: Dict[str, int]):
    total_layers = len(layer_bits)
    int4_layers = sum(1 for b in layer_bits.values() if b == 4)
    int8_layers = sum(1 for b in layer_bits.values() if b == 8)
    lambda_int4 = (int4_layers / total_layers) if total_layers else 0.0
    effective_qw = lambda_int4 * 4.0 + (1.0 - lambda_int4) * 8.0
    return {
        "total_layers": total_layers,
        "int4_layers": int4_layers,
        "int8_layers": int8_layers,
        "lambda_int4": lambda_int4,
        "effective_qw": effective_qw,
    }


def mixed_precision_qat_fine_tune(
    fp32_model: nn.Module,
    layer_bits: Dict[str, int],
    trainloader,
    testloader,
    device: str,
):
    qat_model = copy.deepcopy(fp32_model)
    qat_model = replace_layers_with_mixed_qat(qat_model, layer_bits).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        qat_model.parameters(),
        lr=QAT_LR,
        momentum=QAT_MOMENTUM,
        weight_decay=QAT_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, QAT_EPOCHS))

    best_acc = -1.0
    best_state = None
    history = []

    for epoch in range(QAT_EPOCHS):
        qat_model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in trainloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = qat_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()

        train_loss = running_loss / max(1, len(trainloader))
        train_acc = 100.0 * correct / max(1, total)
        test_loss, test_acc = evaluate(qat_model, testloader, device)

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
            }
        )

        print(
            f"Mixed-QAT epoch {epoch + 1}/{QAT_EPOCHS} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.2f}% | "
            f"test_loss={test_loss:.4f} | test_acc={test_acc:.2f}%"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = copy.deepcopy(qat_model.state_dict())

    if best_state is not None:
        qat_model.load_state_dict(best_state)

    return qat_model, best_acc, history


def resolve_path(path_text: str):
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if MODEL_ARCH not in MODEL_BUILDERS:
        raise ValueError(f"MODEL_ARCH must be one of: {sorted(MODEL_BUILDERS.keys())}")

    checkpoint_path = resolve_path(CHECKPOINT_PATH)
    data_dir = resolve_path(DATA_DIR)
    results_path = resolve_path(RESULTS_JSON)
    save_dir = resolve_path(SAVE_DIR)
    tau_suffix = str(SENSITIVITY_THRESHOLD).replace(".", "p")

    # Avoid overwriting previous mixed-QAT runs when using the default results filename.
    if results_path.name == "qat_mixed_precision_results.json":
        results_path = results_path.with_name(
            f"qat_mixed_precision_results_tau_{tau_suffix}.json"
        )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Using device: {device}")
    print(f"Loading step3 checkpoint: {checkpoint_path}")

    trainloader = build_train_loader(data_dir, BATCH_SIZE_TRAIN, NUM_WORKERS)
    testloader = build_test_loader(data_dir, BATCH_SIZE_TEST, NUM_WORKERS)

    fp32_model = MODEL_BUILDERS[MODEL_ARCH]().to(device)
    fp32_model = load_checkpoint(fp32_model, checkpoint_path, device)

    fp32_loss, fp32_acc = evaluate(fp32_model, testloader, device)
    print(f"Step3 FP32 baseline -> loss={fp32_loss:.4f}, acc={fp32_acc:.2f}%")

    print("Running sensitivity analysis (one-layer INT4 probes over INT8 baseline)...")
    sens_baseline_acc, deltas, forced_int8_layers = analyze_layer_sensitivity(fp32_model, testloader, device)

    sweep_results = []
    for tau in SENSITIVITY_THRESHOLDS_SWEEP:
        layer_bits_tau = assign_bits_from_sensitivity(fp32_model, deltas, tau, forced_int8_layers)
        summary_tau = summarize_bit_allocation(layer_bits_tau)
        sweep_results.append(
            {
                "threshold": tau,
                **summary_tau,
            }
        )
        print(
            f"Threshold tau={tau:.2f}% -> INT4={summary_tau['int4_layers']}, "
            f"INT8={summary_tau['int8_layers']}, effective_qw={summary_tau['effective_qw']:.2f}"
        )

    layer_bits = assign_bits_from_sensitivity(
        fp32_model,
        deltas,
        SENSITIVITY_THRESHOLD,
        forced_int8_layers,
    )
    bit_summary = summarize_bit_allocation(layer_bits)

    print(
        f"Selected tau={SENSITIVITY_THRESHOLD:.2f}% -> INT4={bit_summary['int4_layers']}, "
        f"INT8={bit_summary['int8_layers']}, effective_qw={bit_summary['effective_qw']:.2f}"
    )

    print("Starting mixed-precision QAT fine-tuning...")
    qat_model, qat_best_acc, qat_history = mixed_precision_qat_fine_tune(
        fp32_model,
        layer_bits,
        trainloader,
        testloader,
        device,
    )

    final_loss, final_acc = evaluate(qat_model, testloader, device)

    save_dir.mkdir(parents=True, exist_ok=True)
    out_ckpt = save_dir / f"{checkpoint_path.stem}_mixed_int4_int8_qat_tau_{tau_suffix}.pth"

    torch.save(
        {
            "net": qat_model.state_dict(),
            "source_checkpoint": str(checkpoint_path),
            "source_model_arch": MODEL_ARCH,
            "quantization": "Sensitivity-guided mixed precision STE QAT",
            "forced_int8_layers": forced_int8_layers,
            "sensitivity_threshold": SENSITIVITY_THRESHOLD,
            "layer_bits": layer_bits,
            "sensitivity_deltas": deltas,
            "qat_epochs": QAT_EPOCHS,
            "qat_lr": QAT_LR,
            "qat_momentum": QAT_MOMENTUM,
            "qat_weight_decay": QAT_WEIGHT_DECAY,
            "fp32_test_loss": fp32_loss,
            "fp32_test_acc": fp32_acc,
            "sensitivity_baseline_acc": sens_baseline_acc,
            "mixed_best_test_acc": qat_best_acc,
            "mixed_final_test_loss": final_loss,
            "mixed_final_test_acc": final_acc,
            "bit_summary": bit_summary,
        },
        out_ckpt,
    )

    print(f"Saved mixed-precision QAT checkpoint to: {out_ckpt}")

    results = {
        "checkpoint": str(checkpoint_path),
        "model_arch": MODEL_ARCH,
        "device": device,
        "forced_int8_layers": forced_int8_layers,
        "sensitivity_threshold": SENSITIVITY_THRESHOLD,
        "threshold_sweep": sweep_results,
        "layer_bits": layer_bits,
        "sensitivity_deltas": deltas,
        "bit_summary": bit_summary,
        "fp32_test_loss": fp32_loss,
        "fp32_test_acc": fp32_acc,
        "mixed_best_test_acc": qat_best_acc,
        "mixed_final_test_loss": final_loss,
        "mixed_final_test_acc": final_acc,
        "acc_drop_from_fp32_to_mixed_final": fp32_acc - final_acc,
        "qat_epochs": QAT_EPOCHS,
        "qat_lr": QAT_LR,
        "qat_momentum": QAT_MOMENTUM,
        "qat_weight_decay": QAT_WEIGHT_DECAY,
        "history": qat_history,
        "saved_checkpoint": str(out_ckpt),
    }

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to: {results_path}")


if __name__ == "__main__":
    main()
