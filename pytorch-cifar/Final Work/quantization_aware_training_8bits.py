"""Run 8-bit quantization-aware training (QAT) on a step3 checkpoint."""

import copy
import json
import sys
from pathlib import Path

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
# This should point to the unstructured-pruned fine-tuned checkpoint produced in step3.
CHECKPOINT_PATH = "Final Weights/step3/resnet18_depthwise_structured_0p7_finetuned_unstructured_0p7_finetuned.pth"
MODEL_ARCH = "resnet18_depthwise"
QAT_EPOCHS = 30
QAT_LR = 0.001
QAT_WEIGHT_DECAY = 5e-4
QAT_MOMENTUM = 0.9
BATCH_SIZE_TRAIN = 128
BATCH_SIZE_TEST = 100
NUM_WORKERS = 2
DATA_DIR = "data"
RESULTS_JSON = "Final Weights/step4/qat_8bit_results.json"
SAVE_DIR = "Final Weights/step4"


class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def fake_quantize(tensor, num_bits=8, symmetric=True):
    if symmetric:
        qmax = (2 ** (num_bits - 1)) - 1
        qmin = -(2 ** (num_bits - 1))
        max_val = tensor.abs().max().clamp(min=1e-5)
        scale = max_val / qmax

        q_tensor = RoundSTE.apply(tensor / scale)
        q_tensor = torch.clamp(q_tensor, qmin, qmax)
        return q_tensor * scale

    qmax = (2 ** num_bits) - 1
    qmin = 0
    min_val = tensor.min()
    max_val = tensor.max()
    scale = (max_val - min_val).clamp(min=1e-5) / qmax

    q_tensor = RoundSTE.apply((tensor - min_val) / scale)
    q_tensor = torch.clamp(q_tensor, qmin, qmax)
    return q_tensor * scale + min_val


class QATConv2d8Bit(nn.Conv2d):
    def forward(self, input):
        quantized_input = fake_quantize(input, num_bits=8, symmetric=False)
        quantized_weight = fake_quantize(self.weight, num_bits=8, symmetric=True)
        return F.conv2d(
            quantized_input,
            quantized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class QATLinear8Bit(nn.Linear):
    def forward(self, input):
        quantized_input = fake_quantize(input, num_bits=8, symmetric=False)
        quantized_weight = fake_quantize(self.weight, num_bits=8, symmetric=True)
        return F.linear(quantized_input, quantized_weight, self.bias)


def replace_layers_with_qat_8bit(model):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            replace_layers_with_qat_8bit(module)

        if isinstance(module, nn.Conv2d):
            qat_conv = QATConv2d8Bit(
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                module.stride,
                module.padding,
                module.dilation,
                module.groups,
                module.bias is not None,
                module.padding_mode,
            )
            qat_conv.weight.data = module.weight.data.clone()
            if module.bias is not None:
                qat_conv.bias.data = module.bias.data.clone()
            model._modules[name] = qat_conv

        elif isinstance(module, nn.Linear):
            qat_linear = QATLinear8Bit(
                module.in_features,
                module.out_features,
                module.bias is not None,
            )
            qat_linear.weight.data = module.weight.data.clone()
            if module.bias is not None:
                qat_linear.bias.data = module.bias.data.clone()
            model._modules[name] = qat_linear

    return model


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


def qat_fine_tune_8bit(fp32_model: nn.Module, trainloader, testloader, device: str):
    qat_model = copy.deepcopy(fp32_model)
    qat_model = replace_layers_with_qat_8bit(qat_model).to(device)

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
            f"QAT-8 epoch {epoch + 1}/{QAT_EPOCHS} | "
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

    print("Starting 8-bit QAT fine-tuning...")
    qat_model, qat_best_acc, qat_history = qat_fine_tune_8bit(
        fp32_model,
        trainloader,
        testloader,
        device,
    )

    final_loss, final_acc = evaluate(qat_model, testloader, device)

    save_dir.mkdir(parents=True, exist_ok=True)
    out_ckpt = save_dir / f"{checkpoint_path.stem}_int8_qat.pth"
    torch.save(
        {
            "net": qat_model.state_dict(),
            "source_checkpoint": str(checkpoint_path),
            "source_model_arch": MODEL_ARCH,
            "quantization": "Custom STE 8-bit",
            "qat_epochs": QAT_EPOCHS,
            "qat_lr": QAT_LR,
            "qat_momentum": QAT_MOMENTUM,
            "qat_weight_decay": QAT_WEIGHT_DECAY,
            "fp32_test_loss": fp32_loss,
            "fp32_test_acc": fp32_acc,
            "int8_best_test_acc": qat_best_acc,
            "int8_final_test_loss": final_loss,
            "int8_final_test_acc": final_acc,
        },
        out_ckpt,
    )
    print(f"Saved QAT checkpoint to: {out_ckpt}")

    results = {
        "checkpoint": str(checkpoint_path),
        "model_arch": MODEL_ARCH,
        "device": device,
        "fp32_test_loss": fp32_loss,
        "fp32_test_acc": fp32_acc,
        "int8_best_test_acc": qat_best_acc,
        "int8_final_test_loss": final_loss,
        "int8_final_test_acc": final_acc,
        "acc_drop_from_fp32_to_int8_final": fp32_acc - final_acc,
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
