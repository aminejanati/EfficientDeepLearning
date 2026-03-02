import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import ResNet18
from torch.ao.quantization import QConfig, QConfigMapping
from torch.ao.quantization.observer import HistogramObserver, PerChannelMinMaxObserver
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx


DEFAULT_WEIGHTS_DIR = PROJECT_ROOT / "Models Weights" / "unstructured_pruning"
DEFAULT_INT4_DIR = PROJECT_ROOT / "Models Weights" / "quantization" / "unstructured_pruning_ptq_4bit"
DEFAULT_RESULTS_JSON = PROJECT_ROOT / "quantization" / "unstructured_pruning_ptq_4bit_results.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Post-training 4-bit-range quantization (no fine-tuning) for all unstructured-pruned checkpoints."
    )
    parser.add_argument("--weights-dir", type=Path, default=DEFAULT_WEIGHTS_DIR)
    parser.add_argument("--glob", type=str, default="*.pth")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_INT4_DIR)
    parser.add_argument("--results-json", type=Path, default=DEFAULT_RESULTS_JSON)
    parser.add_argument("--batch-size-calib", type=int, default=128)
    parser.add_argument("--batch-size-test", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--calib-batches", type=int, default=100)
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "fbgemm", "x86", "qnnpack"])
    parser.add_argument("--eval-device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--max-models", type=int, default=0, help="0 means process all models.")
    return parser.parse_args()


def build_data_loaders(batch_size_calib=128, batch_size_test=100, num_workers=2):
    print("==> Preparing CIFAR-10 data for calibration/evaluation...")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=str(PROJECT_ROOT / "data"),
        train=True,
        download=True,
        transform=transform,
    )
    calib_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size_calib,
        shuffle=False,
        num_workers=num_workers,
    )

    testset = torchvision.datasets.CIFAR10(
        root=str(PROJECT_ROOT / "data"),
        train=False,
        download=True,
        transform=transform,
    )
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size_test,
        shuffle=False,
        num_workers=num_workers,
    )

    return calib_loader, test_loader


def load_checkpoint_to_model(checkpoint_path):
    model = ResNet18()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["net"] if isinstance(checkpoint, dict) and "net" in checkpoint else checkpoint
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [warn] Missing keys in {checkpoint_path.name}: {missing}")
    if unexpected:
        print(f"  [warn] Unexpected keys in {checkpoint_path.name}: {unexpected}")

    return model


def evaluate_model(model, data_loader, device):
    criterion = nn.CrossEntropyLoss()
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / max(1, len(data_loader))
    acc = 100.0 * correct / max(1, total)
    return avg_loss, acc


def select_quant_backend(requested_backend):
    supported_engines = torch.backends.quantized.supported_engines

    if requested_backend != "auto":
        if requested_backend not in supported_engines:
            raise RuntimeError(
                f"Requested backend '{requested_backend}' is not supported on this machine. "
                f"Available: {supported_engines}"
            )
        return requested_backend

    for backend in ("fbgemm", "x86", "qnnpack"):
        if backend in supported_engines:
            return backend

    raise RuntimeError(f"No supported quantization backend found. Available: {supported_engines}")


def make_4bit_qconfig_mapping():
    activation_observer = HistogramObserver.with_args(
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
        quant_min=0,
        quant_max=15,
    )

    weight_observer = PerChannelMinMaxObserver.with_args(
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        quant_min=-8,
        quant_max=7,
    )

    qconfig_4bit = QConfig(activation=activation_observer, weight=weight_observer)
    return QConfigMapping().set_global(qconfig_4bit)


def ptq_convert_no_finetune_4bit(fp32_model, calib_loader, backend, calib_batches=100):
    torch.backends.quantized.engine = backend
    qconfig_mapping = make_4bit_qconfig_mapping()

    fp32_cpu = copy.deepcopy(fp32_model).cpu().eval()
    example_inputs = (torch.randn(1, 3, 32, 32),)
    prepared_model = prepare_fx(fp32_cpu, qconfig_mapping, example_inputs)

    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(calib_loader):
            if batch_idx >= calib_batches:
                break
            prepared_model(inputs)

    quantized_model = convert_fx(prepared_model)
    return quantized_model


def process_checkpoint(checkpoint_path, calib_loader, test_loader, eval_device, backend, output_dir, calib_batches):
    print(f"\n==> Processing: {checkpoint_path.name}")
    fp32_model = load_checkpoint_to_model(checkpoint_path).to(eval_device)

    fp32_loss, fp32_acc = evaluate_model(fp32_model, test_loader, eval_device)
    print(f"  FP32 Test -> loss: {fp32_loss:.4f}, acc: {fp32_acc:.2f}%")

    quantized_model = ptq_convert_no_finetune_4bit(fp32_model, calib_loader, backend, calib_batches=calib_batches)
    quant_loss, quant_acc = evaluate_model(quantized_model, test_loader, "cpu")
    print(f"  4-bit PTQ Test -> loss: {quant_loss:.4f}, acc: {quant_acc:.2f}%")

    save_name = checkpoint_path.stem + "_int4_ptq_nofinetune.pth"
    save_path = output_dir / save_name
    torch.save(
        {
            "net": quantized_model.state_dict(),
            "source_checkpoint": str(checkpoint_path),
            "quantization": "PTQ 4-bit range (no fine-tuning)",
            "backend": backend,
            "calib_batches": calib_batches,
            "fp32_test_acc": fp32_acc,
            "int4_test_acc": quant_acc,
        },
        save_path,
    )

    return {
        "checkpoint": checkpoint_path.name,
        "int4_checkpoint": save_name,
        "fp32_test_loss": round(fp32_loss, 6),
        "fp32_test_acc": round(fp32_acc, 4),
        "int4_test_loss": round(quant_loss, 6),
        "int4_test_acc": round(quant_acc, 4),
        "acc_drop": round(fp32_acc - quant_acc, 4),
    }


def main():
    args = parse_args()

    weights_dir = args.weights_dir
    output_dir = args.output_dir
    results_json = args.results_json

    if not weights_dir.exists():
        raise FileNotFoundError(f"Weights directory does not exist: {weights_dir}")

    checkpoint_paths = sorted(weights_dir.glob(args.glob))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints found in {weights_dir} with pattern '{args.glob}'")

    if args.max_models > 0:
        checkpoint_paths = checkpoint_paths[: args.max_models]

    output_dir.mkdir(parents=True, exist_ok=True)
    results_json.parent.mkdir(parents=True, exist_ok=True)

    backend = select_quant_backend(args.backend)
    print(f"Using quantization backend: {backend}")
    print(f"Evaluation device for FP32: {args.eval_device}")
    print(f"No fine-tuning: calibration only ({args.calib_batches} batches)")

    calib_loader, test_loader = build_data_loaders(
        batch_size_calib=args.batch_size_calib,
        batch_size_test=args.batch_size_test,
        num_workers=args.num_workers,
    )

    results = []
    for checkpoint_path in checkpoint_paths:
        result = process_checkpoint(
            checkpoint_path=checkpoint_path,
            calib_loader=calib_loader,
            test_loader=test_loader,
            eval_device=args.eval_device,
            backend=backend,
            output_dir=output_dir,
            calib_batches=args.calib_batches,
        )
        results.append(result)

    summary = {
        "weights_dir": str(weights_dir),
        "output_dir": str(output_dir),
        "results_count": len(results),
        "backend": backend,
        "calib_batches": args.calib_batches,
        "no_finetuning": True,
        "bit_width": 4,
        "results": results,
    }

    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved summary: {results_json}")
    if results:
        avg_fp32 = sum(r["fp32_test_acc"] for r in results) / len(results)
        avg_int4 = sum(r["int4_test_acc"] for r in results) / len(results)
        avg_drop = avg_fp32 - avg_int4
        print(f"Average FP32 acc : {avg_fp32:.2f}%")
        print(f"Average INT4 acc : {avg_int4:.2f}%")
        print(f"Average acc drop : {avg_drop:.2f}%")


if __name__ == "__main__":
    main()
