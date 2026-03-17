"""Run unstructured pruning on a structured-pruned distilled CIFAR-10 model."""

import copy
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
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
# This should point to the structured-pruned distilled checkpoint produced in step2.
CHECKPOINT_PATH = "Final Weights/step2/resnet18_depthwise_structured_0p7_finetuned.pth"
MODEL_ARCH = "resnet18_depthwise"
PRUNING_AMOUNT = 0.7
PRUNE_CONV = True
PRUNE_LINEAR = True
FINETUNE_EPOCHS = 30
FINETUNE_LR = 0.01
FINETUNE_WEIGHT_DECAY = 5e-4
FINETUNE_MOMENTUM = 0.9
BATCH_SIZE_TRAIN = 128
BATCH_SIZE_TEST = 100
NUM_WORKERS = 2
DATA_DIR = "data"
RESULTS_JSON = "Final Weights/step3/unstructured_pruning_results.json"
SAVE_DIR = "Final Weights/step3"


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
    model.eval()
    correct = 0
    total = 0
    use_amp = device == "cuda"

    with torch.inference_mode():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(inputs)
            else:
                outputs = model(inputs)

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return 100.0 * correct / total


def fine_tune(model: nn.Module, trainloader, testloader, device: str, epochs: int, lr: float):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=FINETUNE_MOMENTUM,
        weight_decay=FINETUNE_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    use_amp = device == "cuda"

    history = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in trainloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()

        train_loss = running_loss / len(trainloader)
        train_acc = 100.0 * correct / total
        test_acc = evaluate(model, testloader, device)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
            }
        )
        print(
            f"Fine-tune epoch {epoch + 1}/{epochs} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.2f}% | test_acc={test_acc:.2f}%"
        )

    return history


def apply_unstructured_pruning(model: nn.Module, amount: float):
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and PRUNE_CONV:
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")
        if isinstance(module, nn.Linear) and PRUNE_LINEAR:
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")


def count_zeroed_weights(model: nn.Module):
    zeroed = 0
    total = 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and PRUNE_CONV:
            weights = module.weight.detach()
            zeroed += (weights == 0).sum().item()
            total += weights.numel()
        if isinstance(module, nn.Linear) and PRUNE_LINEAR:
            weights = module.weight.detach()
            zeroed += (weights == 0).sum().item()
            total += weights.numel()
    return int(zeroed), int(total)


def resolve_path(path_text: str):
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if MODEL_ARCH not in MODEL_BUILDERS:
        raise ValueError(f"MODEL_ARCH must be one of: {sorted(MODEL_BUILDERS.keys())}")
    if PRUNING_AMOUNT < 0.0 or PRUNING_AMOUNT >= 1.0:
        raise ValueError(f"Invalid pruning amount {PRUNING_AMOUNT}. Use values in [0.0, 1.0).")

    checkpoint_path = resolve_path(CHECKPOINT_PATH)
    data_dir = resolve_path(DATA_DIR)
    results_path = resolve_path(RESULTS_JSON)
    save_dir = resolve_path(SAVE_DIR)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Using device: {device}")
    print(f"Loading checkpoint: {checkpoint_path}")

    trainloader = build_train_loader(data_dir, BATCH_SIZE_TRAIN, NUM_WORKERS)
    testloader = build_test_loader(data_dir, BATCH_SIZE_TEST, NUM_WORKERS)

    model = MODEL_BUILDERS[MODEL_ARCH]().to(device)
    model = load_checkpoint(model, checkpoint_path, device)

    baseline_acc = evaluate(model, testloader, device)
    print(f"Baseline accuracy: {baseline_acc:.2f}%")

    results = {
        "checkpoint": str(checkpoint_path),
        "model_arch": MODEL_ARCH,
        "device": device,
        "baseline_accuracy": baseline_acc,
        "pruning_type": "l1_unstructured",
        "pruning_amount": PRUNING_AMOUNT,
        "prune_conv": PRUNE_CONV,
        "prune_linear": PRUNE_LINEAR,
        "runs": [],
    }

    amount = PRUNING_AMOUNT
    pruned_model = copy.deepcopy(model)
    apply_unstructured_pruning(pruned_model, amount=amount)
    pruned_acc_before_ft = evaluate(pruned_model, testloader, device)
    zeroed_weights, total_weights = count_zeroed_weights(pruned_model)
    zeroed_ratio = (100.0 * zeroed_weights / total_weights) if total_weights else 0.0

    print("Starting post-pruning fine-tuning...")
    finetune_history = fine_tune(
        pruned_model,
        trainloader,
        testloader,
        device,
        epochs=FINETUNE_EPOCHS,
        lr=FINETUNE_LR,
    )
    pruned_acc_after_ft = finetune_history[-1]["test_acc"] if finetune_history else pruned_acc_before_ft

    run_result = {
        "amount": amount,
        "accuracy_before_finetune": pruned_acc_before_ft,
        "accuracy_after_finetune": pruned_acc_after_ft,
        "zeroed_weights": zeroed_weights,
        "total_weights": total_weights,
        "zeroed_ratio_percent": zeroed_ratio,
        "finetune_epochs": FINETUNE_EPOCHS,
        "finetune_lr": FINETUNE_LR,
        "finetune_history": finetune_history,
    }
    results["runs"].append(run_result)

    print(
        f"amount={amount:.2f} | acc_before_ft={pruned_acc_before_ft:.2f}% | "
        f"acc_after_ft={pruned_acc_after_ft:.2f}% | "
        f"zeroed_weights={zeroed_weights}/{total_weights} ({zeroed_ratio:.2f}%)"
    )

    save_dir.mkdir(parents=True, exist_ok=True)
    suffix = str(amount).replace(".", "p")
    ckpt_name = f"{checkpoint_path.stem}_unstructured_{suffix}_finetuned.pth"
    out_ckpt = save_dir / ckpt_name
    torch.save(
        {
            "net": pruned_model.state_dict(),
            "base_checkpoint": str(checkpoint_path),
            "amount": amount,
            "accuracy_before_finetune": pruned_acc_before_ft,
            "accuracy_after_finetune": pruned_acc_after_ft,
            "pruning_type": "l1_unstructured",
            "prune_conv": PRUNE_CONV,
            "prune_linear": PRUNE_LINEAR,
            "finetune_epochs": FINETUNE_EPOCHS,
            "finetune_lr": FINETUNE_LR,
        },
        out_ckpt,
    )
    print(f"Saved fine-tuned pruned checkpoint to: {out_ckpt}")

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to: {results_path}")


if __name__ == "__main__":
    main()