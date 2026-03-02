import argparse
import copy
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision
import torchvision.transforms as transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import ResNet18
import train_augmentation


def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    use_amp = device == "cuda"

    with torch.inference_mode():
        for inputs, labels in test_loader:
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

    acc = 100.0 * correct / total
    print(f"Accuracy: {acc:.2f}%")
    return acc


def build_test_loader(batch_size=100, num_workers=2):
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    testset = torchvision.datasets.CIFAR10(
        root=str(PROJECT_ROOT / "data"),
        train=False,
        download=True,
        transform=transform_test,
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return testloader


def apply_unstructured_pruning(model, amount, module_types):
    for module in model.modules():
        if isinstance(module, module_types):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")


def save_plots(sweep_accuracies, histories, epochs, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)

    sweep_amounts = sorted(sweep_accuracies.keys())
    sweep_values = [sweep_accuracies[a] for a in sweep_amounts]

    plt.figure(figsize=(10, 5))
    plt.plot(sweep_amounts, sweep_values, marker="o")
    plt.xlabel("Unstructured Pruning Amount")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy vs Unstructured Weight Pruning Amount (Base Model 0.7 Structured Pruned)")
    plt.grid(True)
    sweep_plot_path = os.path.join(plot_dir, "sweep_accuracy.png")
    plt.savefig(sweep_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    epochs_range = range(1, epochs + 1)
    amounts = sorted(histories.keys())

    plt.figure(figsize=(10, 5))
    for amount in amounts:
        plt.plot(
            epochs_range,
            histories[amount]["test_accs"],
            marker="o",
            label=f"prune={amount}",
        )
    plt.xlabel("Fine-tuning Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Unstructured Pruning: Accuracy over Fine-tuning (Pretrained Base Model 0.7 Structured Pruned)")
    plt.grid(True)
    plt.legend()
    ft_acc_path = os.path.join(plot_dir, "finetune_accuracy.png")
    plt.savefig(ft_acc_path, dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    for amount in amounts:
        plt.plot(
            epochs_range,
            histories[amount]["test_losses"],
            marker="o",
            label=f"prune={amount}",
        )
    plt.xlabel("Fine-tuning Epoch")
    plt.ylabel("Test Loss")
    plt.title("Unstructured Pruning: Loss over Fine-tuning (Pretrained Base Model 0.7 Structured Pruned)")
    plt.grid(True)
    plt.legend()
    ft_loss_path = os.path.join(plot_dir, "finetune_loss.png")
    plt.savefig(ft_loss_path, dpi=150, bbox_inches="tight")
    plt.close()

    print("\nSaved plots:")
    print(f"- {sweep_plot_path}")
    print(f"- {ft_acc_path}")
    print(f"- {ft_loss_path}")


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path

    output_json_path = Path(args.output_json)
    if not output_json_path.is_absolute():
        output_json_path = PROJECT_ROOT / output_json_path

    plot_dir = Path(args.plot_dir)
    if not plot_dir.is_absolute():
        plot_dir = PROJECT_ROOT / plot_dir

    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    state_dict = checkpoint["net"]
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    base_model = ResNet18().to(device)
    base_model.load_state_dict(new_state_dict)
    base_model.half()

    testloader = build_test_loader(batch_size=args.eval_batch_size, num_workers=args.num_workers)

    module_types = (nn.Conv2d, nn.Linear)

    print("\n=== Baseline (before extra unstructured pruning) ===")
    evaluate(base_model, testloader, device)

    print("\n=== Sweep only (no finetune) ===")
    sweep_accuracies = {}
    for amount in args.sweep_amounts:
        model_pruned = copy.deepcopy(base_model)
        apply_unstructured_pruning(model_pruned, amount, module_types)
        acc = evaluate(model_pruned, testloader, device)
        sweep_accuracies[amount] = acc

    print("\n=== Finetune selected pruning levels ===")
    histories = {}
    for prune_amount in args.finetune_amounts:
        print(f"\n===== Unstructured pruning amount: {prune_amount} =====")
        model_to_train = copy.deepcopy(base_model)
        apply_unstructured_pruning(model_to_train, prune_amount, module_types)

        pre_ft_acc = evaluate(model_to_train, testloader, device)
        print(f"Pre-finetune accuracy ({prune_amount}): {pre_ft_acc:.2f}%")

        save_name = (
            f"Resnet18_unstructured_pruned_{int(prune_amount * 100)}_"
            f"basemodel_0.7structured_augmix_{args.epochs}ep"
        )

        results = train_augmentation.train_with_augmentation(
            model=model_to_train,
            epochs=args.epochs,
            resume=False,
            mixup=True,
            mixup_alpha=1.0,
            save_name=save_name,
            lr=args.lr,
            weights_subdir="unstructured_pruning",
            plots_subdir="unstructured_pruning",
        )

        histories[prune_amount] = {
            "pre_ft_acc": pre_ft_acc,
            "test_accs": results["test_accs"],
            "test_losses": results["test_losses"],
            "best_acc": results["best_acc"],
            "best_epoch": results["best_epoch"],
            "save_path": results["save_path"],
        }

        print(
            f"prune={prune_amount} | best_acc={results['best_acc']:.2f}% "
            f"at epoch {results['best_epoch']} | saved={results['save_path']}"
        )

    output = {
        "sweep_accuracies": sweep_accuracies,
        "finetune_histories": histories,
    }
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved summary to {output_json_path}")

    save_plots(
        sweep_accuracies=sweep_accuracies,
        histories=histories,
        epochs=args.epochs,
        plot_dir=str(plot_dir),
    )


def parse_float_list(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Run unstructured pruning + finetuning (base model: structured-pruned 0.7)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="Models Weights/structured_pruning/Resnet18_structured_pruned_70_augmix_25ep.pth",
        help="path to pretrained base checkpoint",
    )
    parser.add_argument("--epochs", type=int, default=30, help="finetuning epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--num-workers", type=int, default=2, help="dataloader workers")
    parser.add_argument(
        "--eval-batch-size", type=int, default=100, help="test batch size for evaluation"
    )
    parser.add_argument(
        "--sweep-amounts",
        type=parse_float_list,
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        help="comma-separated sweep amounts, e.g. 0,0.1,0.2",
    )
    parser.add_argument(
        "--finetune-amounts",
        type=parse_float_list,
        default=[0.1, 0.3, 0.5, 0.7],
        help="comma-separated finetune amounts, e.g. 0.1,0.3,0.5,0.7",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="unstructured_pruning_base0_7_results.json",
        help="output summary json path",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="Model Plots/unstructured_pruning/Resnet18_unstructured_pruned_basemodel_0.7structured_augmix",
        help="directory to save generated plots",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()