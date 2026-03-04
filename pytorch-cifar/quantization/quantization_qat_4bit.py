import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import ResNet18

DEFAULT_WEIGHTS_DIR = PROJECT_ROOT / "Models Weights" / "unstructured_pruning"
DEFAULT_INT4_DIR = PROJECT_ROOT / "Models Weights" / "quantization" / "unstructured_pruning_qat_4bit"
DEFAULT_RESULTS_JSON = PROJECT_ROOT / "quantization" / "unstructured_pruning_qat_4bit_results.json"

# ---------------------------------------------------------------------------
# 1. Custom 4-bit Fake Quantization with Straight-Through Estimator (STE)
# ---------------------------------------------------------------------------

class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-Through Estimator: pass gradients unchanged
        return grad_output

def fake_quantize(tensor, num_bits=4, symmetric=True):
    if symmetric:
        qmax = (2 ** (num_bits - 1)) - 1
        qmin = -(2 ** (num_bits - 1))
        # Find absolute max for scaling
        max_val = tensor.abs().max().clamp(min=1e-5)
        scale = max_val / qmax
        
        # Quantize and Dequantize
        q_tensor = RoundSTE.apply(tensor / scale)
        q_tensor = torch.clamp(q_tensor, qmin, qmax)
        return q_tensor * scale
    else:
        qmax = (2 ** num_bits) - 1
        qmin = 0
        min_val = tensor.min()
        max_val = tensor.max()
        
        scale = (max_val - min_val).clamp(min=1e-5) / qmax
        
        # Quantize and Dequantize
        q_tensor = RoundSTE.apply((tensor - min_val) / scale)
        q_tensor = torch.clamp(q_tensor, qmin, qmax)
        return q_tensor * scale + min_val

class QATConv2d(nn.Conv2d):
    def forward(self, input):
        # 4-bit Asymmetric activations, 4-bit Symmetric weights
        quantized_input = fake_quantize(input, num_bits=4, symmetric=False)
        quantized_weight = fake_quantize(self.weight, num_bits=4, symmetric=True)
        
        return F.conv2d(
            quantized_input, quantized_weight, self.bias, 
            self.stride, self.padding, self.dilation, self.groups
        )

class QATLinear(nn.Linear):
    def forward(self, input):
        quantized_input = fake_quantize(input, num_bits=4, symmetric=False)
        quantized_weight = fake_quantize(self.weight, num_bits=4, symmetric=True)
        
        return F.linear(quantized_input, quantized_weight, self.bias)

def replace_layers_with_qat(model):
    """Recursively replaces standard layers with our custom 4-bit QAT layers."""
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            replace_layers_with_qat(module)
        
        if isinstance(module, nn.Conv2d):
            qat_conv = QATConv2d(
                module.in_channels, module.out_channels, module.kernel_size,
                module.stride, module.padding, module.dilation, module.groups,
                module.bias is not None, module.padding_mode
            )
            qat_conv.weight.data = module.weight.data.clone()
            if module.bias is not None:
                qat_conv.bias.data = module.bias.data.clone()
            model._modules[name] = qat_conv
            
        elif isinstance(module, nn.Linear):
            qat_linear = QATLinear(
                module.in_features, module.out_features, module.bias is not None
            )
            qat_linear.weight.data = module.weight.data.clone()
            if module.bias is not None:
                qat_linear.bias.data = module.bias.data.clone()
            model._modules[name] = qat_linear
    return model

# ---------------------------------------------------------------------------
# 2. Standard Training & Evaluation Utilities
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Reliable 4-bit QAT using STE.")
    parser.add_argument("--weights-dir", type=Path, default=DEFAULT_WEIGHTS_DIR)
    parser.add_argument("--glob", type=str, default="*.pth")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_INT4_DIR)
    parser.add_argument("--results-json", type=Path, default=DEFAULT_RESULTS_JSON)
    parser.add_argument("--batch-size-train", type=int, default=128)
    parser.add_argument("--batch-size-test", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--qat-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--train-device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    return parser.parse_args()

def build_data_loaders(batch_size_train=128, batch_size_test=100, num_workers=2):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root=str(PROJECT_ROOT / "data"), train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=True, num_workers=num_workers)
    testset = torchvision.datasets.CIFAR10(root=str(PROJECT_ROOT / "data"), train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader

def load_checkpoint_to_model(checkpoint_path):
    model = ResNet18()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["net"] if isinstance(checkpoint, dict) and "net" in checkpoint else checkpoint
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    return model

def evaluate_model(model, data_loader, device):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return total_loss / max(1, len(data_loader)), 100.0 * correct / max(1, total)

# ---------------------------------------------------------------------------
# 3. QAT Execution 
# ---------------------------------------------------------------------------

def qat_finetune_4bit(fp32_model, train_loader, test_loader, train_device, args):
    # Swap layers out for our custom 4-bit QAT layers
    qat_model = copy.deepcopy(fp32_model)
    qat_model = replace_layers_with_qat(qat_model).to(train_device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(qat_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.qat_epochs))

    best_acc = -1.0
    best_state = None

    for epoch in range(args.qat_epochs):
        qat_model.train()
        train_loss, correct, total = 0.0, 0, 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device=train_device), targets.to(device=train_device)
            optimizer.zero_grad()
            outputs = qat_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        scheduler.step()
        
        train_acc = 100.0 * correct / total
        test_loss, test_acc = evaluate_model(qat_model, test_loader, train_device)
        
        print(f"    Epoch {epoch + 1}/{args.qat_epochs} | "
              f"Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}% | "
              f"Test Acc: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = copy.deepcopy(qat_model.state_dict())

    if best_state is not None:
        qat_model.load_state_dict(best_state)

    return qat_model, best_acc

def main():
    args = parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.results_json.parent.mkdir(parents=True, exist_ok=True)
    
    train_loader, test_loader = build_data_loaders(args.batch_size_train, args.batch_size_test, args.num_workers)

    results = []
    checkpoint_paths = sorted(args.weights_dir.glob(args.glob))
    
    for ckpt_path in checkpoint_paths:
        print(f"\n==> Processing: {ckpt_path.name}")
        fp32_model = load_checkpoint_to_model(ckpt_path).to(args.train_device)
        fp32_loss, fp32_acc = evaluate_model(fp32_model, test_loader, args.train_device)
        print(f"  FP32 Test -> loss: {fp32_loss:.4f}, acc: {fp32_acc:.2f}%")

        qat_model, qat_acc = qat_finetune_4bit(fp32_model, train_loader, test_loader, args.train_device, args)
        
        save_name = ckpt_path.stem + "_int4_qat.pth"
        save_path = args.output_dir / save_name
        
        torch.save({
            "net": qat_model.state_dict(),
            "source_checkpoint": str(ckpt_path),
            "quantization": "Custom STE 4-bit",
            "qat_epochs": args.qat_epochs,
            "fp32_test_acc": fp32_acc,
            "int4_test_acc": qat_acc,
        }, save_path)
        
        results.append({
            "checkpoint": ckpt_path.name,
            "fp32_test_acc": round(fp32_acc, 4),
            "int4_test_acc": round(qat_acc, 4),
            "acc_drop": round(fp32_acc - qat_acc, 4),
        })

    with open(args.results_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved summary to: {args.results_json}")

if __name__ == "__main__":
    main()