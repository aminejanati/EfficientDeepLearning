import torch
import torch.nn.utils.prune as prune
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


from models import ResNet18



# Load ResNet model
model = ResNet18()
checkpoint = torch.load('Models Weights/Resnet18_V0.pth', map_location='cuda')
state_dict = checkpoint['net']

# Remove 'module.' if using DataParallel
new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)

# Move model to GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# Preparing data
print('==> Preparing data..')
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

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


def evaluate(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.half()  # Convert input to half precision
            outputs = model(inputs)  # Model is already in half precision
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    acc = 100 * correct / total
    print(f"Half Precision Model Accuracy: {acc:.2f}%")
    return acc


# Use model.half to quantize the model
model.half()

# Evaluate the quantized model
evaluate(model, testloader)

new_acc = evaluate(model, testloader)

# Save the quantized model with the same state structure
state = {
    'net': model.state_dict(),  # Save model weights
    'best_acc': new_acc,        # Save the accuracy obtained after quantization
    'hyperparams': {
        'learning_rate': 0.1,  # Replace with your hyperparameters if needed
        'batch_size': 128,
        'momentum': 0.9,
        'quantization': 'FP16'  # Add a flag to indicate half precision
    }
}

torch.save(state, 'Models/q_factorized2.pth')

print("Quantized model saved successfully.")