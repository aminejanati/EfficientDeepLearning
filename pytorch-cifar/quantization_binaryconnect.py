import argparse
from datetime import datetime
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn



from models import ResNet18
from utils import progress_bar

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import binaryconnect


def normalize_state_dict_for_model(state_dict, model):
    is_data_parallel = isinstance(model, torch.nn.DataParallel)
    normalized = {}
    for key, value in state_dict.items():
        if is_data_parallel and not key.startswith('module.'):
            normalized[f'module.{key}'] = value
        elif not is_data_parallel and key.startswith('module.'):
            normalized[key.replace('module.', '', 1)] = value
        else:
            normalized[key] = value
    return normalized

# Default hyperparameters    
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--epochs', default=200, type=int, help='number of training epochs')
parser.add_argument('--save-name', default=None, type=str,
                    help='custom filename for the final best model (without extension)')
args = parser.parse_args()

# device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ResNet18().to(device)
if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

# Load pretrained weights
checkpoint = torch.load('Models Weights/Resnet18_V0.pth', map_location=device)
state_dict = checkpoint['net']
# Remove 'module.' if using DataParallel
new_state_dict = normalize_state_dict_for_model(state_dict, model)
model.load_state_dict(new_state_dict)

# Lists to track metrics
train_losses = []
train_accs = []
test_losses = []
test_accs = []
start_epoch = 0
best_acc = 0.0
best_epoch = 0
best_state = None

# Preparing data
print('==> Preparing data..')
#transforms.RandomCrop(32, padding=4),
#transforms.RandomHorizontalFlip(),
transform_train = transforms.Compose([
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

# resume training if specified
if args.resume:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth', map_location=device)
    resume_state = checkpoint.get('net', checkpoint)
    resume_state = normalize_state_dict_for_model(resume_state, model)
    model.load_state_dict(resume_state)
    best_acc = checkpoint.get('acc', checkpoint.get('best_acc', 0.0))
    start_epoch = checkpoint['epoch']

# model hyperparameters
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)  
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

# BinaryConnect wrapper
modelbc = binaryconnect.BC(model)
modelbc.model = modelbc.model.to(device)

# function to evaluate the model 
def evaluate(model, test_loader):
    model.eval()
    correct, total = 0, 0
    model_dtype = next(model.parameters()).dtype
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device=device, dtype=model_dtype)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    acc = 100 * correct / total
    print(f"Model Accuracy: {acc:.2f}%")
    return acc

# Funcion to train the model for one epoch
def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0.0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        modelbc.binarization()
        outputs = modelbc.model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        modelbc.restore()
        optimizer.step()
        modelbc.clip()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return train_loss / len(trainloader), 100. * correct / total
        
def test(epoch):
    global best_acc, best_epoch, best_state
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100 * correct / total
    print(f"Test Accuracy: {acc:.2f}%")

    avg_loss = test_loss / len(testloader)
    if acc > best_acc:
        best_acc = acc
        best_epoch = epoch
        best_state = {
            'net': model.state_dict(),
            'acc': best_acc,
            'epoch': epoch,
        }

    return avg_loss, acc

for epoch in range(start_epoch, start_epoch + args.epochs):
    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)
    
    # Record metrics
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    
    scheduler.step()

if best_state is None:
    best_state = {
        'net': model.state_dict(),
        'acc': best_acc,
        'epoch': start_epoch + args.epochs - 1,
    }

weights_dir = './Models Weights'
if not os.path.isdir(weights_dir):
    os.mkdir(weights_dir)

if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
save_stem = args.save_name if args.save_name else f'resnet18_best_{timestamp}'
save_path = os.path.join(weights_dir, f'{save_stem}.pth')
torch.save(best_state, save_path)
torch.save(best_state, './checkpoint/ckpt.pth')

plots_root_dir = './Model Plots'
if not os.path.isdir(plots_root_dir):
    os.mkdir(plots_root_dir)

model_plot_dir = os.path.join(plots_root_dir, save_stem)
os.makedirs(model_plot_dir, exist_ok=True)

# Plot training history
print('\n==> Plotting training history...')
epochs_range = range(start_epoch, start_epoch + args.epochs)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot accuracy
ax1.plot(epochs_range, train_accs, 'b-', label='Train Accuracy', linewidth=2)
ax1.plot(epochs_range, test_accs, 'r-', label='Test Accuracy', linewidth=2)
ax1.axvline(x=best_epoch, color='g', linestyle='--', linewidth=1.5, label=f'Best Model (Epoch {best_epoch})')
ax1.axhline(y=best_acc, color='g', linestyle=':', linewidth=1, alpha=0.7)
ax1.scatter([best_epoch], [best_acc], color='green', s=100, zorder=5, marker='*')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_title('Training and Test Accuracy', fontsize=14, fontweight='bold')
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)
ax1.text(best_epoch, best_acc, f' {best_acc:.2f}%', fontsize=10, color='green', 
         verticalalignment='bottom', fontweight='bold')

# Plot loss
ax2.plot(epochs_range, train_losses, 'b-', label='Train Loss', linewidth=2)
ax2.plot(epochs_range, test_losses, 'r-', label='Test Loss', linewidth=2)
ax2.axvline(x=best_epoch, color='g', linestyle='--', linewidth=1.5, label=f'Best Model (Epoch {best_epoch})')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.set_title('Training and Test Loss', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(model_plot_dir, 'training_history.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f'Training history plot saved to {plot_path}')
print(f'Best model achieved {best_acc:.2f}% accuracy at epoch {best_epoch}')
print(f'Best model saved at {save_path}')
plt.show()


