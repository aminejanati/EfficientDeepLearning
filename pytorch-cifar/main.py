'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import matplotlib.pyplot as plt
from datetime import datetime

from models import *
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--epochs', default=200, type=int, help='number of training epochs')
parser.add_argument('--save-name', default=None, type=str,
                    help='custom filename for the final best model (without extension)')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Lists to track metrics
train_losses = []
train_accs = []
test_losses = []
test_accs = []
best_epoch = 0
best_state = None
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Data
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

# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
#net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
#net = SimpleDLA()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0.0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    epoch_loss = train_loss/len(trainloader)
    epoch_acc = 100.*correct/total
    
    return epoch_loss, epoch_acc


def test(epoch):
    global best_acc, best_epoch, best_state
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    epoch_loss = test_loss/len(testloader)
    
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        best_state = state
        best_acc = acc
        best_epoch = epoch
    
    return epoch_loss, acc


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
        'net': net.state_dict(),
        'acc': best_acc,
        'epoch': start_epoch + args.epochs - 1,
    }

weights_dir = './Models Weights/baseline'
if not os.path.isdir(weights_dir):
    os.makedirs(weights_dir, exist_ok=True)

if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')

save_stem = args.save_name if args.save_name else f'resnet18_best_{timestamp}'
save_path = os.path.join(weights_dir, f'{save_stem}.pth')
torch.save(best_state, save_path)
torch.save(best_state, './checkpoint/ckpt.pth')

plots_root_dir = './Model Plots/baseline'
if not os.path.isdir(plots_root_dir):
    os.makedirs(plots_root_dir, exist_ok=True)

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
