'''Train CIFAR10 with PyTorch (augmentations + MixUp).'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from models import *
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training (Augmentations + MixUp)')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--epochs', default=200, type=int, help='number of training epochs')
parser.add_argument('--mixup', action='store_true',
                    help='enable MixUp data augmentation')
parser.add_argument('--mixup-alpha', default=1.0, type=float,
                    help='MixUp beta distribution alpha')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0
start_epoch = 0

train_losses = []
train_accs = []
test_losses = []
test_accs = []
best_epoch = 0

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter(f'runs/cifar10_resnet18_augmix_{timestamp}')
print(f'==> TensorBoard logging to: runs/cifar10_resnet18_augmix_{timestamp}')
print('==> Start TensorBoard with: tensorboard --logdir=runs')

print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.RandomGrayscale(p=0.1),
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

print('==> Building model..')
net = ResNet18()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
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


def mixup_data(inputs, targets, alpha):
    if alpha <= 0:
        return inputs, targets, targets, 1.0
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    batch_size = inputs.size(0)
    index = torch.randperm(batch_size, device=inputs.device)
    mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
    targets_a = targets
    targets_b = targets[index]
    return mixed_inputs, targets_a, targets_b, lam


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0.0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        if args.mixup:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, args.mixup_alpha)
            outputs = net(inputs)
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        else:
            outputs = net(inputs)
            loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        if args.mixup:
            correct += lam * predicted.eq(targets_a).sum().item()
            correct += (1 - lam) * predicted.eq(targets_b).sum().item()
        else:
            correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    epoch_loss = train_loss/len(trainloader)
    epoch_acc = 100.*correct/total

    writer.add_scalar('Loss/train', epoch_loss, epoch)
    writer.add_scalar('Accuracy/train', epoch_acc, epoch)

    return epoch_loss, epoch_acc


def test(epoch):
    global best_acc, best_epoch
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

    acc = 100.*correct/total
    epoch_loss = test_loss/len(testloader)

    writer.add_scalar('Loss/test', epoch_loss, epoch)
    writer.add_scalar('Accuracy/test', acc, epoch)

    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
        best_epoch = epoch
        writer.add_scalar('Best/accuracy', best_acc, epoch)

    return epoch_loss, acc


for epoch in range(start_epoch, start_epoch + args.epochs):
    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)

    scheduler.step()

print('\n==> Plotting training history...')
epochs_range = range(start_epoch, start_epoch + args.epochs)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

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

ax2.plot(epochs_range, train_losses, 'b-', label='Train Loss', linewidth=2)
ax2.plot(epochs_range, test_losses, 'r-', label='Test Loss', linewidth=2)
ax2.axvline(x=best_epoch, color='g', linestyle='--', linewidth=1.5, label=f'Best Model (Epoch {best_epoch})')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.set_title('Training and Test Loss', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./checkpoint/training_history_augmix.png', dpi=150, bbox_inches='tight')
print('Training history plot saved to ./checkpoint/training_history_augmix.png')
print(f'Best model achieved {best_acc:.2f}% accuracy at epoch {best_epoch}')
print('Best model saved at ./checkpoint/ckpt.pth')
plt.show()

writer.close()
print('\n==> TensorBoard logs saved. View with: tensorboard --logdir=runs')
