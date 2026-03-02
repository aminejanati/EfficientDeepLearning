'''Train CIFAR10 with PyTorch (augmentations + MixUp).'''
import os
import argparse
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from models import ResNet18
from utils import progress_bar


def _prepare_data(batch_size_train=128, batch_size_test=100, num_workers=2):
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
        trainset, batch_size=batch_size_train, shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size_test, shuffle=False, num_workers=num_workers)

    return trainloader, testloader


def _mixup_data(inputs, targets, alpha):
    if alpha <= 0:
        return inputs, targets, targets, 1.0
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    batch_size = inputs.size(0)
    index = torch.randperm(batch_size, device=inputs.device)
    mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
    targets_a = targets
    targets_b = targets[index]
    return mixed_inputs, targets_a, targets_b, lam


def _save_plots(train_accs, test_accs, train_losses, test_losses, best_epoch, best_acc, start_epoch, epochs, model_plot_dir):
    print('\n==> Plotting training history...')
    epochs_range = range(start_epoch, start_epoch + epochs)

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
    plot_path = os.path.join(model_plot_dir, 'training_history_augmix.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f'Training history plot saved to {plot_path}')
    plt.show()


def train_with_augmentation(
    epochs=200,
    lr=0.1,
    resume=False,
    mixup=False,
    mixup_alpha=1.0,
    save_name=None,
    model=None,
    weights_subdir='baseline',
    plots_subdir='baseline',
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0
    start_epoch = 0

    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    best_epoch = 0
    best_state = None

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    trainloader, testloader = _prepare_data()

    print('==> Building model..')
    net = model if model is not None else ResNet18()
    net = net.to(device)
    if device == 'cuda' and not isinstance(net, torch.nn.DataParallel):
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model_dtype = next(net.parameters()).dtype

    def train_one_epoch(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0.0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.to(model_dtype)
            optimizer.zero_grad()

            if mixup:
                inputs, targets_a, targets_b, lam = _mixup_data(inputs, targets, mixup_alpha)
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
            if mixup:
                correct += lam * predicted.eq(targets_a).sum().item()
                correct += (1 - lam) * predicted.eq(targets_b).sum().item()
            else:
                correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        epoch_loss = train_loss/len(trainloader)
        epoch_acc = 100.*correct/total

        return epoch_loss, epoch_acc

    def test_one_epoch(epoch):
        nonlocal best_acc, best_epoch, best_state
        net.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                inputs = inputs.to(model_dtype)
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

    for epoch in range(start_epoch, start_epoch + epochs):
        train_loss, train_acc = train_one_epoch(epoch)
        test_loss, test_acc = test_one_epoch(epoch)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        scheduler.step()

    if best_state is None:
        best_state = {
            'net': net.state_dict(),
            'acc': best_acc,
            'epoch': start_epoch + epochs - 1,
        }

    weights_dir = os.path.join('./Models Weights', weights_subdir)
    if not os.path.isdir(weights_dir):
        os.makedirs(weights_dir, exist_ok=True)

    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')

    save_stem = save_name if save_name else f'resnet18_augmix_best_{timestamp}'
    save_path = os.path.join(weights_dir, f'{save_stem}.pth')
    torch.save(best_state, save_path)
    torch.save(best_state, './checkpoint/ckpt.pth')

    plots_root_dir = os.path.join('./Model Plots', plots_subdir)
    if not os.path.isdir(plots_root_dir):
        os.makedirs(plots_root_dir, exist_ok=True)

    model_plot_dir = os.path.join(plots_root_dir, save_stem)
    os.makedirs(model_plot_dir, exist_ok=True)

    """"
    _save_plots(
        train_accs,
        test_accs,
        train_losses,
        test_losses,
        best_epoch,
        best_acc,
        start_epoch,
        epochs,
        model_plot_dir,
    )
    """
    print(f'Best model achieved {best_acc:.2f}% accuracy at epoch {best_epoch}')
    print(f'Best model saved at {save_path}')

    return {
        'best_acc': best_acc,
        'best_epoch': best_epoch,
        'save_path': save_path,
        'model': net,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'train_losses': train_losses,
        'test_losses': test_losses,
    }


def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training (Augmentations + MixUp)')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--epochs', default=200, type=int, help='number of training epochs')
    parser.add_argument('--mixup', action='store_true',
                        help='enable MixUp data augmentation')
    parser.add_argument('--mixup-alpha', default=1.0, type=float,
                        help='MixUp beta distribution alpha')
    parser.add_argument('--save-name', default=None, type=str,
                        help='custom filename for the final best model (without extension)')
    args = parser.parse_args()

    train_with_augmentation(
        epochs=args.epochs,
        lr=args.lr,
        resume=args.resume,
        mixup=args.mixup,
        mixup_alpha=args.mixup_alpha,
        save_name=args.save_name,
    )


if __name__ == '__main__':
    main()