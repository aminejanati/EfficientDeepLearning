'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
from datetime import datetime
import matplotlib.pyplot as plt
from models import *
from utils import progress_bar


# ----------------------
# Training configuration
# ----------------------
LEARNING_RATE = 0.1
RESUME_FROM_CHECKPOINT = False
EPOCHS = 200
SAVE_NAME = "resnet18_g2"
RUN_NAME = datetime.now().strftime('%Y%m%d_%H%M%S')

# Distillation configuration
USE_DISTILLATION = True
DISTILL_ALPHA = 0.5
DISTILL_TEMPERATURE = 4.0
TEACHER_CKPT_PATH = './Models Weights/baseline/Resnet18_V2.pth'

TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 100
NUM_WORKERS = 2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

weights_dir = 'Models Weights/step1'
if not os.path.isdir(weights_dir):
    os.makedirs(weights_dir, exist_ok=True)

save_stem = SAVE_NAME if SAVE_NAME else f'{RUN_NAME}'
save_path = os.path.join(weights_dir, f'{save_stem}.pth')

# Lists to track metrics
train_losses = []
train_accs = []
test_losses = []
test_accs = []
best_epoch = 0
best_state = None

# Data
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
    trainset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


def build_student_model():
    return ResNet18_GroupedG2()


def build_teacher_model():
    return ResNet18()


def load_model_checkpoint(model, ckpt_path, current_device):
    checkpoint = torch.load(ckpt_path, map_location=current_device)
    state_dict = checkpoint['net'] if isinstance(checkpoint, dict) and 'net' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    return model

# Model
print('==> Building model..')
net = build_student_model()
# net = ResNet18_GroupedG4()
# net = ResNet18_GroupedG8()
# net = ResNet18_Depthwise()

teacher_net = None
if USE_DISTILLATION:
    print('==> Building teacher model for distillation..')
    assert os.path.isfile(TEACHER_CKPT_PATH), f'Error: teacher checkpoint not found at {TEACHER_CKPT_PATH}'
    teacher_net = build_teacher_model().to(device)
    teacher_net = load_model_checkpoint(teacher_net, TEACHER_CKPT_PATH, device)
    teacher_net.eval()
    for param in teacher_net.parameters():
        param.requires_grad = False

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    if teacher_net is not None:
        teacher_net = torch.nn.DataParallel(teacher_net)
        teacher_net.eval()

if RESUME_FROM_CHECKPOINT:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(save_path), f'Error: checkpoint not found at {save_path}'
    checkpoint = torch.load(save_path, map_location=device)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1

criterion = nn.CrossEntropyLoss()
distill_criterion = nn.KLDivLoss(reduction='batchmean')
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    train_ce_loss = 0
    train_kd_loss = 0
    correct = 0.0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = net(inputs)

        ce_loss = criterion(outputs, targets)
        kd_loss = torch.tensor(0.0, device=device)
        loss = ce_loss
        if USE_DISTILLATION and teacher_net is not None:
            with torch.no_grad():
                teacher_outputs = teacher_net(inputs)
            temp = DISTILL_TEMPERATURE
            kd_loss = distill_criterion(
                F.log_softmax(outputs / temp, dim=1),
                F.softmax(teacher_outputs / temp, dim=1)
            ) * (temp * temp)
            loss = DISTILL_ALPHA * ce_loss + (1.0 - DISTILL_ALPHA) * kd_loss

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_ce_loss += ce_loss.item()
        train_kd_loss += kd_loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if USE_DISTILLATION:
            progress_bar(
                batch_idx,
                len(trainloader),
                'Loss: %.3f | CE: %.3f | KD: %.3f | Acc: %.3f%% (%d/%d)'
                % (
                    train_loss/(batch_idx+1),
                    train_ce_loss/(batch_idx+1),
                    train_kd_loss/(batch_idx+1),
                    100.*correct/total,
                    correct,
                    total,
                ),
            )
        else:
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


for epoch in range(start_epoch, start_epoch + EPOCHS):
    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)
    
    # Record metrics
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)

    # Temporary checkpoint: single file overwritten each epoch.
    epoch_state = {
        'net': net.state_dict(),
        'acc': test_acc,
        'epoch': epoch,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'test_loss': test_loss,
        'test_acc': test_acc,
    }
    torch.save(epoch_state, save_path)
    
    scheduler.step()

if best_state is None:
    best_state = {
        'net': net.state_dict(),
        'acc': best_acc,
        'epoch': start_epoch + EPOCHS - 1,
    }

