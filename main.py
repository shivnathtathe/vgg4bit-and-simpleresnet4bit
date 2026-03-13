### Copyright (c) 2025 Shivnath Tathe. All rights reserved.
#### This code is licensed for private academic and research use only.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math

# Symmetric quantization with gradient clipping

def quantize_symmetric(tensor, bits=4):
    """Symmetric quantization with gradient clipping"""
    n_levels = 2**(bits-1) - 1  # 7 for 4-bit signed

    # Find scale using abs max
    abs_max = tensor.abs().max()
    abs_max = abs_max + 1e-8  # Prevent division by zero

    scale = abs_max / n_levels

    # Quantize
    tensor_q = torch.round(tensor / scale).clamp(-n_levels, n_levels)

    # Dequantize
    tensor_deq = tensor_q * scale

    return tensor_deq, scale

# 4-bit Convolution Layer
class Conv4bit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize with proper scaling for conv layers
        n = in_channels * kernel_size * kernel_size
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * math.sqrt(2.0 / n))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

    def forward(self, x):
        # Straight-through estimator
        w_quant, _ = quantize_symmetric(self.weight, bits=4)
        w_ste = self.weight + (w_quant - self.weight).detach()
        return F.conv2d(x, w_ste, self.bias, self.stride, self.padding)

# 4-bit Linear Layer
class Linear4bit(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * math.sqrt(1.0 / in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x):
        w_quant, _ = quantize_symmetric(self.weight, bits=4)
        w_ste = self.weight + (w_quant - self.weight).detach()
        return F.linear(x, w_ste, self.bias)

#4-bit VGG Network
class VGG4bit(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1
            Conv4bit(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            Conv4bit(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2
            Conv4bit(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            Conv4bit(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3
            Conv4bit(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            Conv4bit(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Classifier
        self.classifier = nn.Sequential(
            Linear4bit(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            Linear4bit(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Simple ResNet 4-bit Network
class SimpleResNet4bit(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Initial convolution
        self.conv1 = Conv4bit(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        # Residual blocks (simplified)
        self.conv2 = Conv4bit(16, 16, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = Conv4bit(16, 32, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = Conv4bit(32, 32, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = Conv4bit(32, 64, 3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = Conv4bit(64, 64, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(64)

        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = Linear4bit(64, num_classes)

    def forward(self, x):
        # Initial conv
        out = F.relu(self.bn1(self.conv1(x)))

        # Block 1
        identity = out
        out = F.relu(self.bn2(self.conv2(out)))
        out = out + identity  # Residual connection

        # Block 2 (downsample)
        out = F.relu(self.bn3(self.conv3(out)))
        out = F.relu(self.bn4(self.conv4(out)))

        # Block 3 (downsample)
        out = F.relu(self.bn5(self.conv5(out)))
        out = F.relu(self.bn6(self.conv6(out)))

        # Classifier
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

#Quantization-Aware Adam with Warmup
class QuantAwareAdamW(torch.optim.AdamW):
    def __init__(self, params, lr=0.001, weight_decay=5e-4, grad_clip=1.0):
        super().__init__(params, lr=lr, weight_decay=weight_decay)
        self.grad_clip = grad_clip

    def step(self):
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.param_groups[0]['params'], self.grad_clip)
        super().step()

        # Soft weight clipping after update
        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    if len(p.shape) >= 2:  # Weight matrices
                        p.data = torch.tanh(p.data / 3.0) * 3.0

#Training & Testing Functions
def train_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)

        if torch.isnan(loss):
            print(f"NaN detected at batch {batch_idx}!")
            return float('inf')

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        if batch_idx % 50 == 0:
            print(f'  Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}, '
                  f'Acc: {100.*correct/total:.2f}%')

    return total_loss / len(loader), 100. * correct / total

def test(model, loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    test_loss /= len(loader.dataset)
    accuracy = 100. * correct / len(loader.dataset)
    return test_loss, accuracy

#Main Training Script
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Data augmentation for CIFAR-10
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Load CIFAR-10
    train_dataset = datasets.CIFAR100('./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100('./data', train=False, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

    # Model selection
    print('\n=== 4-bit Training on CIFAR-100 ===')
    print('Choose model:')
    print('1. SimpleResNet4bit (lighter, faster)')
    print('2. VGG4bit (heavier, potentially more accurate)')

    model_choice = input('Enter choice (1 or 2): ').strip()

    if model_choice == '2':
        model = VGG4bit(num_classes=100).to(device)
        model_name = 'VGG4bit'
    else:
        model = SimpleResNet4bit(num_classes=100).to(device)
        model_name = 'SimpleResNet4bit'

    print(f'\nUsing {model_name}')

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params:,}')

    # Optimizer with warmup
    optimizer = QuantAwareAdamW(model.parameters(), lr=0.001, weight_decay=5e-4, grad_clip=0.5)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # Training loop
    best_acc = 0
    checkpoint_path = '/content/vgg4bit_checkpoint.pth'
    for epoch in range(150):  # Less epochs for initial testing
        print(f'\n=== Epoch {epoch+1}/150 (LR: {scheduler.get_last_lr()[0]:.6f}) ===')

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, epoch)

        if train_loss == float('inf'):
            print("Training failed!")
            break

        test_loss, test_acc = test(model, test_loader, device)

        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

        # Check quantization statistics
        with torch.no_grad():
            unique_counts = []
            for name, module in model.named_modules():
                if isinstance(module, (Conv4bit, Linear4bit)):
                    w_quant, scale = quantize_symmetric(module.weight, bits=4)
                    unique = torch.unique(torch.round(module.weight / scale))
                    unique_counts.append(len(unique))

            print(f'Unique weight values per layer: min={min(unique_counts)}, '
                  f'max={max(unique_counts)}, avg={sum(unique_counts)/len(unique_counts):.1f}')

        best_acc = max(best_acc, test_acc)
        scheduler.step()
        if test_acc > 95:
            print(f"\nReached {test_acc:.2f}% accuracy! Exceptional for 4-bit training.")
            break

    print(f'\n=== Final Results ===')
    print(f'Best Test Accuracy: {best_acc:.2f}%')

    # Memory comparison im MB
    float32_memory = total_params * 4 / (1024**2)
    int4_memory = total_params * 0.5 / (1024**2)
    print(f'\nMemory Usage:')
    print(f'Float32: {float32_memory:.2f} MB')
    print(f'Int4: {int4_memory:.2f} MB')
    print(f'Compression ratio: {float32_memory/int4_memory:.1f}x')

if __name__ == '__main__':
    main()