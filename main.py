import matplotlib
matplotlib.use('TkAgg')  # 指定后端
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import os
import multiprocessing
import numpy as np
import random

# Memory Optimization
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()

# 数据预处理
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])


# 轻量化CNN模型
class LightCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def denormalize(tensor):
    """将归一化的张量转换为可显示的图片格式"""
    mean = torch.tensor([0.4914, 0.4822, 0.4465])
    std = torch.tensor([0.2023, 0.1994, 0.2010])
    return tensor * std[:, None, None] + mean[:, None, None]


def predict_from_batch(model, batch_data, class_names, device):
    """从测试批次中随机选择样本进行预测"""
    images, labels = batch_data
    idx = random.randint(0, images.size(0) - 1)

    model.eval()
    with torch.no_grad():
        inputs = images[idx].unsqueeze(0).to(device)
        outputs = model(inputs)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
        _, predicted = torch.max(outputs.data, 1)

    return (
        images[idx],  # 原始图像张量
        labels[idx].item(),  # 真实标签
        predicted.item(),  # 预测结果
        probabilities[predicted].item()  # 置信度
    )


if __name__ == '__main__':
    multiprocessing.freeze_support()

    # 初始化配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # 数据加载
    batch_size = 128
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 模型初始化
    model = LightCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # 训练记录
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    # 训练循环
    for epoch in range(30):
        model.train()
        epoch_train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            epoch_train_loss += loss.item()

            if torch.cuda.is_available():
                mem = torch.cuda.memory_allocated() / 1024 ** 2
                print(f'Epoch {epoch + 1} 显存占用: {mem:.2f}MB', end='\r')

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_loss += loss.item()

        # 记录指标
        train_loss = epoch_train_loss / len(trainloader)
        train_acc = 100 * train_correct / train_total
        val_loss = val_loss / len(testloader)
        val_acc = 100 * val_correct / val_total

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f'Epoch {epoch + 1}: '
              f'Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | '
              f'Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%')

    # 可视化训练曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'r-', label='Training Loss')
    plt.plot(val_losses, 'b-', label='Validation Loss')
    plt.title('Loss vs Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, 'r-', label='Training Accuracy')
    plt.plot(val_accuracies, 'b-', label='Validation Accuracy')
    plt.title('Accuracy vs Epochs')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

    # 从测试集随机选择样本进行预测
    test_batch = next(iter(testloader))
    img_tensor, true_label, pred_label, confidence = predict_from_batch(model, test_batch, class_names, device)

    # 反归一化并显示结果
    img_np = denormalize(img_tensor).numpy().transpose(1, 2, 0)
    img_np = np.clip(img_np, 0, 1)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img_np)
    plt.title(f'True: {class_names[true_label]}')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    result_text = (f'Predicted: {class_names[pred_label]}\n'
                   f'Confidence: {confidence:.2f}%\n'
                   f'Status: {"✓" if pred_label == true_label else "✗"}')
    plt.text(0.1, 0.5, result_text,
             fontsize=12,
             bbox=dict(facecolor='white', alpha=0.9),
             color='green' if pred_label == true_label else 'red')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # 保存模型
    torch.save(model.state_dict(), 'auto_test_model.pth')
    print("模型已保存为 auto_test_model.pth")