from Model import Net
from DataSet import train_loader, test_loader
import torch.nn as nn
import argparse
from pathlib import Path
import torch.optim as optim
from torch.optim import lr_scheduler
import torch
import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

# python train.py --epochs 10 --num_classes 2
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of output classes")
    parser.add_argument("--save_path", type=str, default='./model.pth', help="Path to save the best model")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")

    args = parser.parse_args()

    epochs = args.epochs
    num_classes = args.num_classes
    save_path = args.save_path

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 初始化模型
    model = Net(num_classes=num_classes)
    model = model.to(device)

    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    lr = 1e-3

    # 优化器与学习率调度
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, verbose=True
    )

    # 损失函数
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 添加标签平滑

    # 训练记录
    records = {
        "train_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    best_acc = 0.0

    # 训练循环
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 30)

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # 混合精度前向传播
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            # 反向传播
            scaler.scale(loss).backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            # 统计指标
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # 计算训练指标
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        # 验证过程
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(images)
                _, predicted = outputs.max(1)

                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_acc = correct / total

        # 学习率调整
        scheduler.step(val_acc)

        # 记录历史
        records["train_loss"].append(train_loss)
        records["train_acc"].append(train_acc)
        records["val_acc"].append(val_acc)

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)

        # 打印统计信息
        print(f"--Train Loss: {train_loss:.4f}  --Train Acc: {train_acc:.4f}  --Val Acc: {val_acc:.4f}")

    # 可视化训练过程
    visualize_training(records, epochs)

def visualize_training(records, epochs):
    plt.figure(figsize=(12, 9))

    # 损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(np.arange(epochs), records["train_loss"], "b-", label="Training Loss")
    plt.title("Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)

    # 准确率曲线
    plt.subplot(2, 2, 2)
    plt.plot(np.arange(epochs), records["train_acc"], "g-", label="Training Accuracy")
    plt.plot(np.arange(epochs), records["val_acc"], "r-", label="Validation Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.show()

if __name__ == "__main__":
    main()
