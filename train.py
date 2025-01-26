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

warnings.filterwarnings("ignore")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--num_classes", type=int)
    parser.add_argument("--save_path", type=str, default='./model.pth')

    args = parser.parse_args()

    # python train.py --epoch 15 --num_classes 2 --save_path ./model.pth

    epochs = args.epoch
    num_classes = args.num_classes
    save_path = Path(args.save_path)

    # 加载模型
    model = Net(num_classes=2)
    model = model.cuda()

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []


    # 定义损失函数和优化器
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    max_accuracy = 0

    for epoch in range(epochs):
        # 训练模型
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()

            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = correct / total
        train_loss = running_loss / len(train_loader)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_accuracy)

        # 测试模型
        model.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.cuda()
                labels = labels.cuda()

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_accuracy = correct / total
        test_acc_list.append(test_accuracy)

        if test_accuracy > max_accuracy:
            max_accuracy = test_accuracy
            torch.save(model.state_dict(), save_path)

        print("Epoch {}: --Training loss: {:.4f} --Training accuracy: {:.4f} --Test accuracy: {:.4f}".format(epoch+1,
                                                                                                    train_loss, train_accuracy, test_accuracy))

        # 更新学习率
        lr_scheduler.step()

    # 保存训练好的模型
    torch.save(model.state_dict(), save_path)

    # 训练结果可视化
    x = ['{}'.format(i + 1) for i in range(epochs)]
    plt.figure(figsize=(10, 8), dpi=80)
    plt.xlabel("训练轮数")
    plt.xlabel("Epoch")
    plt.plot(x, train_loss_list, color='blue')
    plt.plot(x, train_acc_list, color='yellow')
    plt.plot(x, test_acc_list, color='red')
    plt.legend(["Train Loss", "Train Accuracy", "Test Accuracy"])
    plt.grid(alpha=0.5)
    plt.show()