import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=48, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            # 第一个全连接层，输入特征为 4608，输出特征为 2048
            nn.Linear(in_features=4608, out_features=2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            # 第二个全连接层，输入特征为 2048，输出特征为 2048
            nn.Linear(in_features=2048, out_features=2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            # 第三个全连接层，输入特征为 2048，输出特征为 1000
            nn.Linear(in_features=2048, out_features=1000),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            # 最终的全连接层，输入特征为 1000，输出特征为 num_classes
            nn.Linear(in_features=1000, out_features=num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x