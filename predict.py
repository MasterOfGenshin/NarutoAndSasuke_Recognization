import matplotlib.pyplot as plt
import torch
from sympy.unify.core import Variable
import numpy
import os
import json
import argparse
from pathlib import Path
from Model import Net
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable

plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str)
    parser.add_argument("--model_path", type=str)

    # python predict.py --img_path ./Data/Sasuke/10021.jpg --model_path model.pth

    args= parser.parse_args()

    img_path = Path(args.img_path)
    model_path = Path(args.model_path)
    json_path = 'class_indices.json'

    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        classes = json.load(f)
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Net(num_classes=2).to(device)
    model.eval()

    # 加载训练好的模型
    model.load_state_dict(torch.load(model_path))

    img = Image.open(img_path) # 打开图片
    img = img.convert('RGB') # 转换为RGB格式

    predict_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # 将图片转换为Tensor
    img_tensor = predict_transform(img)
    img_tensor = Variable(torch.unsqueeze(img_tensor, dim=0).float(), requires_grad=False).to(device)

    # 测试模式
    model.eval()

    with torch.no_grad():
        outputs_tensor = model(img_tensor)
        outputs = torch.softmax(outputs_tensor, dim=1)

        predict, index = torch.max(outputs, 1)

        predict = predict.detach().cpu().numpy()
        index = index.detach().cpu().numpy()
        predict_classes = classes[str(index[0])]
        predict_prob = predict[0] * 100

        print("Classes: {} Prob: {:.4f}%".format(predict_classes, predict_prob))

        # 展示预测结果
        plt.imshow(img)
        plt.title("Classes: {}     Prob: {:.2f}%".format(predict_classes, predict_prob))
        plt.xticks([])
        plt.yticks([])
        plt.show()