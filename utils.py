import os
import shutil
import json
import random

from sklearn.model_selection import train_test_split


def save_classes_indices(path):

    classes = [cla for cla in os.listdir(path) if os.path.isdir(os.path.join(path, cla))]

    classes.sort()

    class_indices = dict((k, v) for v, k in enumerate(classes))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

def split_images(path, data_dir, dataset_dir):
    save_classes_indices(path)
    # 获取所有类别
    train_dir = os.path.join(dataset_dir, 'train')
    test_dir = os.path.join(dataset_dir, 'test')

    # 创建train和test文件夹
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 获取所有类别（文件夹名称）
    categories = [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]
    # print(categories)
    # 遍历所有类别
    for category in categories:
        category_path = os.path.join(data_dir, category)

        # 创建每个类别在train和test中的子文件夹
        category_train_dir = os.path.join(train_dir, category)
        category_test_dir = os.path.join(test_dir, category)
        os.makedirs(category_train_dir, exist_ok=True)
        os.makedirs(category_test_dir, exist_ok=True)

        # 获取类别中的所有图片文件
        images = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]
        print(images)
        # 随机打乱图片列表
        random.shuffle(images)

        # 设置训练集和测试集比例
        split_ratio = 0.8
        split_index = int(len(images) * split_ratio)

        # 将图片分为训练集和测试集
        train_images, test_images = train_test_split(images, test_size=0.2, random_state=10)
        # 移动图片到训练集和测试集对应的文件夹
        for img in train_images:
            src_img_path = os.path.join(category_path, img)
            dst_img_path = os.path.join(category_train_dir, img)
            shutil.copy(src_img_path, dst_img_path)

        for img in test_images:
            src_img_path = os.path.join(category_path, img)
            dst_img_path = os.path.join(category_test_dir, img)
            shutil.copy(src_img_path, dst_img_path)

# split_images('./Data', 'Data', 'Dataset')