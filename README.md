# NarutoAndSasuke_recognizition

## Introduction
This project is based on CNN(AlexNet) to classify Naruto and Sasuke pictures.

## Requirements
```python
pip install -r requirements.txt
```

## Train Models
```python
python train.py --epoch 15 --num_classes 2 --save_path ./model.pth
```
* `--epoch` The number of training rounds.
* `--num_classes` Number of picture types.
* `--save_path` Customize the path to save the model.

You can customize the above parameters.

## Get predictions
```python
python predict.py --img_path 'The path of the picture that you want to predict' --model_path ./model.pth
```
* `--img_path` The path of the picture that you want to predict
* `--model_path` The path of the already trained model.

You can customize the above parameters.