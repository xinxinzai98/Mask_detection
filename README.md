# Mask_detection
人脸口罩检测，基于[Kaggle](https://www.kaggle.com/andrewmvd/face-mask-detection)上的开放人脸数据库
## 文件结构
* `input`训练数据集
    * `annotation`照片中人脸位置与是否佩戴口罩的标记
    * `imgae`训练图像
* `model`已经训练好模型权重
* `environment.yml`conda环境文件
* `build_model.py`训练模型
* `mask_detection.py`预测文件
## 模型结构
## 运行方法
利用miniconda创建环境
```
conda env create -f environment.yml -n mask_detection
```
### 训练模型
你可以训练自己模型
```
python build_model.py
```
运行文件会被保存在`model`文件夹下
### 加载模型
加载已经训练完成的模型，val_accuracy = 0.89
```
python mask_detection.py /path/to/your/img
```
## 测试集曲线
