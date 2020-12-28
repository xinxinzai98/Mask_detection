import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle as pkl
import cv2
import os
import seaborn as sns

from plotly import graph_objects as go
from plotly import express as px
from xml.etree import ElementTree as et
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from collections import Counter
from keras.models import Sequential
from keras.layers.experimental import preprocessing as ps
from keras.layers import Activation, Conv2D, BatchNormalization, Dense, Flatten, Dropout, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.losses import CategoricalCrossentropy
from sklearn.metrics import confusion_matrix

# 初始化gpu加速计算
def initialization():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(
                logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

# 读取数据
def read_data(annotations_path, data_path):
    data = []
    for dir, _, files in os.walk(annotations_path):
        for file_ in files:
            dict_ = dict(img_path=None, objs=[])
            path = os.path.join(dir, file_)
            tree = et.parse(path)
            dict_['img_path'] = os.path.join(
                data_path, tree.find('filename').text)
            for obj in tree.findall('object'):
                label = obj.find('name').text
                xmin = int(obj.find('bndbox/xmin').text)
                ymin = int(obj.find('bndbox/ymin').text)
                xmax = int(obj.find('bndbox/xmax').text)
                ymax = int(obj.find('bndbox/ymax').text)
                dict_['objs'].append(
                    [xmin, ymin, xmax, ymax, label2category[label]])
            data.append(dict_)
    return data


def augment_data(input_img, aument_model, iterate):
    img_list = []
    for _ in range(iterate):
        batch = tf.expand_dims(input_img, 0)
        aug = aument_model(batch)
        img_list.append(np.array(aug[0]))
    return img_list

# 图像切割
def img_segmentation(img_obj_set, img_size):
    img_dic = []
    label_dic = []
    for img_obj in img_obj_set:
        img_path = img_obj['img_path']
        for (xmin, ymin, xmax, ymax, label) in img_obj['objs']:
            img = cv2.imread(img_path)
            crop_img = img[ymin: ymax, xmin: xmax]
            re_img = cv2.resize(crop_img, img_size)
            re_img = re_img/255
            img_dic.append(re_img)
            label_dic.append(label)
    return img_dic, label_dic

# 图像扩充
def img_augment(img_dic, label_dic):
    x = []
    y = []

    aug_model = Sequential()
    aug_model.add(ps.RandomFlip())
    aug_model.add(ps.RandomRotation(0.4))

    for i in range(len(img_dic)):
        target = to_categorical(label_dic[i], num_classes=3)
        if label_dic[i] == 2:
            aug_img = augment_data(img_dic[i], aug_model, 10)
            for aug in aug_img:
                x.append(np.array(aug))
                y.append(target)
        elif label_dic[i] == 1:
            aug_img = augment_data(img_dic[i], aug_model, 3)
            for aug in aug_img:
                x.append(np.array(aug))
                y.append(target)
        else:
            x.append(img_dic[i])
            y.append(target)
    return x, y

# CNN模型
def build_model():
    model = Sequential()

    model.add(
        Conv2D(32, (3, 3), input_shape=x_train.shape[1:], activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(1, 1), padding='same'))
    model.add(Dropout(0.2))

    model.add(
        Conv2D(64, (3, 3), input_shape=x_train.shape[1:], activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(1, 1), padding='same'))
    model.add(Dropout(0.2))

    model.add(
        Conv2D(128, (3, 3), input_shape=x_train.shape[1:], activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(1, 1), padding='same'))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 图片展示
def display_pic(pic_set, window_plot_x, window_plot_y):
    for i in range(len(pic_set)):
        img = pic_set[i]
        title = "img"+str(i+1)
        plt.subplot(window_plot_x, window_plot_y, i+1)
        plt.imshow(img)
        plt.title(title, fontsize=8)
        plt.xticks([])
        plt.yticks([])
    plt.show()


initialization()
annotations_path = './input/annotations'
data_path = './input/images'
label2category = {'with_mask': 0,
                  'without_mask': 1, 'mask_weared_incorrect': 2}

data = read_data(annotations_path, data_path)
img_dic, label_dic = img_segmentation(data, (64, 64))
x, y = img_augment(img_dic, label_dic)
x_train, x_test, y_train, y_test = train_test_split(np.array(x), np.array(y), test_size=0.2, random_state=7)

model = build_model()
checkpoint = ModelCheckpoint('Mask_detection_AI.h5', monitor='val_accuracy', verbose=1, save_best_only=True,save_weights_only=True, mode='max')
earlystop = EarlyStopping(monitor='val_accuracy', patience=10)
history = model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test), callbacks=[checkpoint,earlystop])
model.evaluate(x_test, y_test)