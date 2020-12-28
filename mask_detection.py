import matplotlib.pyplot as plt
import cv2
import sys
import tensorflow as tf
import numpy as np
from keras.models import load_model
from mtcnn.mtcnn import MTCNN
def initializationGPU():
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

def img_segmentation(img_path, img_size):
    img_dic = []
    detector = MTCNN()
    img = cv2.imread(img_path)
    faces = detector.detect_faces(img)
    print(len(faces))
    for face in faces:
        x, y, width, height = face['box']
        img = cv2.imread(img_path)
        crop_img = img[y:y+height, x:x+width]
        re_img = cv2.resize(crop_img, img_size)
        re_img = re_img/255
        img_dic.append(re_img)
    return img_dic, faces

def mark_face_pic(img_path, face_obj, label):
    img = cv2.imread(img_path)
    category2color = {0: (0, 255, 0), 1: (0, 0, 255), 2: (0, 165, 255)}
    for i,face in enumerate(face_obj):
        x, y, width, height = face['box']
        cv2.rectangle(img , (x, y), (x+width, y+height), category2color[label[i]], 5)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

initializationGPU()
model = load_model("Mask_detection_AI.h5")
img_dic,faces = img_segmentation(sys.argv[1],(64,64))
mark_face_pic(sys.argv[1],faces,np.argmax(model.predict(np.array(img_dic)), axis=1))