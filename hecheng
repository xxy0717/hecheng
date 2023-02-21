import os
import cv2
import dlib
import numpy as np
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# 人脸识别模型
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


# 提取面部特征
def get_landmarks(image):
    rects = detector(image, 1)
    if len(rects) == 0:
        return None
    return np.matrix([[p.x, p.y] for p in predictor(image, rects[0]).parts()])


# 根据面部特征生成孩子面部特征
def generate_child_feature(mother_landmarks, father_landmarks):
    child_landmarks = np.zeros((68, 2))
    for i in range(68):
        if i < 30:
            child_landmarks[i] = mother_landmarks[i]
        else:
            child_landmarks[i] = father_landmarks[i]
    return child_landmarks


# 生成孩子照片
def generate_child_photo(mother_img, father_img):
    # 读取父亲、母亲的照片
    mother = cv2.imread(mother_img)
    father = cv2.imread(father_img)

    # 提取面部特征
    mother_landmarks = get_landmarks(mother)
    father_landmarks = get_landmarks(father)

    # 生成孩子面部特征
    child_landmarks = generate_child_feature(mother_landmarks, father_landmarks)

    # 生成孩子照片
    child = np.zeros(mother.shape, dtype=np.uint8)
    hull = cv2.convexHull(np.array(child_landmarks), returnPoints=False)
    cv2.fillConvexPoly(child, hull, (255, 255, 255))
    mask = np.zeros(mother.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, (255, 255, 255))
    mother_face = cv2.bitwise_and(mother, mask)
    child_face =
