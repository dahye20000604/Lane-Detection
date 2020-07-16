import os
import cv2
import numpy as np
import time
from threading import Thread
import platform
import tflite_runtime.interpreter as tflite

cap = cv2.VideoCapture('./video/curve.avi')  # 0: default camera
# cap = cv2.VideoCapture("test.mp4") #동영상 파일에서 읽기

ret = cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
ret = cap.set(3, 640)
ret = cap.set(4, 480)

check, frame = cap.read()
modeldir = '.\models\mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
# MODEL_NAME = modeldir
# Name of the .tflite file, if different than detect.tflite
# I'm not sure why use graph_name
# GRAPH_NAME = 'detect.tflite'
# GRAPH_NAME = 'edgetpu.tflite'
use_TPU = True
# Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.

# resolution = '1920x1080'
resolution = '640x480'

resW, resH = resolution.split('x')
imW, imH = int(resW), int(resH)
# Minimum confidence threshold for displaying detected objects
min_conf_threshold = 0.5
'''
PATH_TO_LABELS = './models/coco_labels2.txt'
# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del (labels[0])

# Get model details
model_path = modeldir
interpreter = tflite.Interpreter(
    model_path=model_path,
    # experimental_delegates=[tflite.load_delegate(EDGETPU_SHARED_LIB)])
    experimental_delegates=[tflite.load_delegate('edgetpu.dll')])
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
# print("input_details[0]['index']", input_details[0]['index'])
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
# print('height = ', height, type(height) )
height = np.int16(height).item()
# print('height2 = ', height, type(height) )
floating_model = int((input_details[0]['dtype'] == np.float32))
# print(type(floating_model))

input_mean = 127.5
input_std = 127.5
'''
# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()
while cap.isOpened():
    t1 = cv2.getTickCount()
    # 카메라 프레임 읽기
    success, frame1 = cap.read()
    try:
        img = frame1.copy()
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edge = cv2.Canny(img_gray, 50, 200, apertureSize=3)
        cv2.imshow('edge', edge)
    except:
        break

    #find point
    k1=True
    k2=True
    for i in range(imH-10,imH-50,-1):
        center = int(imW // 2)
        if k1==True:
            for j in range(center,center-250,-1):
                color=edge[i,j]
                if color==255:
                    p1=(j,i)
                    k1=False
        if k2==True:
            for j in range(center, center+250):
                color=edge[i,j]
                if color==255:
                    p2=(j,i)
                    k2=False
    k1 = True
    k2 = True
    for i in range(imH-175,imH-200,-1):
        center=int((p1[0]+p2[0])//2)
        if k1==True:
            for j in range(center,center-100,-1):
                color=edge[i,j]
                if color==255:
                    p3=(j,i)
                    k1=False
        if k2==True:
            for j in range(center, center+100):
                color=edge[i,j]
                if color==255:
                    p4=(j,i)
                    k2=True
    k1 = True
    k2 = True
    for i in range(imH-225,imH-250,-1):
        center=int((p3[0]+p4[0])//2)
        if k1==True:
            for j in range(center,center-50,-1):
                color=edge[i,j]
                if color==255:
                    p5=(j,i)
                    k1=False
        if k2==True:
            for j in range(center, center+50):
                color=edge[i,j]
                if color==255:
                    p6=(j,i)
                    k2=False
    k1 = True
    k2 = True
    for i in range(imH-270,imH-275,-1):
        center = int((p5[0] + p6[0])-(p4[0]+p3[0])//2)
        if k1==True:
            for j in range(center,center-40,-1):
                color=edge[i,j]
                if color==255:
                    p7=(j,i)
                    k1=False
        if k2==True:
            for j in range(center, center+40):
                color=edge[i,j]
                if color==255:
                    p8=(j,i)
                    k2=False
    k1=True
    k2=True
    for i in range(imH-280,imH-285,-1):
        center = int((p7[0] + p8[0])-(p5[0]+p6[0])//2)
        if k1==True:
            for j in range(center,center-40,-1):
                color=edge[i,j]
                if color==255:
                    p9=(j,i)
                    k1=False
        if k2==True:
            for j in range(center, center+40):
                color=edge[i,j]
                if color==255:
                    p10=(j,i)
                    k2=False
    k1 = True; k2 = True
    for i in range(imH-285,imH-290,-1):
        center = int((p9[0] + p10[0])-(p7[0]+p8[0])//2)
        if k1==True:
            for j in range(center,center-40,-1):
                color=edge[i,j]
                if color==255:
                    p11=(j,i)
                    k1=False
        if k2==True:
            for j in range(center, center+40):
                color=edge[i,j]
                if color==255:
                    p12=(j,i)
                    k2=False
    for i in range(imH-290,imH-295,-1):
        center = int((p11[0] + p12[0])-(p9[0]+p10[0])//2)
        cv2.line(img, (center,i), (center,i), (0, 255, 0), 5)
        if k1==True:
            for j in range(center,center-40,-1):
                color=edge[i,j]
                if color==255:
                    p13=(j,i)
                    k1=False
        if k2==True:
            for j in range(center, center+40):
                color=edge[i,j]
                if color==255:
                    p14=(j,i)
                    k2=False
    print([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12])
    # Draw point
    for x,y in [p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12]:
        cv2.line(img, (x,y), (x,y), (255, 0, 0), 5)
    cv2.imshow('img',img)

    # Draw framerate in corner of frame
    cv2.putText(img, 'FPS: {0:.2f}'.format(frame_rate_calc), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2,
                cv2.LINE_AA)
    cv2.imshow('img',img)
    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2 - t1) / freq
    frame_rate_calc = 1 / time1
    # cv2.imshow('img', img)

    # Press 'q' to quit
    if cv2.waitKey(0) == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
cap.release()

