import os
import cv2
import numpy as np
import time
from threading import Thread
import platform
import tflite_runtime.interpreter as tflite
cap = cv2.VideoCapture('./video/straight.avi')  # 0: default camera
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
interpreter=tflite.Interpreter(
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

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()
i=0
x=0
y=0
while cap.isOpened():
    t1 = cv2.getTickCount()
    # 카메라 프레임 읽기
    success, frame1 = cap.read()
    try:
        img = frame1.copy()
        img2 = frame1.copy()

        # arbitrarily triangle crop
        mask = np.zeros(img.shape, dtype=np.uint8)
        v1 = (0, img.shape[0] - 1)
        v2 = (img.shape[1] - 1, img.shape[0] - 1)
        v3 = (img.shape[1] // 2, 0)
        roi_corners = np.array([v1, v2, v3])
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
        cv2.fillPoly(mask, np.int32([roi_corners]), ignore_mask_color)
        img=cv2.bitwise_and(img,mask)
    except:
        break

    # grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Canny edge
    min, max = (100, 200)
    edge = cv2.Canny(img_gray, min, max, apertureSize = 3)
    cv2.line(edge, v1, v3, (0, 0, 0), 2)
    cv2.line(edge, v2, v3, (0, 0, 0), 2)
    cv2.imshow('edge2', edge)

    # Hough Transform
    k=200
    lines = cv2.HoughLines(edge, 1, np.pi / 180, k)
    while len(lines)<30:
        k-=10
        lines = cv2.HoughLines(edge, 1, np.pi / 180, k)
    L1 = []  # [a b]
    L2 = []  # [c]
    for line in lines:
        r, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        if theta and abs(a/b)>0.3:
            x0 = a * r
            y0 = b * r
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)
            m=-a/b
            if abs(a/b)>50:
                pass
            else:
                cv2.line(img2, (x1, y1), (x2, y2), (0, 255, 0), 1)
                L1.append([a, b])
                L2.append([r])

    # Find Vanishing Point
    U = np.linalg.pinv(L1)  # pseudo inverse
    i+=1
    x, y = np.dot(U, L2)
    cv2.line(img2, (x, y), (x, y), (0, 0, 255), 5)
    cv2.imshow('vanishing point1',img2)
    # crop using vanishing point
    mask = np.zeros(img.shape, dtype=np.uint8)
    v1=(x-250,img.shape[1])
    v2=(x+250,img.shape[1])
    v3=(x,y-50)
    roi_corners = np.array([v1, v2, v3])
    channel_count = img.shape[2]
    ignore_mask_color = (255,) * channel_count
    cv2.fillPoly(mask, np.int32([roi_corners]), ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    cv2.imshow('first',masked_image)

    #second
    # grayscale
    img2=masked_image
    img_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Canny edge
    min, max = (100, 200)
    edge = cv2.Canny(img_gray, min, max, apertureSize=3)
    cv2.line(edge, v1, v3, (0, 0, 0), 3)
    cv2.line(edge, v2, v3, (0, 0, 0), 3)
    cv2.imshow('edge2', edge)

    # Hough Transform
    k = 100
    lines = cv2.HoughLines(edge, 1, np.pi / 180, k)
    while len(lines) < 30:
        k -= 10
        lines = cv2.HoughLines(edge, 1, np.pi / 180, k)
    L1 = []  # [a b]
    L2 = []  # [c]
    L=[]
    for line in lines:
        r, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        if theta and abs(a / b) > 0.3:
            x0 = a * r
            y0 = b * r
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)
            m = -a / b
            if abs(a / b) > 50:
                pass
            else:
                cv2.line(img2, (x1, y1), (x2, y2), (0, 255, 0), 1)
                L1.append([a, b])
                L2.append([r])
                if y1>y2:
                    L.append([x1,y1])
                else:
                    L.append([x2,y2])
    
    # Find Vanishing Point
    U = np.linalg.pinv(L1)  # pseudo inverse
    i += 1
    x, y = np.dot(U, L2)
    cv2.line(img2, (x, y), (x, y), (0, 0, 255), 5)
    cv2.imshow('vanishing point2', img2)
    # crop using vanishing point
    mask = np.zeros(img.shape, dtype=np.uint8)
    v1 = (x - 250, img.shape[1])
    v2 = (x + 250, img.shape[1])
    v3 = (x, y - 50)
    roi_corners = np.array([v1, v2, v3])
    channel_count = img.shape[2]
    ignore_mask_color = (255,) * channel_count
    cv2.fillPoly(mask, np.int32([roi_corners]), ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    cv2.imshow('second', masked_image)

    #line detection
    for px,py in L:
        cv2.line(masked_image, (x,y), (px,py), (255,0,0), 2)

    cv2.imshow('line', masked_image)
    '''
    #lane detection

    # grayscale
    img_gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

    # Canny edge
    min, max = (100, 200)
    edge = cv2.Canny(img_gray, min, max, apertureSize=3)
    cv2.line(edge, v1, v3, (0, 0, 0), 2)
    cv2.line(edge, v2, v3, (0, 0, 0), 2)
    cv2.line(edge, v1, v2, (0, 0, 0), 5)
    cv2.imshow('edge2', edge)

    #차선 점찍기
    mask = np.zeros(img.shape, dtype=np.uint8)
    lines=cv2.HoughLinesP(edge, 1, np.pi/180, 30, 50, 0)
    lp=(x,y)
    rp=(img.shape[1], img.shape[1])
    m1=0
    m2=0
    for line in lines:
        x1, y1, x2, y2=line[0]
        if x1<x and x1>lp[0] and y1<imH-30:
            lp=(x1,y1)
            m1 = (y1 - y) / (x1 - x)
        if x1>x and x1<rp[0] and y1<imH-30:
            rp=(x1,y1)
            m2 = (y1 - y) / (x1 - x)
    if m1==0:
        m1=-m2
    if m2==0:
        m2=-m1
    cv2.line(img, (x, y), (int((img.shape[0] - y) / m1 + x), img.shape[0] - 1), (255, 0, 0), 2)
    cv2.line(img, (x, y), (int((img.shape[0] - y) / m2 + x), img.shape[0] - 1), (255, 0, 0), 2)
    cv2.imshow('last',img)
    '''
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

