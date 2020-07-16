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
i = 0
x = 0
y = 0
while cap.isOpened():
    t1 = cv2.getTickCount()
    # 카메라 프레임 읽기
    success, frame1 = cap.read()
    try:
        img = frame1.copy()
        # arbitrarily triangle crop
        mask = np.zeros(img.shape, dtype=np.uint8)
        v1 = (0, img.shape[0] - 1)
        v2 = (img.shape[1] - 1, img.shape[0] - 1)
        v3 = (img.shape[1] // 2, 0)
        roi_corners = np.array([v1, v2, v3])
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
        #cv2.fillPoly(mask, np.int32([roi_corners]), ignore_mask_color)
        #img = cv2.bitwise_and(img, mask)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edge = cv2.Canny(img_gray, 50, 200, apertureSize=3)
        # cv2.line(edge, v1, v3, (0, 0, 0), 2)
        # cv2.line(edge, v2, v3, (0, 0, 0), 2)
        # cv2.line(edge, v1, v2, (0, 0, 0), 5)
        # cv2.imshow('1',img)
    except:
        break

    #first_cut : img.shape[0]-200 : img.shape[0]
    img_cut1 = img[img.shape[0] - 200:img.shape[0], :]
    img1=img_cut1.copy()
    # Canny edge
    img_cut1_edge = edge[img.shape[0]-200:img.shape[0], :]
    cv2.imshow('first : Canny edge', img_cut1_edge)
    # Hough Transform
    lines=cv2.HoughLines(img_cut1_edge, 1, np.pi / 180, 100)
    L1 = []  # [a b]
    L2 = []  # [c]
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
                m = (y1 - y2) / (x2 - x1)
                py1 = img_cut1.shape[0]
                px1 = int((py1 - y1) / m + x1)
                py2 = 0
                px2 = int((py2 - y1) / m + x1)
                cv2.line(img_cut1, (px1, py1), (px2, py2), (0, 255, 0), 1)
                L1.append([a, b])
                L2.append([r])
    cv2.imshow('first : Hough Transform',img_cut1)
    # Find Vanishing Point
    U1 = np.linalg.pinv(L1)  # pseudo inverse
    cut1_x, cut1_y = np.dot(U1, L2)
    # crop using vanishing point
    mask = np.zeros(img_cut1.shape, dtype=np.uint8)
    v1 = (int(cut1_x - 250), img_cut1.shape[0])
    v2 = (int(cut1_x + 250), img_cut1.shape[0])
    v3 = (cut1_x, cut1_y)
    roi_corners = np.array([v1, v2, v3])
    channel_count = img.shape[2]
    ignore_mask_color = (255,) * channel_count
    cv2.fillPoly(mask, np.int32([roi_corners]), ignore_mask_color)
    cut1_masked_image = cv2.bitwise_and(img_cut1, mask)
    cv2.imshow('first : ROI', cut1_masked_image)

    # Canny edge
    img_cut1_gray = cv2.cvtColor(cut1_masked_image, cv2.COLOR_BGR2GRAY)
    img_cut1_edge = cv2.Canny(img_cut1_gray, 50, 200, apertureSize=3)
    cv2.line(img_cut1_edge, v1, v3, (0, 0, 0), 5)
    cv2.line(img_cut1_edge, v2, v3, (0, 0, 0), 5)
    cv2.imshow('first : Canny edge2', img_cut1_edge)
    # Hough Transform
    lines = cv2.HoughLines(img_cut1_edge, 1, np.pi / 180, 80)
    L2 = []  # [c]
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
                cv2.line(img1, (x1, y1), (x2, y2), (0, 255, 0), 1)
                L1.append([a, b])
                L2.append([r])
    cv2.imshow('first : lane detection', img1)

    #second_cut : img.shape[0]-250 : img.shape[0]-200
    img_cut2 = img[img.shape[0]-250:img.shape[0]-200, :]
    img2=img_cut2.copy()
    # Canny edge
    img_cut2_edge = edge[img.shape[0]-250:img.shape[0]-200, :]
    cv2.imshow('second : Canny edge', img_cut2_edge)
    # Hough Transform
    lines=cv2.HoughLines(img_cut2_edge, 1, np.pi / 180, 50)
    L1 = []  # [a b]
    L2 = []  # [c]
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
                # cv2.line(img_cut1, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.line(img_cut2, (x1, y1), (x2, y2), (0, 255, 0), 1)
                L1.append([a, b])
                L2.append([r])
    cv2.imshow('second : Hough Transform', img_cut2)
    # Find Vanishing Point
    try:
        U2 = np.linalg.pinv(L1)  # pseudo inverse
        cut2_x, cut2_y = np.dot(U2, L2)
        # crop using vanishing point
        mask = np.zeros(img_cut2.shape, dtype=np.uint8)
        cut2_v1 = (int(imW//2 - 100), img_cut2.shape[0])
        cut2_v2 = (int(imW//2 + 100), img_cut2.shape[0])
        cut2_v3 = (cut2_x, cut2_y)
        cut2_roi_corners = np.array([cut2_v1,cut2_v2, cut2_v3])
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
        cv2.fillPoly(mask, np.int32([cut2_roi_corners]), ignore_mask_color)
        cut2_masked_image = cv2.bitwise_and(img_cut2, mask)
    except:
        pass
    cv2.imshow('second : ROI', cut2_masked_image)

    cut2_img_gray = cv2.cvtColor(cut2_masked_image, cv2.COLOR_BGR2GRAY)
    cut2_edge = cv2.Canny(cut2_img_gray, 50, 200, apertureSize=3)
    cv2.line(cut2_edge, cut2_v1, cut2_v3, (0, 0, 0), 2)
    cv2.line(cut2_edge, cut2_v2, cut2_v3, (0, 0, 0), 2)
    cv2.imshow('second : Canny edge2', cut2_edge)
    # Hough Transform
    lines = cv2.HoughLines(cut2_edge, 1, np.pi / 180, 20)
    try:
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
    except:
        pass
    cv2.imshow('second : lane detection', img2)

    #third : img.shape[0] - 280: img.shape[0] - 250
    img_cut3 = img[img.shape[0] - 280:img.shape[0] - 250, :]
    img3 = img_cut3.copy()
    # Canny edge
    img_cut3_edge = edge[img.shape[0] - 280:img.shape[0] - 250, :]
    cv2.imshow('third : Canny edge', img_cut3_edge)
    # Hough Transform
    lines = cv2.HoughLines(img_cut3_edge, 1, np.pi / 180, 30)
    L1 = []  # [a b]
    L2 = []  # [c]
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
                cv2.line(img_cut3, (x1, y1), (x2, y2), (0, 255, 0), 1)
                L1.append([a, b])
                L2.append([r])
    cv2.imshow('third : Hough Transform', img_cut3)
    # Find Vanishing Point
    U3 = np.linalg.pinv(L1)  # pseudo inverse
    cut3_x, cut3_y = np.dot(U3, L2)
    # crop using vanishing point
    mask = np.zeros(img_cut3.shape, dtype=np.uint8)
    cut3_v1 = (int(imW // 2 - 70), img_cut3.shape[0])
    cut3_v2 = (int(imW // 2 + 70), img_cut3.shape[0])
    cut3_v3 = (cut3_x, cut3_y)
    cut3_roi_corners = np.array([cut3_v1, cut3_v2, cut3_v3])
    channel_count = img.shape[2]
    ignore_mask_color = (255,) * channel_count
    cv2.fillPoly(mask, np.int32([cut3_roi_corners]), ignore_mask_color)
    cut3_masked_image = cv2.bitwise_and(img_cut3, mask)

    cv2.imshow('third : ROI', cut3_masked_image)

    cut3_img_gray = cv2.cvtColor(cut3_masked_image, cv2.COLOR_BGR2GRAY)
    cut3_edge = cv2.Canny(cut3_img_gray, 50, 200, apertureSize=3)
    cv2.line(cut3_edge, cut3_v1, cut3_v3, (0, 0, 0), 2)
    cv2.line(cut3_edge, cut3_v2, cut3_v3, (0, 0, 0), 2)
    cv2.imshow('third : Canny edge2', cut3_edge)
    # Hough Transform
    lines = cv2.HoughLines(cut3_edge, 1, np.pi / 180, 20)
    try:
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
                    cv2.line(img3, (x1, y1), (x2, y2), (0, 255, 0), 1)
    except:
        pass

    cv2.imshow('third : lane detection', img3)
    add_img=cv2.vconcat([img[:img.shape[0]-280,:],img3,img2,img1])
    cv2.imshow('add', add_img)

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

