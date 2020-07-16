# 사다리꼴 이용해서 점찾기
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
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

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()
#out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30.0, (int(imW), int(imH)))
# right point
right = [0] * imH
# left point
left = [0] * imH
while cap.isOpened():
    t1 = cv2.getTickCount()
    # 카메라 프레임 읽기
    success, frame1 = cap.read()
    try:
        # ROI
        img = frame1.copy();
        frame=frame1.copy()
        mask = np.zeros(img.shape, dtype=np.uint8)
        v1 = (int(imW // 2 - 200), imH);
        v2 = (int(imW // 2 + 200), imH);
        v3 = (int(imW // 2), 0)
        roi_corners = np.array([v1, v2, v3]);
        cv2.fillPoly(mask, np.int32([roi_corners]), (255, 255, 255))
        img_crop = cv2.bitwise_and(img, mask)
        # cv2.imshow('crop', img_crop)

        # edge
        img_gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
        edge = cv2.Canny(img_gray, 100, 200, apertureSize=3)
        cv2.line(edge, v1, v3, (0, 0, 0), 2);
        cv2.line(edge, v2, v3, (0, 0, 0), 2)
        cv2.imshow('edge', edge)
    except:
        break

    #car detection
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # print("frame_rgb", frame_rgb.shape, frame_rgb.dtype)
    # frame_resized=frame_rgb[:widhth,:height]
    frame_resized = frame_rgb[imH // 2 - height // 2:imH // 2 + height // 2,
                    imW // 2 - width // 2:imW // 2 + width // 2]
    # frame_resized = cv2.resize(frame_rgb, (width, height))

    # print("frame_resized", width, height, frame_resized.shape, frame_resized.dtype)

    # print("frame_resized=", type(frame_resized))
    input_data = np.expand_dims(frame_resized, axis=0)
    # print(input_data.shape, input_data.dtype)
    # print(input_data[0][0])
    # print("after expend=", type(frame_resized), type(input_data))
    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    # print("input_details[0]['index']", input_details[0]['index'], input_data.shape)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects
    # num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects
    # (inaccurate and not needed)
    # print(type(boxes), type(classes), type(scores))
    #


    # determine center
    c = imW // 2
    # determine width
    w = 200
    #width=[None]*imH
    first = 0
    second = 0
    # find point
    left_list=[]
    right_list=[]
    for i in range(imH - 10, 200, -5):
        l = 0;
        r = 0
        if left[i]:
            for j in range(left[i]-5,left[i]+5):
                if edge[i,j]==255:
                    l=(j,i)
                    break
            if l==0:
                left[i]=0
        else:
            for j in range(c, c - 200, -1):
                if edge[i, j] == 255:
                    l = (j, i)
                    break
        if right[i]:
            for j in range(right[i]-3,right[i]+3):
                if edge[i,j]==255:
                    r=(j,i)
                    break
            if abs(right[i]-left[i-1])<3:
                for j in range(c, c + 200):
                    if edge[i, j] == 255:
                        r = (j, i)
                        break
            if r==0:
                right[i]=0
        else:
            for j in range(c, c + 200):
                if edge[i, j] == 255:
                    r = (j, i)
                    break
        if r and l:
            if r[0] - l[0] < w and r[0] - l[0] > 0:
                cv2.line(img,(r[0],i),(r[0],i),(255,255,0),2)
                cv2.line(img, (l[0], i), (l[0], i), (255, 0, 255), 2)
                right[i] = r[0]
                left[i] = l[0]
                w = r[0] - l[0]
                #width[i]=w
                c = (l[0] + r[0]) // 2
                left_list.append((l[0],i))
                right_list.append((r[0], i))
                if first == 0:
                    first = i
                else:
                    if second == 0:
                        second = i
            else:
                left[i]=0
                right[i]=0
        else:
            left[i]=0
            right[i]=0
    #print(left, right)
    cv2.imshow('img1',img)
    img2=frame1.copy()

    #cv2.imshow('img2',img2)
    copy_img = frame1.copy()
    try:
        xl = left[first] - (left[second] - left[first]) / (first - second) * (imH - first)
        xr = right[first] + (right[first] - right[second]) / (first - second) * (imH - first)
        left[-1] = xl
        right[-1] = xr
        point = []
        for i in range(imH):
            if left[i] and right[i]:
                point.append([(left[i], i), (right[i], i)])
        xl = point[0][0][0] + (point[0][0][0] - point[1][0][0]) / (point[1][0][1] - point[0][0][1]) * (point[0][0][1] - 100)
        xr = point[0][1][0] - (point[1][1][0] - point[0][1][0]) / (point[1][1][1] - point[0][1][1]) * (point[0][1][1] - 100)
        #left[100] = xl
        #right[100] = xr
        point.insert(0, [(xl, 100), (xr, 100)])
        for i in range(len(point)-1):
            lx1=point[i][0][0] #lx1>lx2, y1<y2
            lx2=point[i+1][0][0]
            rx1=point[i][1][0]
            rx2=point[i+1][1][0]
            for j in range(point[i][0][1]+1,point[i+1][0][1]):
                left[j]=int(lx1-(lx1-lx2)/(point[i+1][0][1]-point[i][0][1])*(j-point[i][0][1]))
                cv2.line(frame1,(left[j],j),(left[j],j),(0,0,0),5)
                right[j]=int(rx1-(rx1-rx2)/(point[i+1][0][1]-point[i][0][1])*(j-point[i][0][1]))
                cv2.line(frame1, (right[j], j), (right[j], j), (0, 0, 0), 5)
        for i in range(len(point) - 1):
            corners = [point[i][0], point[i][1], point[i + 1][1], point[i + 1][0]]
            # for j in range(point[i][0][0],point[i+1][0][0]):
            # m=
            cv2.fillPoly(copy_img, np.int32([corners]), (255, 255, 255))
        image_new = cv2.addWeighted(copy_img, 0.4, frame1.copy(), 0.6, 0)
    except:
        point = []
        for i in range(imH):
            if left[i] and right[i]:
                point.append([(left[i], i), (right[i], i)])
        xl = point[0][0][0] + (point[0][0][0] - point[1][0][0]) / (point[1][0][1] - point[0][0][1]) * (
                    point[0][0][1] - 100)
        xr = point[0][1][0] - (point[1][1][0] - point[0][1][0]) / (point[1][1][1] - point[0][1][1]) * (
                    point[0][1][1] - 100)
        # left[100] = xl
        # right[100] = xr
        point.insert(0, [(xl, 100), (xr, 100)])
        for i in range(len(point) - 1):
            lx1 = point[i][0][0]  # lx1>lx2, y1<y2
            lx2 = point[i + 1][0][0]
            rx1 = point[i][1][0]
            rx2 = point[i + 1][1][0]
            for j in range(point[i][0][1] + 1, point[i + 1][0][1]):
                left[j] = int(lx1 - (lx1 - lx2) / (point[i + 1][0][1] - point[i][0][1]) * (j - point[i][0][1]))
                cv2.line(frame1, (left[j], j), (left[j], j), (0, 0, 0), 5)
                right[j] = int(rx1 - (rx1 - rx2) / (point[i + 1][0][1] - point[i][0][1]) * (j - point[i][0][1]))
                cv2.line(frame1, (right[j], j), (right[j], j), (0, 0, 0), 5)
        for i in range(len(point) - 1):
            corners = [point[i][0], point[i][1], point[i + 1][1], point[i + 1][0]]
            # for j in range(point[i][0][0],point[i+1][0][0]):
            # m=
            cv2.fillPoly(copy_img, np.int32([corners]), (255, 255, 255))
        image_new = cv2.addWeighted(copy_img, 0.4, frame1.copy(), 0.6, 0)
        pass
    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if (scores[i] > min_conf_threshold) and (scores[i] <= 1.0):
            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them
            # to be within image using max() and min()
            ymin = int(max(1, boxes[i][0] * height))
            xmin = int(max(1, boxes[i][1] * width))
            ymax = int(min(imH, boxes[i][2] * height))
            xmax = int(min(imW, boxes[i][3] * width))
            cv2.rectangle(frame_resized, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
            xmin = imW // 2 - (width // 2 - xmin)
            ymin = imH // 2 - (height // 2 - ymin)
            xmax = imW // 2 + (xmax - width // 2)
            ymax = imH // 2 + (ymax - height // 2)
            cv2.rectangle(image_new, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

            # Draw labelq
            if classes[i] > len(labels):
                class_num == len(labels)  # maximum value
            else:
                class_num = int(classes[i])

            object_name = labels[class_num]  # Look up object name from "labels" array using class index

            label = '%s: %d%%' % (object_name, int(scores[i] * 100))  # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # Get font size
            label_ymin = max(ymin, labelSize[1] + 10)  # Make sure not to draw label too close to top of window
            cv2.rectangle(image_new, (xmin, label_ymin - labelSize[1] - 10),
                          (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255),
                          cv2.FILLED)  # Draw white box to put label text in
            cv2.putText(image_new, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),
                        2)  # Draw label text
    # Draw framerate in corner of frame
    cv2.putText(image_new, 'FPS: {0:.2f}'.format(frame_rate_calc), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0),
                2,
                cv2.LINE_AA)

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', image_new)
    cv2.imshow('test', frame_resized)
    #cv2.imshow('rectangle', image_new)

    # cv2.imshow('fit', img_fit)



    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2 - t1) / freq
    frame_rate_calc = 1 / time1
    cv2.imshow('img', image_new)

    #out.write(image_new)
    # Press 'q' to quit
    if cv2.waitKey(0) == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
cap.release()
#out.release()
