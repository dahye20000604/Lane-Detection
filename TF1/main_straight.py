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
while cap.isOpened():
    t1 = cv2.getTickCount()
    # 카메라 프레임 읽기
    success, frame1 = cap.read()
    img=frame1
    # grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Canny edge
    min, max = (50, 200)
    edge = cv2.Canny(img_gray, min, max, apertureSize=3)
    cv2.imshow('Canny Edge', edge)

    # Hough Transform
    lines = cv2.HoughLines(edge, 1, np.pi / 180, 300)
    L1 = []  # [a b]
    L2 = []  # [c]
    for line in lines:
        r, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * r
        y0 = b * r
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        L1.append([a, b])
        L2.append([r])
    # Find Vanishing Point
    U = np.linalg.pinv(L1)  # pseudo inverse
    x, y = np.dot(U, L2)
    cv2.line(img, (x, y), (x, y), (0, 0, 255), 5)
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
    while cap.isOpened():
        t1 = cv2.getTickCount()
        # 카메라 프레임 읽기
        success, frame1 = cap.read()
        try:
            img = frame1.copy()
        except:
            break
        # grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Canny edge
        min, max = (50, 200)
        edge = cv2.Canny(img_gray, min, max, apertureSize=3)
        cv2.imshow('Canny Edge', edge)

        # Hough Transform
        lines = cv2.HoughLines(edge, 1, np.pi / 180, 140)
        L1 = []  # [a b]
        L2 = []  # [c]
        for line in lines:
            r, theta = line[0]
            if theta > 0.5:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * r
                y0 = b * r
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * a)
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * a)
                m = (y1 - y2) / (x1 - x2)
                if abs(m) < 0.2 or abs(m) > 10:
                    pass
                else:
                    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    L1.append([a, b])
                    L2.append([r])
        # Find Vanishing Point
        if L1 == []:
            continue
        U = np.linalg.pinv(L1)  # pseudo inverse
        x, y = np.dot(U, L2)
        cv2.line(img, (x, y), (x, y), (0, 0, 255), 5)

        # vertex
        mask = np.zeros(img.shape, dtype=np.uint8)
        v1 = [x - 150, img.shape[1]]
        v2 = [x + 150, img.shape[1]]
        v3 = [x, y]
        roi_corners = np.array([v1, v2, v3])
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
        cv2.fillPoly(mask, np.int32([roi_corners]), ignore_mask_color)
        # from Masterfool: use cv2.fillConvexPoly if you know it's convex
        # apply the mask
        masked_image = cv2.bitwise_and(img, mask)
        cv2.imshow('crop', masked_image)
        # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2 - t1) / freq
        frame_rate_calc = 1 / time1
        cv2.imshow('img', img)
        # Press 'q' to quit
        if cv2.waitKey(100) == ord('q'):
            break

    # Clean up
    cv2.destroyAllWindows()
    cap.release()

    cv2.imshow('img', img)

    try:
        frame = masked_image.copy()
    except:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # print("frame_rgb", frame_rgb.shape, frame_rgb.dtype)
    # frame_resized=frame_rgb[:widhth,:height]
    frame_resized = cv2.resize(frame_rgb, (width, height))
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
    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if (scores[i] > min_conf_threshold) and (scores[i] <= 1.0):
            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them
            # to be within image using max() and min()
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

            # Draw label
            if classes[i] > len(labels):
                class_num == len(labels)  # maximum value
            else:
                class_num = int(classes[i])

            object_name = labels[class_num]  # Look up object name from "labels" array using class index

            label = '%s: %d%%' % (object_name, int(scores[i] * 100))  # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # Get font size
            label_ymin = max(ymin, labelSize[1] + 10)  # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10),
                          (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255),
                          cv2.FILLED)  # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),
                        2)  # Draw label text

    # Draw framerate in corner of frame
    cv2.putText(frame, 'FPS: {0:.2f}'.format(frame_rate_calc), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2,
                cv2.LINE_AA)

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2 - t1) / freq
    frame_rate_calc = 1 / time1

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
cap.release()

