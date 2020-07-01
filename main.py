######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# for the cpu version check TFlite_detection.py
# this is lite version of TFlife for using TPU

import os
import cv2
import numpy as np
import time
from threading import Thread
import platform
import tflite_runtime.interpreter as tflite

# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch:
# https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""

    def __init__(self, resolution=(640, 480), framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3, resolution[0]
                              )
        ret = self.stream.set(4, resolution[1])

        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

        # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
        # Start the thread that reads frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()


    def read(self):
        # Return the most recent frame
        return self.frame

    def stop(self):
        # Indicate that the camera and thread should be stopped
        self.stopped = True


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

# Initialize video stream
videostream = VideoStream(resolution=(imW, imH), framerate=30).start()
time.sleep(1)

# for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
while True:
    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Grab frame from video stream
    frame1 = videostream.read()

    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = frame1.copy()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #print("frame_rgb", frame_rgb.shape, frame_rgb.dtype)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    #print("frame_resized", width, height, frame_resized.shape, frame_resized.dtype)

    # print("frame_resized=", type(frame_resized))
    input_data = np.expand_dims(frame_resized, axis=0)
    #print(input_data.shape, input_data.dtype)
    #print(input_data[0][0])
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
                class_num == len(labels)    # maximum value
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
videostream.stop()

