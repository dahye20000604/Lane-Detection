import cv2
import numpy as np

cap = cv2.VideoCapture('./video/straight.avi')  # 0: default camera
# cap = cv2.VideoCapture("test.mp4") #동영상 파일에서 읽기

ret = cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
ret = cap.set(3, 640)
ret = cap.set(4, 480)

cap = cv2.VideoCapture('./video/straight.avi')  # 0: default camera
# resolution = '1920x1080'
resolution = '640x480'

resW, resH = resolution.split('x')
imW, imH = int(resW), int(resH)

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()
#out = cv2.VideoWriter('output_c.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30.0, (int(imW), int(imH)))
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
        #cv2.imshow('edge', edge)
    except:
        break
    # determine center
    c = imW // 2
    # determine width
    w = 200
    # width=[None]*imH
    first = 0
    second = 0
    # find point
    left_list = []
    right_list = []
    count = 0
    for i in range(imH - 10, 200, -5):
        l = 0;
        r = 0
        if left[i]:
            for j in range(left[i] - 5, left[i] + 5):
                if edge[i, j] == 255:
                    l = (j, i)
                    break
            if l == 0:
                left[i] = 0
        else:
            for j in range(c, c - 200, -1):
                if edge[i, j] == 255:
                    l = (j, i)
                    break
        if right[i]:
            for j in range(right[i] - 3, right[i] + 3):
                if edge[i, j] == 255:
                    r = (j, i)
                    break
            '''
            if abs(right[i]-left[i-1])<3:
                for j in range(c, c + 200):
                    if edge[i, j] == 255:
                        r = (j, i)
                        break
            '''
            if r == 0:
                right[i] = 0
        else:
            for j in range(c, c + 200):
                if edge[i, j] == 255:
                    r = (j, i)
                    break
        if r and l:
            final = i
            if r[0] - l[0] < w and r[0] - l[0] > 0:
                count += 1
                # cv2.line(img,(r[0],i),(r[0],i),(255,255,0),2)
                # cv2.line(img, (l[0], i), (l[0], i), (255, 0, 255), 2)
                left[i] = int(l[0])
                right[i] = int(r[0])
                w = r[0] - l[0]
                # width[i]=w
                c = (l[0] + r[0]) // 2
                left_list.append((l[0], i))
                right_list.append((r[0], i))
                if first == 0:
                    first = i
                else:
                    if second == 0:
                        second = i
            else:
                left[i] = 0
                right[i] = 0
        else:
            left[i] = 0
            right[i] = 0
    if count < 20:
        final = 0
    else:
        for i in range(final):
            left[i] = 0
            right[i] = 0
    # cv2.imshow('img1',img)
    img2 = frame1.copy()

    point_l = [];
    point_r = []
    for i in range(final, imH):
        if left[i] != 0:
            point_l.append((int(left[i]), i))
        if right[i] != 0:
            point_r.append((int(right[i]), i))
    for i in range(len(point_l) - 1):
        pass
        # cv2.line(img,point_l[i],point_l[i+1],(255,255,0),3)
    for i in range(len(point_r) - 1):
        pass
        # cv2.line(img,point_r[i],point_r[i+1],(255,255,0),3)
    # cv2.imshow('img2', img)

    # cv2.imshow('img2',img2)
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
    except:
        pass
    finally:
        for i in range(len(point) - 1):
            lx1 = point[i][0][0]  # lx1>lx2, y1<y2
            lx2 = point[i + 1][0][0]
            rx1 = point[i][1][0]
            rx2 = point[i + 1][1][0]
            for j in range(point[i][0][1] + 1, point[i + 1][0][1]):
                left[j] = int(lx1 - (lx1 - lx2) / (point[i + 1][0][1] - point[i][0][1]) * (j - point[i][0][1]))
                # cv2.line(frame1, (left[j], j), (left[j], j), (0, 0, 0), 5)
                right[j] = int(rx1 - (rx1 - rx2) / (point[i + 1][0][1] - point[i][0][1]) * (j - point[i][0][1]))
                # cv2.line(frame1, (right[j], j), (right[j], j), (0, 0, 0), 5)
        for i in range(len(point) - 1):
            corners = [point[i][0], point[i][1], point[i + 1][1], point[i + 1][0]]
            # for j in range(point[i][0][0],point[i+1][0][0]):
            # m=
            cv2.fillPoly(copy_img, np.int32([corners]), (255, 255, 255))
        image_new = cv2.addWeighted(copy_img, 0.4, frame1.copy(), 0.6, 0)
        pass
        # Draw framerate in corner of frame
    cv2.putText(image_new, 'FPS: {0:.2f}'.format(frame_rate_calc), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0),
                2,
                cv2.LINE_AA)
    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2 - t1) / freq
    frame_rate_calc = 1 / time1
    cv2.imshow('img', image_new)

    # out.write(image_new)
    # Press 'q' to quit
    if cv2.waitKey(10) == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
cap.release()
# out.release()