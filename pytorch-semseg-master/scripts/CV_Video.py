
import os
import cv2
import numpy as np
import tensorflow as tf
import sys

sys.path.append("..")

MODEL_NAME = 'inference_graph'
#VIDEO_NAME = 'football1.mp4'
#VIDEO_NAME = 'fbhd.mp4'
VIDEO_NAME = 'fb5.mp4'
#VIDEO_NAME = 'instat.mp4'

CWD_PATH = os.getcwd()
PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME)

def MySobel(img, type = 'colorized'):
    ddepth = cv2.CV_16S
    scale = 1
    delta = 0

    img = cv2.GaussianBlur(img, (3, 3), 0)
    grad_x = cv2.Sobel(img, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(img, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    img = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    if type != 'gray':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return  img

def TeamDetectionv2(image):

    mask = MySobel(image)
    ret, mask  = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)
    kernel_Close = np.ones((9, 9), np.uint8)
    kernel_Erode = np.ones((2, 2), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_Close)
    #mask = cv2.erode(mask, kernel_Erode, iterations=1)

    image = cv2.bitwise_and(image, image, mask=mask)

    return image


def LineDetection(image):
    image = cv2.GaussianBlur(image, (3, 3), 0)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    mask = cv2.Canny(gray,45,90,apertureSize = 3)

    #ret, mask  = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)

    kernel_Close = np.ones((6, 6), np.uint8)
    kernel_Open = np.ones((5, 5), np.uint8)
    kernel_Erode = np.ones((2, 2), np.uint8)

    #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_Close)
    #mask = cv2.erode(mask, kernel_Erode, iterations=1)
    #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_Open)

    #mask = cv2.medianBlur(mask, 5)
    #image = cv2.bitwise_and(image, image, mask=mask)

    """
    width = np.size(image, 1)
    height = np.size(image, 0)

    lines = cv2.HoughLinesP(mask,rho = 1,theta = 1*np.pi/180,threshold = 200,minLineLength = 100,maxLineGap = 2)
    for x in range(0, len(lines)):
        for x1,y1,x2,y2 in lines[x]:
            x1,y1,x2,y2 = LengthenLine(int(x1),int(y1),int(x2),int(y2), [width, height])
            cv2.line(image,(x1,y1),(x2,y2),(255,0,0),2)
    """
    return mask

video = cv2.VideoCapture(PATH_TO_VIDEO)

while(video.isOpened()):

    ret, frame = video.read()
    frame_expanded = np.expand_dims(frame, axis=0)

    #frame = TeamDetectionv2(frame)
    frame = LineDetection(frame)

    cv2.imshow('Object detector', frame)

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
