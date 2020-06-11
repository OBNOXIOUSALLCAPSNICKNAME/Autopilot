import os
import cv2
import sys
import math
import shapely
import statistics
import numpy as np
import pandas as pd
import tensorflow as tf

from collections import defaultdict
from collections import Counter
from math import atan2, degrees

from shapely.geometry import LineString
from scipy.spatial import distance
from skimage.transform import (hough_line, hough_line_peaks, probabilistic_hough_line)

from utils import label_map_util
from utils import visualization_utils as vis_util


sys.path.append("..")

CWD_PATH = os.getcwd()

#VIDEO_NAME = 'line2.mp4'
#VIDEO_NAME = 'line1.mp4'
#VIDEO_NAME = '4th.mp4'
#VIDEO_NAME = '3rd.mp4'
#VIDEO_NAME = '2nd.mp4'
#VIDEO_NAME = '1st.mp4'
VIDEO_NAME = 'lines.mp4'
PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME)

#===============

def WarpPersp(p, img):

    rows,cols = img.shape

    u0 = (cols)/2.0
    v0 = (rows)/2.0

    w1 = distance.euclidean(p[0],p[1])
    w2 = distance.euclidean(p[2],p[3])

    h1 = distance.euclidean(p[0],p[2])
    h2 = distance.euclidean(p[1],p[3])

    w = max(w1,w2)
    h = max(h1,h2)

    ar_vis = float(w)/float(h)

    m1 = np.array((p[0][0],p[0][1],1)).astype('float32')
    m2 = np.array((p[1][0],p[1][1],1)).astype('float32')
    m3 = np.array((p[2][0],p[2][1],1)).astype('float32')
    m4 = np.array((p[3][0],p[3][1],1)).astype('float32')

    k2 = np.dot(np.cross(m1,m4),m3) / np.dot(np.cross(m2,m4),m3)
    k3 = np.dot(np.cross(m1,m4),m2) / np.dot(np.cross(m3,m4),m2)

    n2 = k2 * m2 - m1
    n3 = k3 * m3 - m1

    n21 = n2[0]
    n22 = n2[1]
    n23 = n2[2]

    n31 = n3[0]
    n32 = n3[1]
    n33 = n3[2]

    f = math.sqrt(np.abs( (1.0/(n23*n33)) * ((n21*n31 - (n21*n33 + n23*n31)*u0 + n23*n33*u0*u0) + (n22*n32 - (n22*n33+n23*n32)*v0 + n23*n33*v0*v0))))

    A = np.array([[f,0,u0],[0,f,v0],[0,0,1]]).astype('float32')

    At = np.transpose(A)
    Ati = np.linalg.inv(At)
    Ai = np.linalg.inv(A)

    ar_real = 1/math.sqrt(np.dot(np.dot(np.dot(n2,Ati),Ai),n2)/np.dot(np.dot(np.dot(n3,Ati),Ai),n3)) + 0.0000001

    if ar_real < ar_vis:
        W = int(w)
        H = int(W / ar_real)
    else:
        H = int(h)
        W = int(ar_real * H)

    if ar_real > 1:
        ar_real = 1/ar_real

    return '%.3f'%ar_real

def order_points(A):
    sortedAc2 = A[np.argsort(A[:,1]),:]

    top2 = sortedAc2[0:2,:]
    bottom2 = sortedAc2[2:,:]

    sortedtop2c1 = top2[np.argsort(top2[:,0]),:]
    top_left = sortedtop2c1[0,:]

    sqdists = distance.cdist(top_left[None], bottom2, 'sqeuclidean')
    rest2 = bottom2[np.argsort(np.max(sqdists,0))[::-1],:]

    res = np.concatenate((sortedtop2c1,rest2),axis =0)

    # костыль
    if res[2][0] > res[3][0]:
        temp = res[2].copy()
        res[2] = res[3]
        res[3] = temp

    return res

def CheckShape(pts):
    x, y, w, h = cv2.boundingRect(pts)
    if w < 15 or h < 15:
        return False
    TW = LineString([(pts[0][0], pts[0][1]), (pts[1][0], pts[1][1])])
    BW = LineString([(pts[2][0], pts[2][1]), (pts[3][0], pts[3][1])])
    LH = LineString([(pts[0][0], pts[0][1]), (pts[2][0], pts[2][1])])
    RH = LineString([(pts[1][0], pts[1][1]), (pts[3][0], pts[3][1])])

    min_W = min(TW.length, BW.length)
    max_W = max(TW.length, BW.length)

    min_H = min(LH.length, RH.length)
    max_H = max(LH.length, RH.length)

    ar_W = min_W / max_W + 0.0000001
    ar_H = min_H / max_H + 0.0000001
    ar = min(min_W, min_H) / max(max_W, max_H)
    if ar_H < 0.5 or ar_W < 0.5 or ar < 0.2:
        return False
    else:
        return True

def Figures(image):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image,60,255,cv2.THRESH_BINARY)
    contorus, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    output = image.copy()
    output = cv2.cvtColor(output,cv2.COLOR_GRAY2RGB)

    for contur in contorus:
        approx = cv2.approxPolyDP(contur, 0.01*cv2.arcLength(contur, True), True)
        if len(approx) == 4:
            new_approx = order_points(np.vstack(approx))
            if CheckShape(new_approx):
                aspect_ratio = WarpPersp(new_approx, image)
                text_point = (int((new_approx[0][0] + new_approx[3][0])/2), int((new_approx[1][1] + new_approx[2][1])/2))

                cv2.drawContours(output, [approx], 0, (28,25,21), 3)
                cv2.fillPoly(output, pts =[approx], color=(200,182,51))
                cv2.putText(output, str(aspect_ratio), text_point, cv2.FONT_HERSHEY_DUPLEX, 1, (28,25,21), 2)

    return output


def HueMask(image):
    width = np.size(image, 1)
    height = np.size(image, 0)

    image = cv2.GaussianBlur(image, (3, 3), 0)
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    unique, counts = np.unique(hsv[:, :, 0], return_counts=True)
    hist_s = pd.Series(counts, unique)

    # Normalize hist
    hist_s = (hist_s / width / height) * 100
    h_argmax = hist_s.argmax()
    h_argright = h_argmax + 5
    h_argleft = h_argmax - 5
    img_slice = hsv[:, :, 0]

    # Change image pixel of 'H' channel by next conditions
    result_np1 = np.where((img_slice > h_argleft) & (img_slice < h_argright), 255, 0)
    result_np2 = np.where(hist_s[img_slice.flatten()] > 0.1, 255, 0).reshape(height, width)
    result_np = np.where(result_np1 == result_np2, result_np1, 0)
    result_np = np.uint8(result_np)

    return result_np

def LineDistance(line1_p1,line1_p2, line2_p):
    p1 = np.asarray((line1_p1[0], line1_p1[1]))
    p2 = np.asarray((line1_p2[0], line1_p2[1]))
    p3 = np.asarray((line2_p[0], line2_p[1]))
    distance = int(np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1))
    return distance

def SegmentLines(lines, **kwargs):
    if len(lines) < 2:
        return [lines]
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    slice = np.float32(lines)[:,4]
    _1, labels, _2 = cv2.kmeans(slice, 2, None, criteria, attempts, flags)

    segments = [[], []]
    for x in range(len(lines)):
        segments[labels[x][0]].append(lines[x])

    avg_1st, _, _ = LineStatistics(segments[0])
    avg_2nd, _, _ = LineStatistics(segments[1])

    if avg_2nd < avg_1st:
        temp = segments[0]
        segments[0] = segments[1]
        segments[1] = temp

    return segments

def LineStatistics(lines, kwarg="default"):
    if len(lines) > 1:
        if kwarg == "default":
            sliced = np.asarray(lines)[:, 4]
        else:
            sliced = np.asarray([abs(ele[4]) for ele in lines])[:,]
        avg = sum(sliced) / len(lines)
        std = statistics.stdev(sliced)
        dif = abs(max(sliced) - min(sliced))
        return avg, std, dif
    else:
        if kwarg == "default":
            return lines[0][4], -1, -1
        else:
            return abs(lines[0][4]), -1, -1


def GroupLines(input_lines, unique = True):
    #return input_lines
    group = []
    output = []
    lines = input_lines.copy()
    length = len(lines)
    i = 0
    while i < length:
        j = 0
        i_x1,i_y1,i_x2,i_y2,angle_i = lines[i]
        group.append(lines[i].copy())
        while j < length:
            if i != j:
                j_x1,j_y1,j_x2,j_y2,angle_j = lines[j]

                distance = LineDistance((i_x1,i_y1), (i_x2,i_y2), (j_x1,j_y1))
                dif_angle = abs(max(angle_i, angle_j) - min(angle_i, angle_j))

                if (dif_angle <= 2 and distance < 20) or (dif_angle < 8 and dif_angle > 2 and distance < 30):
                    group.append(lines[j].copy())
                    del lines[j]

                    length -= 1
                    if i == length and i > j: i -= 1
                    j -= 1
            j += 1

        average_x1 = sum([element[0] for element in group]) / len(group)
        average_y1 = sum([element[1] for element in group]) / len(group)
        average_x2 = sum([element[2] for element in group]) / len(group)
        average_y2 = sum([element[3] for element in group]) / len(group)
        angle_avg = np.degrees(np.arctan2(average_y1 - average_y2, average_x1 - average_x2))

        if unique == True:
            output.append([average_x1,average_y1,average_x2,average_y2,angle_avg])
        else:
            for x in range(len(group)):
                output.append([average_x1,average_y1,average_x2,average_y2,angle_avg])
        group.clear()
        i += 1

    return output

def DetectPart(segments):
    angles_dif = []
    for i in range(len(segments[0])):
        angle_1st = abs(segments[0][i][4])
        for j in range(len(segments[1])):
            angle_2nd = abs(segments[1][j][4])
            angles_dif.append(abs(angle_1st - angle_2nd))
    avg_dif = sum(angles_dif) / len(angles_dif)

    avg_1st, _, _ = LineStatistics(segments[0], "absolute")
    avg_2nd, _, _ = LineStatistics(segments[1], "absolute")
    if avg_dif < 40:
        if avg_1st < avg_2nd:
            return "right " + str('%.3f'%avg_1st) + " " + str('%.3f'%avg_2nd)
        else:
            return "left " + str('%.3f'%avg_1st) + " " + str('%.3f'%avg_2nd)
    else:
        return "center " + str('%.3f'%avg_1st) + " " + str('%.3f'%avg_2nd)

def PreprocImage(image):
    field_mask = HueMask(image)

    field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    field_mask = cv2.dilate(field_mask, np.ones((60, 60), np.uint8),iterations = 1)
    field_mask = cv2.erode(field_mask, np.ones((60, 60), np.uint8),iterations = 1)

    image = cv2.bitwise_and(image, image, mask=field_mask)
    image = cv2.GaussianBlur(image, (7, 7), 0)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    mask = cv2.Canny(gray,35,90,apertureSize = 3)

    field_mask = cv2.erode(field_mask, np.ones((10, 10), np.uint8),iterations = 1)

    mask = cv2.bitwise_and(mask, mask, mask=field_mask)

    return mask

def LineDetection(image, origin):

    message = "unknown"

    #_lines = cv2.HoughLinesP(image,rho = 1,theta = 1*np.pi/180,threshold = 120,minLineLength = 120,maxLineGap = 30)
    #if _lines is None:
        #_lines = cv2.HoughLinesP(image,rho = 1,theta = 1*np.pi/180,threshold = 90,minLineLength = 90,maxLineGap = 50)

    _lines = cv2.HoughLinesP(image,rho = 1,theta = 1*np.pi/180,threshold = 200,minLineLength = 200,maxLineGap = 30)
    if _lines is None:
        _lines = cv2.HoughLinesP(image,rho = 1,theta = 1*np.pi/180,threshold = 130,minLineLength = 130,maxLineGap = 50)

    if _lines is None:
        cv2.rectangle(origin, (10, 10), (260, 70), (255, 255, 255), cv2.FILLED)
        cv2.putText(origin, message, (50,50), cv2.FONT_HERSHEY_DUPLEX, 2, (0,0,0), 2)
        return origin

    lines = []
    for x in range(0, len(_lines)):
        x1, y1, x2, y2 = _lines[x][0];
        angle = np.degrees(np.arctan2(y1 - y2, x1 - x2))
        lines.append([x1, y1, x2, y2, angle])

    grouped_lines = GroupLines(lines)
    _, _, dif = LineStatistics(grouped_lines, "absolute")

    if dif > 2:
        segments = SegmentLines(grouped_lines)
        #SetLines(origin, segments[0], 2, (255,0,0))
        SetLines(origin, segments[0], 2, (28,25,21))
        if len(segments) > 1:
            SetLines(origin, segments[1], 2, (200,182,51))
            #SetLines(origin, segments[1], 2, (0,0,255))
            message = DetectPart(segments)
    else:
        #SetLines(origin, grouped_lines, 2, (0,0,255))
        SetLines(origin, grouped_lines, 2, (28,25,21))

    #cv2.rectangle(origin, (10, 10), (530, 70), (255, 255, 255), cv2.FILLED)
    #cv2.putText(origin, message, (50,50), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,0), 1)

    #return origin

    width = np.size(image, 1)
    height = np.size(image, 0)
    width_resize = width * 2
    height_resize = height * 2

    blank_image = np.zeros((height_resize, width_resize, 3), np.uint8)
    cv2.rectangle(blank_image,(0,0), (width_resize, height_resize), 0, cv2.FILLED)
    offset_X = int((width_resize - width) / 2)
    offset_Y = int((height_resize - height) / 2)

    SetLines(blank_image, grouped_lines, 1, (0,0,255), offset_X, offset_Y)

    #return blank_image

    blank_image = Figures(blank_image)
    return blank_image

    crop = blank_image[offset_Y:offset_Y+height, offset_X:offset_X+width]
    return crop
    _, crop = cv2.threshold(crop,50,255,cv2.THRESH_BINARY)
    crop = (255 - crop)
    origin = cv2.bitwise_and(origin, origin, mask=crop)

    return origin



def LengthenLine(input_x1, input_y1, input_x2, input_y2, width, height):
    p1 = (input_x1, input_y1)
    p2 = (input_x2, input_y2)

    theta = np.arctan2(p1[1]-p2[1], p1[0]-p2[0])
    endpt_x = int(p2[0] - 4000*np.cos(theta))
    endpt_y = int(p2[1] - 4000*np.sin(theta))

    startpt_x = int(p1[0] + 4000*np.cos(theta))
    startpt_y = int(p1[1] + 4000*np.sin(theta))

    line = LineString([(endpt_x, endpt_y), (startpt_x, startpt_y)])

    top_W = LineString([(0, 0), (width, 0)])
    bottom_W = LineString([(0, height), (width, height)])
    left_h = LineString([(0, 0), (0, height)])
    right_H = LineString([(width, 0), (width, height)])

    p1 = line.intersection(top_W)
    p2 = line.intersection(bottom_W)
    p3 = line.intersection(left_h)
    p4 = line.intersection(right_H)

    res = []
    if hasattr(p1, 'x'):
        res.append(p1)
    if hasattr(p2, 'x'):
        res.append(p2)
    if hasattr(p3, 'x'):
        res.append(p3)
    if hasattr(p4, 'x'):
        res.append(p4)

    startpt_x = res[0].x
    startpt_y = res[0].y
    endpt_x = res[1].x
    endpt_y = res[1].y

    return int(startpt_x), int(startpt_y), int(endpt_x), int(endpt_y)

def SetLines(image, lines, thickness, color, offset_x = 0, offset_y = 0):
    width = np.size(image, 1)
    height = np.size(image, 0)

    for x in range(len(lines)):
        x1,y1,x2,y2, _ = lines[x]
        x1,y1,x2,y2 = LengthenLine(x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y, width, height)
        cv2.line(image,(x1, y1),(x2, y2),color, thickness)



video = cv2.VideoCapture(PATH_TO_VIDEO)

message_place = 'unknown'

count = 0
while(video.isOpened()):
    count += 1
    ret, frame = video.read()
    frame_expanded = np.expand_dims(frame, axis=0)

    mask = PreprocImage(frame)
    frame = LineDetection(mask, frame)

    frame = cv2.resize(frame, (1800, 1020), interpolation=cv2.INTER_AREA)
    cv2.imshow('Object detector', frame)

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
