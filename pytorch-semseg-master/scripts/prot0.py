
import os
import cv2
import sys
import time
import glob
import math
import shapely
import statistics
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

from collections import defaultdict
from collections import Counter
from math import atan2, degrees
from lxml import etree
from shapely.geometry import LineString
from scipy.spatial import distance
from skimage.transform import (hough_line, hough_line_peaks, probabilistic_hough_line)

sys.path.append("..")

directory_path = "C:/examples/"

xml_path = os.path.join(directory_path,'*.xml')
images_path = os.path.join(directory_path,'*g')

xml_files = glob.glob(xml_path)
image_files = glob.glob(images_path)

image_list = []
xml_list = []

for file in image_files:
    img = cv2.imread(file)
    image_list.append(img)

for file in xml_files:
    tree = ET.parse(file)
    root = tree.getroot()
    xml_list.append(root)

all_players_coord = []
for xml in xml_list:
    players_coord = []
    for child in xml.iter('object'):
        if child[0].text != 'ball':
            for element in child.iter('bndbox'):
                coords = ()
                for attr in element:
                    coords = coords + (int(attr.text),)
                players_coord.append(coords)
    all_players_coord.append(players_coord)


def DeletePlayers(image, labels, offset, color):
    for j in range(len(labels)):
        pt1 = (labels[j][0] - offset, labels[j][1] - offset)
        pt2 = (labels[j][2] + offset, labels[j][3] + offset)
        cv2.rectangle(image,pt1, pt2, color, cv2.FILLED)
    return image

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

    _1, labels, _2 = cv2.kmeans(np.float32(lines)[:, 4], 2, None, criteria, attempts, flags)

    segments = [[], []]
    for x in range(len(lines)):
        segments[labels[x][0]].append(lines[x])

    avg_1st, _, _ = LineStatistics(segments[0])
    avg_2nd, _, _ = LineStatistics(segments[1])
    if avg_2nd < avg_1st and (avg_1st < 0 or avg_2nd < 0):
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
        return abs(lines[0][4]), -1, -1

def BestfitLines(lines):
    all_angles = []
    angles_4p = [ [] for i in range(4) ]

    for x in range(len(lines)):
        x1,y1,x2,y2, angle = lines[x]
        if angle >= 0 and angle < 90:
            part = 0
        elif angle >= 90 and angle <= 180:
            part = 1
        elif angle >= -90 and angle < 0:
            part = 2
        elif angle >= -180 and angle < -90:
            part = 3
        angles_4p[part].append(angle)
        all_angles.append(part)

    avg_4p = [99,99,99,99]
    std_4p = [99,99,99,99]
    dif_4p = [99,99,99,99]

    #print('===========================================')
    for x in range(4):
        if len(angles_4p[x]) >= 2:
            avg_4p[x] = sum(angles_4p[x]) / len(angles_4p[x])
            std_4p[x] = statistics.stdev(angles_4p[x])
            dif_4p[x] = abs(max(angles_4p[x]) - min(angles_4p[x]))
            #print('avg = ', avg_4p[x], ' std = ', std_4p[x], ' dif = ', dif_4p[x])

    fitted_lines = []
    for x in range(len(lines)):
        x1,y1,x2,y2, angle = lines[x]
        part = all_angles[x]
        if (abs(avg_4p[part] - angle) >= std_4p[part] and std_4p[part] > 5) or (abs(avg_4p[part] - angle) >= std_4p[part]*2 and dif_4p[part] > 10):
        #if abs(avg_4p[part] - angle) >= std_4p[part] and std_4p[part] > 5 or abs(avg_4p[part] - angle) >= std_4p[part]*2:
            pass
        else:
            fitted_lines.append(lines[x])

    return fitted_lines

def GroupLines(input_lines, unique = True):
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
        angle_avg = np.rad2deg(np.arctan2(average_y1 - average_y2, average_x1 - average_x2))

        if unique == True:
            output.append([average_x1,average_y1,average_x2,average_y2,angle_avg])
        else:
            for x in range(len(group)):
                output.append([average_x1,average_y1,average_x2,average_y2,angle_avg])
        group.clear()
        i += 1

    return output

def SetLines(image, lines, thickness, color, offset_x = 0, offset_y = 0):
    width = np.size(image, 1)
    height = np.size(image, 0)

    for x in range(len(lines)):
        x1,y1,x2,y2, _ = lines[x]
        x1,y1,x2,y2 = LengthenLine(x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y, width, height)
        cv2.line(image,(x1, y1),(x2, y2),color, thickness)

def DetectPart(segments):
    angles_dif = []
    for i in range(len(segments[0])):
        angle_1st = abs(segments[0][i][4])
        for j in range(len(segments[1])):
            angle_2nd = abs(segments[1][j][4])
            angles_dif.append(abs(angle_1st - angle_2nd))
    avg_dif = sum(angles_dif) / len(angles_dif)

    if avg_dif < 25:
        avg_1st, _, _ = LineStatistics(segments[0], "absolute")
        avg_2nd, _, _ = LineStatistics(segments[1], "absolute")
        #print(avg_1st, avg_2nd)
        if avg_1st < avg_2nd:
            return "right"
        else:
            return "left"
    else:
        return "center"

def PreprocImage(image, players_coord):
    field_mask = HueMask(image)

    field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    #field_mask = DeletePlayers(field_mask, players_coord, 0, 255)
    field_mask = cv2.dilate(field_mask, np.ones((60, 60), np.uint8),iterations = 1)
    field_mask = cv2.erode(field_mask, np.ones((60, 60), np.uint8),iterations = 1)

    image = cv2.bitwise_and(image, image, mask=field_mask)
    image = cv2.GaussianBlur(image, (7, 7), 0)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    mask = cv2.Canny(gray,35,90,apertureSize = 3)

    field_mask = cv2.erode(field_mask, np.ones((10, 10), np.uint8),iterations = 1)

    mask = cv2.bitwise_and(mask, mask, mask=field_mask)
    #mask = DeletePlayers(mask, players_coord, 3, 0)

    return mask

def LineDetection(image, origin):

    message = "unknown"

    _lines = cv2.HoughLinesP(image,rho = 1,theta = 1*np.pi/180,threshold = 200,minLineLength = 200,maxLineGap = 30)
    if _lines is None:
        _lines = cv2.HoughLinesP(image,rho = 1,theta = 1*np.pi/180,threshold = 130,minLineLength = 130,maxLineGap = 50)

    if _lines is None:
        cv2.rectangle(origin, (10, 10), (260, 70), (255, 255, 255), cv2.FILLED)
        cv2.putText(origin, message, (50,50), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,0), 1)
        return origin

    lines = []
    for x in range(0, len(_lines)):
        x1, y1, x2, y2 = _lines[x][0];
        angle = np.rad2deg(np.arctan2(y1 - y2, x1 - x2))
        lines.append([x1, y1, x2, y2, angle])

    grouped_lines = GroupLines(lines)
    _, _, dif = LineStatistics(grouped_lines, "absolute")

    if dif > 5:
        segments = SegmentLines(grouped_lines)
        SetLines(origin, segments[0], 2, (255,0,0))
        if len(segments) > 1:
            SetLines(origin, segments[1], 2, (0,0,255))
            message = DetectPart(segments)
    else:
        SetLines(origin, grouped_lines, 2, (0,0,255))


    cv2.rectangle(origin, (10, 10), (260, 70), (255, 255, 255), cv2.FILLED)
    cv2.putText(origin, message, (50,50), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,0), 1)

    #cv2.imshow("1", origin)
    #cv2.waitKey(0)

    return origin


start = time.time()

for i in range(len(image_list)):
    name = 'image ' + str(i) + '.jpg'
    mask = PreprocImage(image_list[i], all_players_coord[i])
    image = LineDetection(mask, image_list[i])
    cv2.imwrite('C:/results/' + name, image)
    i+=1
end = time.time()
print ('done in', round((end - start), 2))
