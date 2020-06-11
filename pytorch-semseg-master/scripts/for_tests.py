
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

def GetSlopes(pts):
    TW = np.rad2deg(np.arctan2(pts[0][1] - pts[1][1], pts[0][0] - pts[1][0]))
    BW = np.rad2deg(np.arctan2(pts[2][1] - pts[3][1], pts[2][0] - pts[3][0]))
    LH = np.rad2deg(np.arctan2(pts[0][1] - pts[2][1], pts[0][0] - pts[2][0]))
    RH = np.rad2deg(np.arctan2(pts[1][1] - pts[3][1], pts[1][0] - pts[3][0]))
    #print('TW: ', TW)
    #print('BW: ', BW)
    #print('LH: ', LH)
    #print('RH: ', RH)

def Figures(image):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image,60,255,cv2.THRESH_BINARY)
    contorus, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contur in contorus:
        approx = cv2.approxPolyDP(contur, 0.01*cv2.arcLength(contur, True), True)
        if len(approx) == 4:
            new_approx = order_points(np.vstack(approx))
            if CheckShape(new_approx):
                GetSlopes(new_approx)
                aspect_ratio = WarpPersp(new_approx, image)
                text_point = (int((new_approx[0][0] + new_approx[3][0])/2), int((new_approx[1][1] + new_approx[2][1])/2))

                cv2.drawContours(image, [approx], 0, (255), 3)
                cv2.putText(image, str(aspect_ratio), text_point, cv2.FONT_HERSHEY_DUPLEX, 1, (255), 1)
                #show = cv2.resize(image, (1920, 1080))
                #cv2.imshow("1", show)
                #cv2.waitKey(0)

    return image

def Figuress(image):
    #image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    temp = image.copy()
    _, temp = cv2.threshold(temp,255,255,cv2.THRESH_BINARY)

    _, image = cv2.threshold(image,60,255,cv2.THRESH_BINARY)
    contorus, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    i = 0
    for contur in contorus:
        i += 1
        approx = cv2.approxPolyDP(contur, 0.005*cv2.arcLength(contur, True), False)
        if len(approx) < 225:
            text_point = (np.vstack(approx)[0][0], np.vstack(approx)[0][1])
            cv2.putText(temp, str(len(approx)), text_point, cv2.FONT_HERSHEY_DUPLEX, 1, (255), 1)
            cv2.drawContours(temp, [approx], 0, (255), 1)

    return temp

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

from collections import defaultdict
def segment_by_angle_kmeans(lines, k=2, **kwargs):
    """Groups lines based on angle with k-means.
    Uses k-means on the coordinates of the angle on the unit circle
    to segment `k` angles inside `lines`.
    """

    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # returns angles in [0, pi] in radians
    angles = np.array([line[0][1] for line in lines])
    # multiply the angles by two and find coordinates of that angle
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)] for angle in angles], dtype=np.float32)

        # run kmeans on the coords
    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    print(labels, centers)
    labels = labels.reshape(-1)  # transpose to row vec

        # segment lines based on their kmeans label
    segmented = defaultdict(list)
    for i, line in zip(range(len(lines)), lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    return segmented

def LineDetection(image, players_coord):
    start = time.time()

    temp = image.copy()
    output = image.copy()

    field_mask = HueMask(image)

    #cv2.imshow("1", field_mask)
    #cv2.waitKey(0)

    field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    field_mask = DeletePlayers(field_mask, players_coord, 0, 255)
    field_mask = cv2.erode(field_mask, np.ones((60, 60), np.uint8),iterations = 1)
    field_mask = cv2.dilate(field_mask, np.ones((70, 70), np.uint8),iterations = 1)

    #cv2.imshow("1", field_mask)
    #cv2.waitKey(0)

    image = cv2.bitwise_and(image, image, mask=field_mask)

    #cv2.imshow("1", image)
    #cv2.waitKey(0)

    image = cv2.GaussianBlur(image, (7, 7), 0)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    mask = cv2.Canny(gray,45,90,apertureSize = 3)

    #cv2.imshow("1", mask)
    #cv2.waitKey(0)

    field_mask = cv2.erode(field_mask, np.ones((12, 12), np.uint8),iterations = 1)

    mask = cv2.bitwise_and(mask, mask, mask=field_mask)
    mask = DeletePlayers(mask, players_coord, 3, 0)

    #cv2.imshow("1", mask)
    #cv2.waitKey(0)

    _lines = cv2.HoughLinesP(mask,rho = 1,theta = 1*np.pi/180,threshold = 200,minLineLength = 200,maxLineGap = 30)
    if _lines is None:
        _lines = cv2.HoughLinesP(mask,rho = 1,theta = 1*np.pi/180,threshold = 130,minLineLength = 130,maxLineGap = 50)

    lines = []
    for x in range(0, len(_lines)):
        lines.append(_lines[x][0])

    #cv2.imshow("1", temp)
    #cv2.waitKey(0)
    #return temp, temp

    width = np.size(image, 1)
    height = np.size(image, 0)
    width_resize = width * 2
    height_resize = height * 2

    blank_image = np.zeros((height_resize, width_resize, 3), np.uint8)
    cv2.rectangle(blank_image,(0,0), (width_resize, height_resize), 0, cv2.FILLED)
    temp = blank_image.copy()
    offset_X = int((width_resize - width) / 2)
    offset_Y = int((height_resize - height) / 2)

    fit_lines = BestfitLines(lines)
    grouped_lines = GroupLines(fit_lines)
    SegmentLines(grouped_lines)
    SetLines(blank_image, grouped_lines, 2, offset_X, offset_Y)

    #cv2.imshow("1", blank_image)
    #cv2.waitKey(0)

    show = cv2.resize(blank_image, (1920, 1080))
    #cv2.imshow("1", show)
    #cv2.waitKey(0)

    blank_image = Figures(blank_image)
    crop = blank_image[offset_Y:offset_Y+height, offset_X:offset_X+width]
    _, crop = cv2.threshold(crop,50,255,cv2.THRESH_BINARY)
    crop = (255 - crop)
    output = cv2.bitwise_and(output, output, mask=crop)

    end = time.time()
    print (round((end - start), 2))

    #cv2.imshow("1", output)
    #cv2.waitKey(0)

    return blank_image, output

def SegmentLines(lines, **kwargs):
    if len(lines) < 2:
        return
    angles = []
    for x in range(len(lines)):
        x1,y1,x2,y2 = lines[x]
        angle = np.rad2deg(np.arctan2(y1 - y2, x1 - x2))
        angles.append(angle)

    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    _, labels, _ = cv2.kmeans(np.float32(angles), 2, None, criteria, attempts, flags)

    segments = [[], []]
    for x in range(len(lines)):
        x1,y1,x2,y2 = lines[x]
        angle = np.rad2deg(np.arctan2(y1 - y2, x1 - x2))
        segments[labels[x][0]].append((x1,y1,x2,y2,angle))

    LineStatistics(segments)
    #return segments

def LineStatistics(segmented_lines):
    stats = []
    for segment in segmented_lines:
        if len(segment) > 1:
            sliced = np.asarray(segment)
            avg = sum(sliced[:, 4]) / len(segment)
            std = statistics.stdev(sliced[:, 4])
            dif = abs(max(sliced[:, 4]) - min(sliced[:, 4]))
            stats.append((avg, std, dif))
            print(avg, std, dif)




def BestfitLines(lines):
    angles_neg = []
    angles_pos = []
    all_angles = []
    angles_4p = [ [] for i in range(4) ]

    for x in range(len(lines)):
        x1,y1,x2,y2 = lines[x]
        angle = np.rad2deg(np.arctan2(y1 - y2, x1 - x2))
        if angle >= 0 and angle < 90:
            part = 0
        elif angle >= 90 and angle <= 180:
            part = 1
        elif angle >= -90 and angle < 0:
            part = 2
        elif angle >= -180 and angle < -90:
            part = 3
        angles_4p[part].append(angle)
        all_angles.append([angle, part])

    avg_4p = [99,99,99,99]
    std_4p = [99,99,99,99]
    dif_4p = [99,99,99,99]

    print('===========================================')
    for x in range(4):
        if len(angles_4p[x]) >= 2:
            avg_4p[x] = sum(angles_4p[x]) / len(angles_4p[x])
            std_4p[x] = statistics.stdev(angles_4p[x])
            dif_4p[x] = abs(max(angles_4p[x]) - min(angles_4p[x]))
            #print('avg = ', avg_4p[x], ' std = ', std_4p[x], ' dif = ', dif_4p[x])

    fitted_lines = []
    for x in range(len(lines)):
        x1,y1,x2,y2 = lines[x]
        angle, part = all_angles[x]
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
        i_x1,i_y1,i_x2,i_y2 = lines[i]
        angle_i = np.rad2deg(np.arctan2(i_y1 - i_y2, i_x1 - i_x2))
        group.append(lines[i].copy())
        while j < length:
            if i != j:
                j_x1,j_y1,j_x2,j_y2 = lines[j]

                distance = LineDistance((i_x1,i_y1), (i_x2,i_y2), (j_x1,j_y1))
                angle_j = np.rad2deg(np.arctan2(j_y1 - j_y2, j_x1 - j_x2))
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
        if unique == True:
            output.append([average_x1,average_y1,average_x2,average_y2])
        else:
            for x in range(len(group)):
                output.append([average_x1,average_y1,average_x2,average_y2])
        group.clear()
        i += 1

    return output

def SetLines(image, lines, thickness, color=(255,0,0),  offset_x = 0, offset_y = 0):
    width = np.size(image, 1)
    height = np.size(image, 0)

    for x in range(len(lines)):
        x1,y1,x2,y2 = lines[x]
        angle = np.rad2deg(np.arctan2(y1 - y2, x1 - x2))
        x1,y1,x2,y2 = LengthenLine(x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y, width, height)
        cv2.line(image,(x1, y1),(x2, y2),color, thickness)


start = time.time()

for i in range(len(image_list)):
    name = 'image ' + str(i) + '.jpg'
    name_field = 'image ' + str(i) + ' 1' + '.jpg'
    black, image = LineDetection(image_list[i], all_players_coord[i])
    cv2.imwrite('C:/results/' + name, image)
    cv2.imwrite('C:/results/' + name_field, black)
    i+=1
end = time.time()
print ('done in', round((end - start), 2))
