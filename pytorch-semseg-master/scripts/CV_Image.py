
import os
import cv2
import sys
import time
import glob
import shapely
import numpy as np
import tensorflow as tf
import matplotlib.colors as colors
import xml.etree.ElementTree as ET
from shapely.geometry import LineString

from lxml import etree
from matplotlib import pyplot as plt
from scipy.spatial import distance
from pyclustering.cluster import cluster_visualizer_multidim

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

def GetAverage(img):
    avg_color_per_row = np.average(img, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)

    height, width, channels = img.shape

    for x in range(0, width):
        for y in range(0, height):
            img[y,x] = avg_color
    return img

def NormalizeHist(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img

def Something(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    #green range
    lower_green = np.array([20,0, 0])
    upper_green = np.array([70, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel_Close = np.ones((6, 6), np.uint8)
    kernel_Open = np.ones((6, 6), np.uint8)
    kernel_Erode = np.ones((5, 5), np.uint8)
    kernel_Dilation = np.ones((6, 6), np.uint8)

    #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_Close)
    #mask = cv2.erode(mask, kernel_Erode, iterations=1)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_Open)
    mask = cv2.dilate(mask,kernel_Dilation,iterations = 1)
    mask = cv2.medianBlur(mask, 9)

    #res = cv2.bitwise_and(img, img, mask=mask)

    #img = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
    #img = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    return mask

def TeamDetection(image):

    tmp = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(image)
    rgba = [b, g, r, alpha]
    image = cv2.merge(rgba, 4)

    #image = NormalizeHist(image)
    #image = cv2.medianBlur(image, 5)
    image = cv2.GaussianBlur(image, (3, 3), 0)

    return image

def ComputeDistance(data):
    list = []
    for i in range(len(data)):
        for j in range(len(data)):
            if i != j and (i, j) not in list and (j, i) not in list:
                    print (str(i) + ' ' + str(j) + '\t', distance.euclidean(data[i], data[j]))
                    list.append((i, j))

def Clustering(images):
    i = 0
    data_bsas = []
    for img in images:
        i += 1
        #img = NormalizeHist(img)
        img = cv2.medianBlur(img, 5)

        array = np.asarray(img)

        temp = []
        for subarray in array:
            for element in subarray:
                if (element[0] + element[1] + element[2]) / 3 > 10:
                    temp.append(element)

        temp = np.asarray(temp)
        arr = (temp.astype(float)) / 255.0
        img_hsv = colors.rgb_to_hsv(arr[..., :3])
        lu1 = img_hsv[..., 0].flatten()
        list = plt.hist(lu1 * 360, bins=360, range=(0.0, 360.0))

        toshow = []

        sum = 0
        for element in list[0]:
            sum += element

        #"""
        for  element in list[0]:
            if 100 * element / sum > 0:
                toshow.append(1)
            else:
                toshow.append(0)
        #"""

        #for  element in list[0]:
            #toshow.append(100 * element / sum)

        #for element in list[0]:
            #toshow.append(int(element))

        name = 'C:/results/' + str(i) + ' plot'
        plt.savefig(name)

        #plt.show()
        plt.cla()

        data_bsas.append(toshow)

    ComputeDistance(data_bsas)

    #"""
    from pyclustering.cluster.bsas import bsas, bsas_visualizer
    from pyclustering.utils import read_sample
    from pyclustering.samples.definitions import SIMPLE_SAMPLES

    max_clusters = 3
    threshold = 1.0

    bsas_instance = bsas(data_bsas, max_clusters, threshold, 'euclidean')
    bsas_instance.process()

    clusters = bsas_instance.get_clusters()
    representatives = bsas_instance.get_representatives()

    print (clusters)

    #bsas_visualizer.show_clusters(data_bsas, clusters, representatives)
    #"""

# это вообще стыд, но нужно хотя бы для проверки, будет ли вообще работать такой способ или нет
def HueMask(image):
    temp = image.copy()
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

    width = np.size(image, 1)
    height = np.size(image, 0)
    total = width * height

    hist = []
    for i in range(180):
        hist.append(0)
    for x in range(width):
        for y in range(height):
            hist[int(hsv[y,x,0])] += 1

    max_val = hist.index(max(hist))
    for i in range(180):
        hist[i] = 100 * hist[i] / total

    #"""
    for x in range(width):
        for y in range(height):
            index = int(int(hsv[y,x,0]))
            if hist[index] >= 0.1 and index < max_val + 10 and index > max_val - 20:
                hsv[y,x] = [0,0,0]
            else:
                hsv[y,x] = [0,0,255]

    image = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = (255 - image)
    #"""

    """
    _hsv = cv2.cvtColor(temp,cv2.COLOR_BGR2HSV)
    lower_green = np.array([max_val - 20,0, 0])
    upper_green = np.array([max_val + 10, 255, 255])
    mask = cv2.inRange(_hsv, lower_green, upper_green)

    #"""

    return image
    #return mask

# велосипед для удлинения линий
def LengthenLine(input_x1, input_y1, input_x2, input_y2, len):
    p1 = (input_x1, input_y1)
    p2 = (input_x2, input_y2)

    theta = np.arctan2(p1[1]-p2[1], p1[0]-p2[0])
    endpt_x = int(p2[0] - len*np.cos(theta))
    endpt_y = int(p2[1] - len*np.sin(theta))

    startpt_x = int(p1[0] + len*np.cos(theta))
    startpt_y = int(p1[1] + len*np.sin(theta))

    line = LineString([(endpt_x, endpt_y), (startpt_x, startpt_y)])
    line1 = LineString([(0, 0), (1920, 0)])
    line2 = LineString([(0, 1080), (1920, 1080)])
    line3 = LineString([(0, 0), (0, 1080)])
    line4 = LineString([(1920, 0), (1920, 1080)])

    p1 = line.intersection(line1)
    p2 = line.intersection(line2)
    p3 = line.intersection(line3)
    p4 = line.intersection(line4)

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

def LineDetection(image, players_coord):
    output = image.copy()
    temp = image

    #cv2.imshow("lul", image)
    #cv2.waitKey(0)

    image = cv2.GaussianBlur(image, (3, 3), 0)
    field_mask = HueMask(image)

    #cv2.imshow("lul", field_mask)
    #cv2.waitKey(0)

    field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))

    #cv2.imshow("lul", field_mask)
    #cv2.waitKey(0)

    for j in range(len(players_coord)):
        pt1 = (players_coord[j][0], players_coord[j][1])
        pt2 = (players_coord[j][2], players_coord[j][3])
        cv2.rectangle(field_mask,pt1, pt2, 255, cv2.FILLED)

    field_mask = cv2.erode(field_mask, np.ones((60, 60), np.uint8),iterations = 1)
    field_mask = cv2.dilate(field_mask, np.ones((60, 60), np.uint8),iterations = 1)

    #cv2.imshow("lul", field_mask)
    #cv2.waitKey(0)

    image = cv2.bitwise_and(image, image, mask=field_mask)

    #cv2.imshow("lul", image)
    #cv2.waitKey(0)

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    mask = cv2.Canny(gray,45,90,apertureSize = 3)

    #cv2.imshow("lul", mask)
    #cv2.waitKey(0)

    field_mask = cv2.erode(field_mask, np.ones((12, 12), np.uint8),iterations = 1)
    mask = cv2.bitwise_and(mask, mask, mask=field_mask)

    #cv2.imshow("lul", mask)
    #cv2.waitKey(0)

    for j in range(len(players_coord)):
        pt1 = (players_coord[j][0]-3, players_coord[j][1]-3)
        pt2 = (players_coord[j][2]+3, players_coord[j][3]+3)
        cv2.rectangle(mask,pt1, pt2, 0, cv2.FILLED)

    #cv2.imshow("lul", mask)
    #cv2.waitKey(0)

    lines = cv2.HoughLinesP(mask,rho = 1,theta = 1*np.pi/180,threshold = 100,minLineLength = 80,maxLineGap = 10)
    output = PutLines(output, lines)

    cv2.imshow("lul", output)
    cv2.waitKey(0)

    return output

def PutLines(image, lines):
    if lines is None:
        return image
    else:

        width = np.size(image, 1)
        height = np.size(image, 0)

        for x in range(0, len(lines)):
            for x1,y1,x2,y2 in lines[x]:
                x1,y1,x2,y2 = LengthenLine(x1,y1,x2,y2, 2000)
                lines[x] = [x1,y1,x2,y2]

        for i in range(0, len(lines)):
            for j in range(0, len(lines)):
                if i != j:
                    i_x1,i_y1,i_x2,i_y2 = lines[i][0]
                    j_x1,j_y1,j_x2,j_y2 = lines[j][0]

                    gradient1 = (i_y1-i_y2)/(i_x1-i_x2)
                    gradient2 = (j_y1-j_y2)/(j_x1-j_x2)

                    dist_x = abs(i_x1 - j_x1)
                    dist_y = abs(i_y1 - j_y1)
                    #shapely.geometry
                    distance = LineDistance((i_x1,i_y1), (i_x2,i_y2), (j_x1,j_y1))
                    dif_grad = abs(gradient1 - gradient2)

                    if (dif_grad < 0.001 and distance < 20) or (dif_grad < 0.03 and dif_grad > 0.001 and distance < 40):
                        average_x1 = (i_x1 + j_x1)/2
                        average_x2 = (i_x2 + j_x2)/2

                        average_y1 = (i_y1 + j_y1)/2
                        average_y2 = (i_y2 + j_y2)/2

                        lines[j][0] = [average_x1,average_y1,average_x2,average_y2]
                        if j != i:
                            j -= 1

        temp = []
        for i in range(0, len(lines)):
            temp.append(lines[i][0])
        unique_lines = [list(x) for x in set(tuple(x) for x in temp)]

        for x in range(len(unique_lines)):
            x1,y1,x2,y2 = unique_lines[x]
            cv2.line(image,(x1,y1),(x2,y2),(255,0,0),1)

        for i in range(len(unique_lines)):
            for j in range(len(unique_lines)):
                i_x1,i_y1,i_x2,i_y2 = unique_lines[i]
                j_x1,j_y1,j_x2,j_y2 = unique_lines[j]

                theta1 = np.arctan2(i_y1-i_y2, i_x1-i_x2)
                theta2 = np.arctan2(j_y1-j_y2, j_x1-j_x2)

                if abs(theta1 - theta2) > 1:
                    line1 = LineString([(i_x1, i_y1), (i_x2, i_y2)])
                    line2 = LineString([(j_x1, j_y1), (j_x2, j_y2)])
                    p = line1.intersection(line2)
                    if hasattr(p, 'x'):
                        cv2.line(image,(int(p.x), int(p.y)),(int(p.x), int(p.y)),(0,255,255),8)
        return image


start = time.time()
for i in range(len(image_list)):
    name = 'image ' + str(i) + '.jpg'
    image = image_list[i]
    image = LineDetection(image, all_players_coord[i])

    #image = cv2.resize(image, None, fx=10, fy=10)
    cv2.imwrite('C:/results/' + name, image)
    i+=1

print ('done')
end = time.time()
print (round((end - start), 2))
