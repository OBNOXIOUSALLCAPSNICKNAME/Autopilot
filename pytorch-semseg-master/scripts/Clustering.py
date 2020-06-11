import os
import cv2
import sys
import time
import glob
import numpy as np
import tensorflow as tf
import matplotlib.colors as colors
import xml.etree.ElementTree as ET

from lxml import etree
from matplotlib import pyplot as plt

from scipy.spatial import distance
from pyclustering.cluster import cluster_visualizer_multidim

from pyclustering.cluster.bsas import bsas, bsas_visualizer
from pyclustering.utils import read_sample
from pyclustering.samples.definitions import SIMPLE_SAMPLES

sys.path.append("..")

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

def K_means(img):
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 6
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    img = res.reshape((img.shape))
    return img

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
    lower_green = np.array([35,15, 15])
    upper_green = np.array([50, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    res = cv2.bitwise_and(img, img, mask=mask)

    img = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
    img = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    return img

def Mask(image):
    mask = MySobel(image)

    ret, mask  = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)

    kernel_Close = np.ones((7, 7), np.uint8)
    kernel_Erode = np.ones((2, 2), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_Close)
    #mask = cv2.erode(mask, kernel_Erode, iterations=1)

    return mask

def ComputeDistance(data):
    list = []
    for i in range(len(data)):
        for j in range(len(data)):
            if i != j and (i, j) not in list and (j, i) not in list:
                    print (str(i) + ' ' + str(j) + '\t', distance.euclidean(data[i], data[j]))
                    list.append((i, j))

def Clustering(images):
    start = time.time()

    i = 0
    data_bsas = []
    for img in images:
        i += 1
        array = np.asarray(img)
        temp = []
        for subarray in array:
            for element in subarray:
                if (element[0] + element[1] + element[2]) / 3 > 5:
                    temp.append(element)

        temp = np.asarray(temp)
        arr = (temp.astype(float)) / 255.0
        img_hsv = colors.rgb_to_hsv(arr[..., :3])
        lu1 = img_hsv[..., 1].flatten()
        list = plt.hist(lu1 * 360, bins=360, range=(0.0, 360.0), histtype='step')

        hsv_hist = []

        sum = 0
        for element in list[0]:
            sum += element

        """
        for  element in list[0]:
            if 100 * element / sum > 0:
                toshow.append(1)
            else:
                toshow.append(0)
        #"""

        for  element in list[0]:
            hsv_hist.append(100 * element / sum)

        #for element in list[0]:
            #toshow.append(int(element))

        name = 'C:/results/' + str(i) + ' plot'
        plt.savefig(name)

        #plt.show()
        plt.cla()

        data_bsas.append(hsv_hist)

    #ComputeDistance(data_bsas)

    """
    max_clusters = 3
    threshold = 1.0

    bsas_instance = bsas(data_bsas, max_clusters, threshold, 'euclidean')
    bsas_instance.process()

    clusters = bsas_instance.get_clusters()
    representatives = bsas_instance.get_representatives()

    print (clusters)
    #bsas_visualizer.show_clusters(data_bsas, clusters, representatives)
    """

    end = time.time()
    print (round((end - start), 2))



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

players_coord = []
for xml in xml_list:
    for child in xml.iter('object'):
        if child[0].text != 'ball':
            for element in child.iter('bndbox'):
                coords = ()
                for attr in element:
                    coords = coords + (int(attr.text),)
                players_coord.append(coords)


for i in range(len(image_list)):
    img = image_list[i]
    mask = Mask(img)
    players = []
    for j in range(len(players_coord)):
        player_img = img[players_coord[j][1]:players_coord[j][3], players_coord[j][0]:players_coord[j][2]]
        player_mask = mask[players_coord[j][1]:players_coord[j][3], players_coord[j][0]:players_coord[j][2]]

        player = NormalizeHist(player_img)
        player = cv2.medianBlur(player, 5)
        #player = cv2.bitwise_and(player, player, mask=player_mask)

        array = np.asarray(img)
        temp = []

        players.append(player)
        name = 'player ' + str(j)
        cv2.imwrite('C:/results/' + name + '.jpg', player)

    Clustering(players)

# startY endY startX endX
# xmin ymin xmax ymax
print('done')
