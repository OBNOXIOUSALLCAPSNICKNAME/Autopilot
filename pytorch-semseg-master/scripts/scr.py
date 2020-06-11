import numpy as np
import cv2
import os
import glob
import itertools
import  pandas as pd
import xml.etree.ElementTree as ET

from lxml import etree
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD

directory_path = "C:/examples/"

def dist(x1, x2):
    d = 0.0
    for i in range(len(x1)):
        d += np.abs(x1[i] - x2[i])
    return d/float(len(x1))

def image_compress(img, n_components=1, shape=(20,50)):
    img = cv2.resize(img, shape)
    #show = cv2.resize(img, (200, 500))
    #cv2.imshow('1', show)
    #cv2.waitKey()
    img_r = np.reshape(img, (img.shape[0],
                             img.shape[1]*
                             img.shape[2]))

    pca = TruncatedSVD(n_components).fit(img_r)
    img_c1 = pca.transform(img_r)
    return img_c1.reshape(img_c1.shape[0]*img_c1.shape[1])

def GrapData():
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
        index = 0
        for child in xml.iter('object'):
            index += 1
            if child[0].text != 'ball':
                for element in child.iter('bndbox'):
                    coords = ()
                    for attr in element:
                        coords = coords + (int(attr.text),)
                    players_coord.append([coords, child[0].text, index])
        all_players_coord.append(players_coord)

    return image_list, all_players_coord

def GrabPlayers(image, coords):
    players = []
    for j in range(len(coords)):
        bndbox,label,index = coords[j]
        crop_img = image[bndbox[1]:bndbox[3], bndbox[0]:bndbox[2]]
        dec_img = image_compress(crop_img)
        players.append([dec_img, label, index])

    return players

def GroupPlayers(players):
    output = []

    unique, counts = np.unique(np.asarray(players)[:,1],return_counts=True)
    for name in unique:
        group = [x for x in players if x[1] == name]
        output.append(group)

    return output

def CalcDistances(groups):
    for group in groups:
        distances = []
        for first, second in itertools.combinations(group, 2):
            _distance = dist(first[0], second[0])
            distances.append(_distance)
        print(
        "\n Reflection to group \"{}\" \n avg: {} \n min: {} \n max: {}"
        .format(group[0][1], '%.3f'%(sum(distances) / len(distances)), '%.3f'%min(distances), '%.3f'%max(distances))
        )

    for group1, group2 in itertools.combinations(groups, 2):
        distances = []
        combinations = itertools.product(group1, group2)
        for pair in combinations:
            _distance = dist(pair[0][0], pair[1][0])
            distances.append(_distance)
        print(
        "\n Group \"{}\" to Group \"{}\" \n avg: {} \n min: {} \n max: {}"
        .format(group1[0][1], group2[0][1], '%.3f'%(sum(distances) / len(distances)), '%.3f'%min(distances), '%.3f'%max(distances))
        )

    print('====================================================')

def ClusteringK_means(players):
    slice = list(np.asarray(players)[:,0])
    kmeans = KMeans(n_clusters=3, random_state=0).fit(slice)

    for x in range(len(players)):
        print(kmeans.labels_[x], "    ", players[x][1])


images, coords = GrapData()

for i in range(len(images)):
    players = GrabPlayers(images[i], coords[i])
    groups = GroupPlayers(players)
    ClusteringK_means(players)
    CalcDistances(groups)
    i += 1
