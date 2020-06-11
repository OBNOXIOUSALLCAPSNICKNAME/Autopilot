
import os
import glob
import cv2
import sys
import time
import numpy as np
import tensorflow as tf
import xml.etree.cElementTree as ET

sys.path.append("..")

input_dir = 'C:/examples/'
output_dir = 'C:/results/'

name_pref = 'ss p '

xml_list = []
xml_path = os.path.join(input_dir,'*.xml')
xml_files = glob.glob(xml_path)

for file in xml_files:
    tree = ET.parse(file)
    root = tree.getroot()
    xml_list.append(root)

def MoveImages():
    for xml in xml_list:
        for child in xml.iter("path"):
            path = str(child.text)
            split_path = path.split('\\')
            cv2.imwrite(output_dir + split_path[len(split_path) - 1], cv2.imread(path))

def Rename():
    i = 0
    for xml in xml_list:
        i += 1
        for child_path in xml.iter("path"):
            path = str(child_path.text)
            path_split = path.split('\\')
            old_name = path_split[len(path_split) - 1]
            img_path = output_dir + name_pref + str(i) + '.jpg'
            xml_path = output_dir + name_pref + str(i) + '.xml'
            child_path.text = img_path.replace('\\', '/')
            child_filename = list(xml.iter("filename"))[0]
            child_filename.text = name_pref + str(i) + '.jpg'

            ET.ElementTree(xml).write(xml_path)
            cv2.imwrite(img_path, cv2.imread(input_dir + old_name))


Rename()
#MoveImages()
print('done')
