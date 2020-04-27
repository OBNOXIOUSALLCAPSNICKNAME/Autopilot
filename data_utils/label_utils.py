import os
import glob
import cv2
import sys
import numpy as np
import random as random
import xml.etree.cElementTree as ET

from image_utils import*


def add_object_to_xml(xml, cls, bbox):
    object = ET.SubElement(xml, "object")
    ET.SubElement(object, "name").text = cls
    ET.SubElement(object, "pose").text = "Unspecified"
    ET.SubElement(object, "truncated").text = "0"
    ET.SubElement(object, "difficult").text = "0"
    bndbox = ET.SubElement(object, "bndbox")
    ET.SubElement(bndbox, "xmin").text = str(int(bbox[0]))
    ET.SubElement(bndbox, "ymin").text = str(int(bbox[1]))
    ET.SubElement(bndbox, "xmax").text = str(int(bbox[2]))
    ET.SubElement(bndbox, "ymax").text = str(int(bbox[3]))

def create_xml(folder, filename, path, dimentions):
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = str(folder)
    ET.SubElement(annotation, "filename").text = str(filename)
    ET.SubElement(annotation, "path").text = str(path)
    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "Unknown"
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(dimentions[0])
    ET.SubElement(size, "height").text = str(dimentions[1])
    ET.SubElement(size, "depth").text = "3"
    ET.SubElement(annotation, "segmented").text = "0"
    return annotation

def parse_xml(xml):
    img_info = xml.find('size')
    size = [int(img_info[0].text), int(img_info[1].text)]

    objects = []
    for child in xml.iter('object'):
        for element in child.iter('bndbox'):
            cls = child[0].text
            coords = []
            for attr in element:
                coords.append(int(attr.text))
            objects.append([cls, coords])

    return objects


def save_as_txt(objects, path, classes, single_cls):
    with open(path, 'w') as f:
        for object in objects:
            cls, bbox = object
            if single_cls:
                row = "{0} {1:.6f} {2:.6f} {3:.6f} {4:.6f}".format(0, bbox[0], bbox[1], bbox[2], bbox[3])
            else:
                row = "{0} {1:.6f} {2:.6f} {3:.6f} {4:.6f}".format(classes.index(cls), bbox[0], bbox[1], bbox[2], bbox[3])
            f.write("%s\n" % row)

def save_as_xml(objects, path, dimentions, single_cls):
    name = path.replace('\\', '/').split('/')[-1].replace('.xml', '.jpg')
    folder = path.replace('\\', '/').split('/')[-2]
    img_path = path.replace('.xml', '.jpg')
    xml = create_xml(folder, name, path, dimentions)
    for object in objects:
        cls, bbox = object
        if single_cls:
            cls = '_object_'
        add_object_to_xml(xml, cls, bbox)
    ET.ElementTree(xml).write(path)
