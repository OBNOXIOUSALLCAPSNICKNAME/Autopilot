import os
import glob
import cv2
import sys
import numpy as np
import random as random


def scale_xml_bbox(bbox, scale, offset):
    bbox[0] = int(scale[0] * bbox[0]) + offset[0]
    bbox[1] = int(scale[1] * bbox[1]) + offset[1]
    bbox[2] = int(scale[0] * bbox[2]) + offset[0]
    bbox[3] = int(scale[1] * bbox[3]) + offset[1]
    return bbox

def scale_xml_labels(boxes, scale, offset):
    for i in range(len(boxes)):
        boxes[i][1] = scale_xml_bbox(boxes[i][1], scale, offset)
    return boxes


def xyxy_to_xywh(objects, size):
    out = []
    for object in objects:
        cls, bbox = object
        bbox = convert_to_xywh(bbox, size)
        out.append([cls, bbox])
    return out

def convert_to_xywh(bbox, size):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (bbox[0] + bbox[2])/2.0
    y = (bbox[1] + bbox[3])/2.0
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh

    return [x,y,w,h]


def get_scale_val(old_dims, new_dims, keep_ar):
    if keep_ar == False:
        scale_x = float(new_dims[0]) / float(old_dims[0])
        scale_y = float(new_dims[1]) / float(old_dims[1])
        return [scale_x, scale_y], [0,0]
    else:
        new_ar = float(new_dims[1])/float(new_dims[0])
        old_ar = float(new_dims[1])/float(new_dims[0])
        offset_x = 0
        offset_y = 0
        if new_ar < old_ar:
            scale = float(new_dims[1]) / float(old_dims[1])
            new_w = int(float(old_dims[0]) * scale)
            offset_x = int((new_dims[0] - new_w) / 2)
        else:
            scale = float(new_dims[0]) / float(old_dims[0])
            new_h = int(float(old_dims[1]) * scale)
            offset_y = int((new_dims[1] - new_h) / 2)
        return [scale, scale], [offset_x, offset_y]

def resize_image(image, resolution, scale, offset):
    result = np.full((resolution[1], resolution[0], 3), [128, 128, 128], dtype=np.uint8)
    image = cv2.resize(image, None, fx=scale[0], fy=scale[1])
    (h, w) = image.shape[:2]
    result[offset[1]:offset[1]+h, offset[0]:offset[0]+w] = image
    return result
