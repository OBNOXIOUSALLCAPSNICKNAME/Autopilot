import os
import glob
import cv2
import sys
import numpy as np

import scipy.misc as m


old_colors = [
[0, 0, 128],
[0, 0, 255],
[0, 128, 128],
[0, 128, 255],
[0, 220, 220],
[32, 11, 119],
[35, 142, 107],
[60, 0, 0],
[60, 20, 220],
[100, 100 ,150],
[128, 64 ,128],
[142, 0, 0],
[153, 153, 190],
[160, 78, 128],
[180, 130, 70],
[180, 165, 180],
[190, 132, 178],
[204, 0, 102],
[229, 255, 201],
[255, 255, 0],
[255, 255, 255],

[0, 0, 0],
[255, 255, 255]
]

new_colors = [
[1, 1, 1],
[2, 2, 2],
[3, 3, 3],
[4, 4, 4],
[5, 5, 5],
[6, 6, 6],
[7, 7, 7],
[8, 8, 8],
[9, 9, 9],
[10, 10, 10],
[11, 11, 11],
[12, 12, 12],
[13, 13, 13],
[14, 14, 14],
[15, 15, 15],
[16, 16, 16],
[17, 17, 17],
[18, 18, 18],
[19, 19, 19],
[20, 20, 20],
[21, 21, 21],

[0, 0, 0],
[0, 0, 0]
]

def list_files(input_path):
    out = [[], []]
    for root, dirs, files in os.walk(input_path):
        root = root.replace('\\', '/')
        if not root.endswith('/'):
            root += '/'

        imgs = glob.glob(root+'*g')
        imgs = [path.replace('\\', '/') for path in imgs]

        lbls = glob.glob(root.replace('Image', 'Label')+'*g')
        lbls = [path.replace('\\', '/') for path in lbls]

        out[0] += imgs
        out[1] += lbls

    return out

def resize_crop(img):
    f = 0
    h, w = img.shape[:2]
    y = int(float(h) * 0.6)

    cropped = img[y:h, 0:w]

    result = cv2.resize(cropped, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)

    return result


def rgb_to_cls(img):
    h, w = img.shape[:2]
    result = np.full((h, w), 0, dtype=np.uint8)

    for x in range(len(old_colors)):
        old_color = np.asarray(old_colors[x])
        new_color = np.asarray(new_colors[x])
        result[np.where((img == old_color).all(axis = 2))] = new_color[0]

    return result



input = 'E:/Autopilot/datasets/temp/Image'
output = 'E:/Autopilot/datasets/result/'

data = list_files(input)

rnd_indices = random.sample(range(0, len(data[0])), len(data[0]))

range_train = len(data[0]) * 0.7
range_test = len(data[0]) * 0.9

for x in range(len(rnd_indices)):
    i = rnd_indices[x]

    img = cv2.imread(data[0][i])
    lbl = cv2.imread(data[1][i])

    lbl = resize_crop(lbl)
    lbl = rgb_to_cls(lbl)


    if x < range_train:
        cv2.imwrite('{}/train/{}.png'.format(output, x), img)
        cv2.imwrite('{}/trainannot/{}.png'.format(output, x), lbl)
    elif x > range_train and x < range_test:
        cv2.imwrite('{}/test/{}.png'.format(output, x), img)
        cv2.imwrite('{}/testannot/{}.png'.format(output, x), lbl)
    else:
        cv2.imwrite('{}/val/{}.png'.format(output, x), img)
        cv2.imwrite('{}/valannot/{}.png'.format(output, x), lbl)
