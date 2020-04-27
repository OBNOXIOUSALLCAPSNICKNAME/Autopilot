import os
import glob
import cv2
import sys
import numpy as np

import scipy.misc as m

def list_files(input_path, output_path):
    out = []
    for root, dirs, files in os.walk(input_path):
        root = root.replace('\\', '/')
        if not root.endswith('/'):
            root += '/'

        input_img = glob.glob(root+'*g')
        input_img = [path.replace('\\', '/') for path in input_img]
        output_img = [path.replace(input_path, output_path) for path in input_img]

        root = root.replace(input_path, output_path)

        if root != output_path:
            out.append([root, input_img, output_img])

    return out

def resize_crop(img):
    f = 0
    h, w = img.shape[:2]
    y = int(float(h) * 0.6)

    cropped = img[y:h, 0:w]

    result = cv2.resize(cropped, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)

    return result



input = 'E:/Autopilot/datasets/temp/'
output = 'E:/Autopilot/datasets/temp_rgb/'

data = list_files(input, output)

for folder in data:
    root, _, _ = folder
    os.mkdir(root)

count = 0
for folder in data:
    _, input_imgs, output_imgs = folder

    for i in range(len(input_imgs)):
        count += 1
        print(count)
        img = cv2.imread(input_imgs[i])
        img = resize_crop(img)
        cv2.imwrite(output_imgs[i], img)
