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

def get_scale_val(image, new_dims, keep_ar):
    (h, w) = image.shape[:2]
    old_dims = [w, h]
    if keep_ar == False:
        scale_x = float(new_dims[0]) / float(old_dims[0])
        scale_y = float(new_dims[1]) / float(old_dims[1])
        return [scale_x, scale_y], [0,0]
    else:
        new_ar = float(new_dims[1])/float(new_dims[0])
        old_ar = float(old_dims[1])/float(old_dims[0])
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

def resize_image(image, resolution, scale, offset, inter):
    result = np.full((resolution[1], resolution[0], 3), [14, 88, 161], dtype=np.uint8)
    image = cv2.resize(image, None, fx=scale[0], fy=scale[1], interpolation=inter)
    (h, w) = image.shape[:2]
    result[offset[1]:offset[1]+h, offset[0]:offset[0]+w] = image
    return result


old_colors = [
[0, 0, 0],
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
[22, 22, 22],

[0, 0, 0],
[0, 0, 0]
]

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

def foo1():
    lbls_path = 'E:/Autopilot/datasets/try/Labels'
    imgs_path = 'E:/Autopilot/datasets/try/Images'
    output = 'E:/Autopilot/datasets/ROAD_LINES_846x271'

    data_lbls = list_files(lbls_path, lbls_path)
    data_imgs = list_files(imgs_path, imgs_path)

    imgs = []
    lbls = []

    for folder in data_imgs:
        imgs += folder[1]

    for folder in data_lbls:
        lbls += folder[1]

    range_train = len(lbls) * 0.7
    range_test = len(lbls) * 0.9

    def add_colors(new, total):
        f = list(new) + total
        return list(np.unique(np.asarray(f), axis=0))

    total_colors = []

    for i in range(len(lbls)):
        #print(i)

        img = cv2.imread(imgs[i])
        #lbl = cv2.imread(lbls[i])

        #lbl = rgb_to_cls(lbl)
        #lbl = resize_crop(lbl)
        #img = resize_crop(img)

        if i < range_train:
            cv2.imwrite('{}/train/{}.png'.format(output, i), img)
            #cv2.imwrite('{}/trainannot/{}.png'.format(output, i), lbl)
        elif i > range_train and i < range_test:
            cv2.imwrite('{}/test/{}.png'.format(output, i), img)
            #cv2.imwrite('{}/testannot/{}.png'.format(output, i), lbl)
        else:
            cv2.imwrite('{}/val/{}.png'.format(output, i), img)
            #cv2.imwrite('{}/valannot/{}.png'.format(output, i), lbl)

def foo2():
    train = 'E:/Autopilot/datasets/ROAD_LINES_846x271/trainannot'
    test = 'E:/Autopilot/datasets/ROAD_LINES_846x271/testannot'
    val = 'E:/Autopilot/datasets/ROAD_LINES_846x271/valannot'

    a1 = list_files(train, train)
    a2 = list_files(test, test)
    a3 = list_files(val, val)

    count = 0

    for folder in a1:
        _, input_imgs, output_imgs = folder

        for i in range(len(input_imgs)):
            count += 1
            print(count)
            lbl = cv2.imread(input_imgs[i])
            lbl = rgb_to_cls(lbl)
            cv2.imwrite(output_imgs[i], lbl)

    for folder in a2:
        _, input_imgs, output_imgs = folder

        for i in range(len(input_imgs)):
            count += 1
            print(count)
            lbl = cv2.imread(input_imgs[i])
            lbl = rgb_to_cls(lbl)
            cv2.imwrite(output_imgs[i], lbl)

    for folder in a3:
        _, input_imgs, output_imgs = folder

        for i in range(len(input_imgs)):
            count += 1
            print(count)
            lbl = cv2.imread(input_imgs[i])
            lbl = rgb_to_cls(lbl)
            cv2.imwrite(output_imgs[i], lbl)

foo2()
