import os
import glob
import cv2
import sys
import argparse
import random
import shutil

import numpy as np
import xml.etree.cElementTree as ET

import label_utils
import image_utils


def run():
    data = walk_tree()
    if None in data:
        pass
    else:
        if os.path.exists(opt.root+'images'): shutil.rmtree(opt.root+'images')
        if os.path.exists(opt.root+'labels'): shutil.rmtree(opt.root+'labels')
        if os.path.exists(opt.root+'labels_xml'): shutil.rmtree(opt.root+'labels_xml')

        os.mkdir(opt.root+'images')
        os.mkdir(opt.root+'labels')
        os.mkdir(opt.root+'labels_xml')

        cls_count = 1 if opt.single_cls else len(data[0])
        create_cfg_file(cls_count)
        create_data_file(cls_count)

        train_samples = []
        test_samples = []

        rnd_indices = random.sample(range(0, len(data[1][1])), len(data[1][1]))
        range_bound = int(len(data[1][1]) * opt.test_p)
        for x in range(len(rnd_indices)):
            i = rnd_indices[x]
            img_path = data[1][1][i]
            name = img_path.replace('\\', '/').split('/')[-1].replace('.jpg', '')
            indices = [x for x, s in enumerate(data[1][0]) if name in s]
            xml_path = data[1][0][indices[0]] if len(indices) else None
            if x <= range_bound:
                type = 'test'
                test_samples.append(img_path.replace('.jpg', '.JPG'))
            else:
                type = 'train'
                train_samples.append(img_path.replace('.jpg', '.JPG'))
            process_sample(img_path, xml_path, i, data[0], type)

        if None not in data[2]:
            for i in range(len(data[2][1])):
                img_path = data[2][1][i]
                name = img_path.replace('\\', '/').split('/')[-1].replace('.jpg', '')
                indices = [x for x, s in enumerate(data[2][0]) if name in s]
                xml_path = data[1][0][indices[0]] if len(indices) else None
                process_sample(img_path, xml_path, i, data[0], 'test')
                test_samples.append(img_path.replace('.jpg', '.JPG'))

        create_map(train_samples, test_samples)

def process_sample(img_path, xml_path, index, classes, type='default'):
    img = cv2.imread(img_path)

    if xml_path is not None:
        xml = ET.parse(xml_path).getroot()
        xml_labels = label_utils.parse_xml(xml)
    else:
        xml_labels = []

    if opt.resolution != [0,0]:
        (h, w) = img.shape[:2]
        old_dims = [w, h]
        scale, offset = image_utils.get_scale_val(old_dims, opt.resolution, opt.keep_ar)
        img = image_utils.resize_image(img, opt.resolution, scale, offset)
        xml_labels = image_utils.scale_xml_labels(xml_labels, scale, offset)

    (h, w) = img.shape[:2]
    new_dims = [w, h]
    txt_labels = image_utils.xyxy_to_xywh(xml_labels, new_dims)

    path_txt = "{}labels/{}_{}.txt".format(opt.root, index, type)
    path_xml = "{}labels_xml/{}_{}.xml".format(opt.root, index, type)
    path_img = "{}images/{}_{}.jpg".format(opt.root, index, type)

    label_utils.save_as_txt(txt_labels, path_txt, classes, opt.single_cls)
    label_utils.save_as_xml(xml_labels, path_xml, old_dims, opt.single_cls)
    cv2.imwrite(path_img, img)


def walk_tree():
    if os.path.exists(opt.root) and os.path.isdir(opt.root):
        output = [None, None, [None, None]]

        orig_root = opt.root+'orig data'
        labelmap = opt.root+'classes.names'

        test_root = opt.root+'orig data/test/'
        test_xmls = opt.root+'orig data/test/labels/'
        test_imgs = opt.root+'orig data/test/images/'

        train_root = opt.root+'orig data/train/'
        train_xmls = opt.root+'orig data/train/labels/'
        train_imgs = opt.root+'orig data/train/images/'


        if os.path.exists(orig_root) and os.path.isdir(orig_root):

            # grab classes names
            if os.path.exists(labelmap) and os.path.isfile(labelmap):
                with open(labelmap, 'r') as f:
                    names = f.read().split('\n')
                output[0] = list(filter(None, names))
            else: print('cant find ', labelmap)

            # grab train data
            if os.path.exists(train_root) and os.path.isdir(train_root):
                train = [None, None]
                if os.path.exists(train_xmls) and os.path.isdir(train_xmls):
                    xmls = glob.glob(train_xmls+'*.xml')
                    if len(xmls): train[0] = xmls
                    else: print('no labels found in  ', train_xmls)
                else: print('cant find ', train_xmls)

                if os.path.exists(train_imgs) and os.path.isdir(train_imgs):
                    imgs = glob.glob(train_imgs+'*g')
                    if len(imgs): train[1] = imgs
                    else: print('no images found in  ', train_imgs)
                else: print('cant find ', train_imgs)
                if None not in train: output[1] = train
            else: print('cant find ', train_root)

            # grab test data if it exists
            if os.path.exists(test_root) and os.path.isdir(test_root):
                test = [None, None]
                if os.path.exists(test_xmls) and os.path.isdir(test_xmls):
                    xmls = glob.glob(test_xmls+'*.xml')
                    if len(xmls): test[0] = xmls

                if os.path.exists(test_imgs) and os.path.isdir(test_imgs):
                    imgs = glob.glob(test_imgs+'*g')
                    if len(imgs): test[1] = imgs
                output[2] = test
        else: print('cant find ', orig_root)
    else: print('missing directory:\n    {}'.format(opt.root))
    return output


def create_map(train_samples, test_samples):
    with open(opt.root+'train.txt', 'w') as f:
        for row in train_samples:
            f.write("%s\n" % row.replace('\\', '/'))
    with open(opt.root+'test.txt', 'w') as f:
        for row in test_samples:
            f.write("%s\n" % row.replace('\\', '/'))

def create_data_file(classes):
    if os.path.exists(opt.root+'yolov3.data'):
        os.remove(opt.root+'yolov3.data')
    with open(opt.root+'yolov3.data', 'a') as f:
        f.write('classes= {}\n'.format(classes))
        f.write('train={}train.txt\n'.format(opt.root))
        f.write('train={}val.txt\n'.format(opt.root))
        f.write('train={}classes.names\n'.format(opt.root))
        f.write('backup=backup/\n')
        f.write('eval=coco')

def create_cfg_file(classes):
    if os.path.exists(opt.root+'yolov3.cfg'):
        os.remove(opt.root+'yolov3.cfg')
    with open('./yolov3.cfg', 'r') as f:
        data = f.readlines()

    data[609] = 'classes={}\n'.format(classes)
    data[695] = 'classes={}\n'.format(classes)
    data[782] = 'classes={}\n'.format(classes)

    data[602] = 'filters={}\n'.format((5 + classes) * 3)
    data[688] = 'filters={}\n'.format((5 + classes) * 3)
    data[775] = 'filters={}\n'.format((5 + classes) * 3)

    with open(opt.root+'yolov3.cfg', 'w') as file:
        file.writelines(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_p', type=float, default=0.2, help='precent of images to test set')
    parser.add_argument('--single_cls', action='store_false', help='merge all classes into "_object_"')
    parser.add_argument('--resolution', nargs='+', type=int, default=[640, 360], help='resolution of images. [0,0] for original')
    parser.add_argument('--keep_ar', action='store_false', help='keep aspect ratio when resize images')
    parser.add_argument('--root', type=str, default='E:/Car-counting/datasets/cars 1/', help='path to dataset')
    opt = parser.parse_args()
    print(opt)

    run()
