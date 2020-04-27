import os
import collections
import torch
import numpy as np
import scipy.misc as m
import cv2
import matplotlib.pyplot as plt

from torch.utils import data
from ptsemseg.augmentations import Compose, RandomHorizontallyFlip, RandomRotate


class customLoader(data.Dataset):
    def __init__(
        self,
        root,
        split="train",
        is_transform=False,
        img_size=None,
        augmentations=None,
        img_norm=True,
        test_mode=False,
    ):
        self.root = root
        self.split = split
        self.img_size = [600, 900]
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.test_mode = test_mode
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.n_classes = 15
        self.files = collections.defaultdict(list)

        if not self.test_mode:
            for split in ["train", "test", "val"]:
                file_list = os.listdir(root + "/" + split)
                self.files[split] = file_list

        self.old_colors = [
        [0, 220, 220],
        [142, 0, 0],
        [153, 153, 190],
        [180, 130, 70],
        [255, 255, 255],
        [128, 64, 128],
        [204, 0, 102],
        [0, 128, 128],
        [100, 100, 150],
        [230, 0, 0],
        [255, 255, 0],
        [0, 128, 255],
        [190, 132, 178],
        [160, 78, 128],

        [0, 0, 0],
        [255, 255, 255],
        [14, 88, 161]]

        self.new_colors = [
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

        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]]

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_name = self.files[self.split][index]
        img_path = self.root + "/" + self.split + "/" + img_name
        lbl_path = self.root + "/" + self.split + "annot/" + img_name

        img = m.imread(img_path)
        img = cv2.resize(img, (450,300))
        img = np.array(img, dtype=np.uint8)

        lbl = m.imread(lbl_path)
        lbl = cv2.resize(lbl, (450,300), interpolation=cv2.INTER_NEAREST)
        lbl = self.rgb_to_cls(lbl)
        lbl = np.array(lbl, dtype=np.int8)

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl

    def rgb_to_cls(self, img):
        h, w = img.shape[:2]
        result = np.full((h, w), 0, dtype=np.uint8)

        for x in range(len(self.old_colors)):
            old_color = np.asarray(self.old_colors[x])
            new_color = np.asarray(self.new_colors[x])
            result[np.where((img == old_color).all(axis = 2))] = new_color[0]

        return result

    def transform(self, img, lbl):
        img = m.imresize(img, (self.img_size[0], self.img_size[1]))  # uint8 with RGB mode
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean
        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def decode_segmap(self, temp, plot=False):
        c1 = [0, 220, 220]
        c2 = [142, 0, 0]
        c3 = [153, 153, 190]
        c4 = [180, 130, 70]
        c5 = [255, 255, 255]
        c6 = [128, 64, 128]
        c7 = [204, 0, 102]
        c8 = [0, 128, 128]
        c9 = [100, 100, 150]
        c10 = [230, 0, 0]
        c11 = [255, 255, 0]
        c12 = [0, 128, 255]
        c13 = [190, 132, 178]
        c14 = [160, 78, 128]

        Bground = [0, 0, 0]

        label_colours = np.array(
            [
                c1,
                c2,
                c3,
                c4,
                c5,
                c6,
                c7,
                c8,
                c9,
                c10,
                c11,
                c12,
                c13,
                c14,

                Bground,
            ]
        )
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = label_colours[l, 0]
            g[temp == l] = label_colours[l, 1]
            b[temp == l] = label_colours[l, 2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb


if __name__ == "__main__":
    local_path = "/home/meetshah1995/datasets/segnet/CamVid"
    augmentations = Compose([RandomRotate(10), RandomHorizontallyFlip()])

    dst = camvidLoader(local_path, is_transform=True, augmentations=augmentations)
    bs = 4
    trainloader = data.DataLoader(dst, batch_size=bs)
    for i, data_samples in enumerate(trainloader):
        imgs, labels = data_samples
        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0, 2, 3, 1])
        f, axarr = plt.subplots(bs, 2)
        for j in range(bs):
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
        plt.show()
        a = input()
        if a == "ex":
            break
        else:
            plt.close()
