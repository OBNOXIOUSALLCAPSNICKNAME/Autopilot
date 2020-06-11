from loaders import DataLoader
import cv2
import numpy as np
import scipy.misc as m

dataloader = DataLoader(
'C:/zaloopa/data/Labels',
'C:/zaloopa/data/roflanebalo',
False, 0)


# in B G R
good_colors = [
[0, 0, 0],

[180, 130, 70],
[60, 20, 220],
[0, 220, 220],

[128, 0, 128],
[0, 0, 255],
[30, 170, 250],

[60, 0, 0],
[100, 60, 0],

[142, 0, 0],
[32, 11, 119],

[232, 35, 244],
[160, 0, 0],
[153, 153, 153]
]

def rgb_to_cls(img):
    h, w = img.shape[:2]
    result = np.full((h, w), 0, dtype=np.uint8)
    for x in range(len(good_colors)):
        good_color = np.asarray(good_colors[x])
        result[np.where((img == good_color).all(axis = 2))] = x
    return result

for img, type, name, _  in dataloader:
    print(dataloader.print_status())
    h, w = img.shape[:2]
    img = img[int(h * 0.65):h, 0:w]
    img = cv2.resize(img, None, fx=0.40, fy=0.40, interpolation=cv2.INTER_NEAREST)
    h, w = img.shape[:2]

    mask = np.zeros((h, w), dtype=bool)
    for i in range(len(good_colors)):
        mask = np.logical_or(mask, (img == good_colors[i]).all(axis=2))
    img[~mask] = [0,0,0]

    img = rgb_to_cls(img)

    dataloader.save_results(img, img_format='png')
