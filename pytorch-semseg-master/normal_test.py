import os, cv2
import torch
import argparse
import numpy as np
import scipy.misc as misc


from ptsemseg.models import get_model
from ptsemseg.loader import get_loader
from ptsemseg.utils import convert_state_dict
from loaders import DataLoader

try:
    import pydensecrf.densecrf as dcrf
except:
    pass

def crop_img(img):
    orig_h, orig_w = img.shape[:2]
    ar = 271. / 846.
    y = int(orig_w * ar)

    crop = img[orig_h-y:orig_h, 0:orig_w]
    crop = cv2.resize(crop, (846, 271))

    return crop

def preproc_img(img, mean):
    img = img[:, :, ::-1]
    img = img.astype(np.float64)
    img -= mean
    img = img.astype(float) / 255.0

    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()

    return img


def decode_segmap(temp, plot=False):
    a1 = [0, 0, 0]
    a2 = [0, 0, 128]
    a3 = [0, 0, 255]
    a4 = [0, 128, 128]
    a5 = [0, 128, 255]
    a6 = [0, 220, 220]
    a7 = [32, 11, 119]
    a8 = [35, 142, 107]
    a9 = [60, 0, 0]
    a10 = [60, 20, 220]
    a11 = [100, 100 ,150]
    a12 = [128, 64 ,128]
    a13 = [142, 0, 0]
    a14 = [153, 153, 190]
    a15 = [160, 78, 128]
    a16 = [180, 130, 70]
    a17 = [180, 165, 180]
    a18 = [190, 132, 178]
    a19 = [204, 0, 102]
    a20 = [229, 255, 201]
    a21 = [255, 255, 0]
    a22 = [255, 255, 255]

    label_colours = np.array(
        [
            a1,
            a2,
            a3,
            a4,
            a5,
            a6,
            a7,
            a8,
            a9,
            a10,
            a11,
            a12,
            a13,
            a14,
            a15,
            a16,
            a17,
            a18,
            a19,
            a20,
            a21,
            a22
        ]
    )
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, 22):
        r[temp == l] = label_colours[l, 0]
        g[temp == l] = label_colours[l, 1]
        b[temp == l] = label_colours[l, 2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    rgb = np.array(rgb, dtype=np.uint8)
    return rgb

def overlay_mask(image, overlay, ignore_color=[0,0,0]):
    ignore_color = np.asarray(ignore_color)
    mask = (overlay==ignore_color).all(-1,keepdims=True)
    out = np.where(mask,image,(image * 0.25 + overlay * 0.75).astype(image.dtype))
    return out


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean = np.array([104.00699, 116.66877, 122.67892])

    dataloader = DataLoader('E:/Autopilot/input/videos', 'E:/Autopilot/output/lol')
    model_path = "E:/Autopilot/pytorch-semseg-master/runs/52280/fcn8s_camvid_best_model.pkl"

    model_file_name = os.path.split(model_path)[1]
    model_name = model_file_name[: model_file_name.find("_")]

    model_dict = {"arch": model_name}
    model = get_model(model_dict, 22, version='camvid')
    state = convert_state_dict(torch.load(model_path)["model_state"])
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    for img in dataloader:
        crop = crop_img(img)
        img = preproc_img(crop, mean)
        img = img.to(device)
        outputs = model(img)

        pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
        decoded = decode_segmap(pred)

        res = overlay_mask(crop, decoded)

        dataloader.save_results(res)
        cv2.imshow('123', res)
        if cv2.waitKey(1) == ord('q'):
            break

run()
