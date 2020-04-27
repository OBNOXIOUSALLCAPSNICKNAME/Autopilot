import numpy as np
import cv2
import os, sys, time, datetime, random
import argparse

from deep_tracking.deep_sort import DeepSort
from deep_tracking.util import COLORS_10, draw_bboxes

from models import *
from utils.datasets import *
from utils.utils import *

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

import torch
import torchvision.transforms as transforms
import numpy as np
import cv2

from deep_tracking.deep.model import Net
import random


class Track:
    def __init__(self):
        self.device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else '')

        self.model = Darknet(opt.config, img_size=opt.img_size)
        self.model.load_state_dict(torch.load(opt.weights, map_location=self.device)['model'])
        self.model.to(self.device).eval()

        self.classes = load_classes(opt.classes)

        self.tracker = DeepSort("deep_tracking/deep/checkpoint/ckpt.t7")

    def run(self):
        data, total_vids, total_imgs = self.list_files()

        if opt.save:
            if os.path.exists(opt.output):
                shutil.rmtree(opt.output)
            os.mkdir(opt.output)

            for folder in data:
                root, _, _, _, _ = folder
                os.mkdir(root)

        os.mkdir('{}objects/'.format(opt.output))
        os.mkdir('{}objects/{}/'.format(opt.output, 0))
        os.mkdir('{}objects/{}/'.format(opt.output, 1))
        os.mkdir('{}objects/{}/'.format(opt.output, 2))
        os.mkdir('{}objects/{}/'.format(opt.output, 3))

        for folder in data:
            _, input_imgs, input_vids, output_imgs, output_vids = folder

        for i in range(len(input_vids)):
            input = cv2.VideoCapture(input_vids[i])
            success, _ = input.read()

            if opt.save:
                fps = input.get(cv2.CAP_PROP_FPS)
                if opt.disp_dim != [0,0]:
                    output_dim = opt.disp_dim
                    w = int(opt.disp_dim[0])
                    h = int(opt.disp_dim[1])
                else:
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                output = cv2.VideoWriter(output_vids[i], cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

            while success:
                _, frame = input.read()
                if frame is not None:
                    result = frame.copy()

                    prep_boxes, prep_classes, prep_scores, _ = self.detect_image(frame)
                    self.save_objects(result, prep_boxes, prep_classes)

                    if prep_boxes is not None:
                        outputs = self.tracker.update(prep_boxes, prep_scores, frame)

                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -1]
                        draw_bboxes(result, bbox_xyxy, identities)

                    output.write(result)

                    cv2.imshow('vid', result)
                    if cv2.waitKey(1) == ord('q'):
                        input.release()
                        if opt.save:
                            output.release()
                            self.tracker.release()
                        break
                    if opt.save:
                        output.write(result)
                else:
                    break
            input.release()
            if opt.save:
                output.release()
                self.tracker.release()


    def IoU(self, first, second):
        ymin_1, xmin_1, ymax_1, xmax_1 = first
        ymin_2, xmin_2, ymax_2, xmax_2 = second

        overlap_x1 = max(xmin_1, xmin_2)
        overlap_y1 = max(ymin_1, ymin_2)
        overlap_x2 = min(xmax_1, xmax_2)
        overlap_y2 = min(ymax_1, ymax_2)

        overlap_width = (overlap_x2 - overlap_x1)
        overlap_height = (overlap_y2 - overlap_y1)

        if (overlap_width < 0) or (overlap_height < 0):
            return 0.0

        first_area = (xmax_1 - xmin_1) * (ymax_1 - ymin_1)
        second_area = (xmax_2 - xmin_2) * (ymax_2 - ymin_2)

        overlap_area = overlap_width * overlap_height
        union_area = first_area + second_area - overlap_area

        iou = overlap_area / union_area + 0.0000001
        return iou

    def detect_image(self, img):
        im0 = img.copy()

        img = letterbox(im0)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        img = torch.from_numpy(img).to(self.device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.model(img)[0]
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=[i for i in range(len(opt.classes))])

        prep_boxes = []
        prep_classes = []
        prep_scores = []

        for i, det in enumerate(pred):
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in det:
                    prep_boxes.append(xyxy)
                    prep_classes.append(int(cls))
                    prep_scores.append(float(conf))

        length = len(prep_classes)
        i = 0
        j = 0
        del_index = 0

        while i < length:
            j = 0
            while j < length:
                if i != j and self.IoU(prep_boxes[i], prep_boxes[j]) > opt.overlap_thres:

                    length -= 1
                    if i == length:
                        i -= 1

                    del_index = j if prep_scores[i] > prep_scores[j] else i

                    del prep_boxes[del_index]
                    del prep_classes[del_index]
                    del prep_scores[del_index]

                    i = 0
                    j = -1
                j += 1
            i += 1

        for i in range(len(prep_boxes)):
            x = int(prep_boxes[i][0])
            y = int(prep_boxes[i][1])
            w = int(prep_boxes[i][2]) - int(prep_boxes[i][0])
            h = int(prep_boxes[i][3]) - int(prep_boxes[i][1])

            prep_boxes[i] = [x, y, w, h]

        # for i, det in enumerate(pred):
        #     if det is not None and len(det):
        #         det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
        #
        #         for *xyxy, conf, cls in det:
        #             x = int(xyxy[0])
        #             y = int(xyxy[1])
        #             w = int(xyxy[2]) - int(xyxy[0])
        #             h = int(xyxy[3]) - int(xyxy[1])
        #
        #             prep_boxes.append([x, y, w, h])
        #             prep_classes.append(int(cls))
        #             prep_scores.append(float(conf))

        return prep_boxes, prep_classes, prep_scores, im0


    def list_files(self):
        out = []
        total_vids = 0
        total_imgs = 0
        for root, dirs, files in os.walk(opt.source):
            root = root.replace('\\', '/')
            if not root.endswith('/'):
                root += '/'

            input_img = glob.glob(root+'*g')
            input_vid = glob.glob(root+'*.mp4')

            input_img = [path.replace('\\', '/') for path in input_img]
            input_vid = [path.replace('\\', '/') for path in input_vid]

            output_img = [path.replace(opt.source, opt.output) for path in input_img]
            output_vid = [path.replace(opt.source, opt.output) for path in input_vid]

            root = root.replace(opt.source, opt.output)

            if root != opt.output:
                out.append([root, input_img, input_vid, output_img, output_vid])
                total_vids += len(input_vid)
                total_imgs += len(input_img)


        if total_vids == 0 and total_imgs == 0:
            return None, 0, 0
        else:
            return out, total_vids, total_imgs

    def save_objects(self, img, boxes, classes):
        for i in range(len(boxes)):
            x, y, w, h = boxes[i]
            name = str(classes[i])
            obj = img[y:y+h, x:x+w]
            fcount = len(glob.glob('{}objects/{}/'.format(opt.output, name)+'*g'))
            cv2.imwrite('{}objects/{}/{}_{}.jpg'.format(opt.output, name, fcount, i), obj)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=int, default=416, help='size of processed image. [0,0] for original')
    parser.add_argument('--conf_thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--overlap-thres', type=float, default=0.3, help='IOU threshold for NMS')

    parser.add_argument('--disp_dim', nargs='+', type=int, default=[1920, 1080], help='size of image to display. [0,0] for original')
    parser.add_argument('--save', action='store_false', help='save displayed results')
    parser.add_argument('--show', action='store_false', help='display results')
    parser.add_argument('--source', type=str, default='E:/Car-counting/input/', help='source folder')
    parser.add_argument('--output', type=str, default='E:/Car-counting/output/', help='output folder')

    parser.add_argument('--config', type=str, default='E:/Car-counting/datasets/cars 1/yolov3.cfg')
    parser.add_argument('--classes', type=str, default='E:/Car-counting/datasets/cars 1/classes.names')
    parser.add_argument('--weights', type=str, default='E:/Car-counting/weights/best.pt')

    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        m = Track()
        m.run()
