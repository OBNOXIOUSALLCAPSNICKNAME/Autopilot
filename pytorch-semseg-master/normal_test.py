import os, cv2
import torch
import argparse
import numpy as np
import scipy.misc as misc


from ptsemseg.models import get_model
from ptsemseg.loader import get_loader
from ptsemseg.utils import convert_state_dict
from loaders import DataLoader

from collections import defaultdict
from collections import Counter
from math import atan2, degrees

from shapely.geometry import LineString
from scipy.spatial import distance

try:
    import pydensecrf.densecrf as dcrf
except:
    pass

def crop_img(img):
    orig_h, orig_w = img.shape[:2]
    ar = 380. / 1355.
    y = int(orig_w * ar)

    crop = img[orig_h-y:orig_h, 0:orig_w]
    crop = cv2.resize(crop, (1084, 304))

    #return img[495:495+304, 583:583+1085]
    return img[770:770+304, 520:520+1085]

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
    a2 = [180, 70, 30]

    label_colours = np.array(
        [
            a1,
            a2
        ]
    )
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(1, 30):
        r[temp == l] = label_colours[1, 0]
        g[temp == l] = label_colours[1, 1]
        b[temp == l] = label_colours[1, 2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    rgb = np.array(rgb, dtype=np.uint8)
    return rgb

def overlay_mask(image, overlay, ignore_color=[0,0,0]):
    ignore_color = np.asarray(ignore_color)
    mask = (overlay==ignore_color).all(-1,keepdims=True)
    out = np.where(mask,image,(image * 0.35 + overlay * 0.65).astype(image.dtype))
    return out


def SetLines(image, lines, thickness=2, color=(255,0,0), offset_x = 0, offset_y = 0):
    width = np.size(image, 1)
    height = np.size(image, 0)

    for x in range(len(lines)):
        x1,y1,x2,y2, _ = np.array(lines[x]).astype('int')
        x1,y1,x2,y2 = LengthenLine(x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y, width, height)
        cv2.line(image,(x1, y1),(x2, y2),color, thickness)

def GroupLines(input_lines, unique = True):

    group = []
    output = []
    lines = input_lines.copy()
    length = len(lines)
    i = 0
    while i < length:
        j = 0
        i_x1,i_y1,i_x2,i_y2,angle_i = lines[i]
        group.append(lines[i].copy())
        while j < length:
            if i != j:
                j_x1,j_y1,j_x2,j_y2,angle_j = lines[j]

                distance = LineDistance((i_x1,i_y1), (i_x2,i_y2), (j_x1,j_y1))
                dif_angle = abs(max(angle_i, angle_j) - min(angle_i, angle_j))

                if (dif_angle <= 3 and distance < 40) or (dif_angle < 15 and dif_angle > 2 and distance < 50):
                    group.append(lines[j].copy())
                    del lines[j]

                    length -= 1
                    if i == length and i > j: i -= 1
                    j -= 1
            j += 1

        average_x1 = sum([element[0] for element in group]) / len(group)
        average_y1 = sum([element[1] for element in group]) / len(group)
        average_x2 = sum([element[2] for element in group]) / len(group)
        average_y2 = sum([element[3] for element in group]) / len(group)
        angle_avg = np.degrees(np.arctan2(average_y1 - average_y2, average_x1 - average_x2))
        x1 = average_x1
        x2 = average_x2
        y1 = average_y1
        y2 = average_y2

        if unique == True:
            output.append([x1,y1,x2,y2,angle_avg])
        else:
            for x in range(len(group)):
                output.append([x1,y1,x2,y2,angle_avg])
        group.clear()
        i += 1
    return output

def LengthenLine(input_x1, input_y1, input_x2, input_y2, width, height):
    p1 = (input_x1, input_y1)
    p2 = (input_x2, input_y2)

    theta = np.arctan2(p1[1]-p2[1], p1[0]-p2[0])
    endpt_x = int(p2[0] - 4000*np.cos(theta))
    endpt_y = int(p2[1] - 4000*np.sin(theta))

    startpt_x = int(p1[0] + 4000*np.cos(theta))
    startpt_y = int(p1[1] + 4000*np.sin(theta))

    line = LineString([(endpt_x, endpt_y), (startpt_x, startpt_y)])

    top_W = LineString([(0, 110), (width, 110)])
    bottom_W = LineString([(0, height), (width, height)])
    left_h = LineString([(0, 110), (0, height)])
    right_H = LineString([(width, 0), (width, height)])

    p1 = line.intersection(top_W)
    p2 = line.intersection(bottom_W)
    p3 = line.intersection(left_h)
    p4 = line.intersection(right_H)

    res = []
    if hasattr(p1, 'x'):
        res.append(p1)
    if hasattr(p2, 'x'):
        res.append(p2)
    if hasattr(p3, 'x'):
        res.append(p3)
    if hasattr(p4, 'x'):
        res.append(p4)

    try:
        startpt_x = res[0].x
        startpt_y = res[0].y
        endpt_x = res[1].x
        endpt_y = res[1].y
        return int(startpt_x), int(startpt_y), int(endpt_x), int(endpt_y)
    except:
        return int(input_x1), int(input_y1), int(input_x2), int(input_y2)

def LineDistance(line1_p1,line1_p2, line2_p):
    p1 = np.asarray((line1_p1[0], line1_p1[1]))
    p2 = np.asarray((line1_p2[0], line1_p2[1]))
    p3 = np.asarray((line2_p[0], line2_p[1]))
    distance = int(np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1))
    return distance


def update_buffer(buffer, grouped_lines):
    if not len(buffer):
        for line in grouped_lines:
            buffer.append([line, [line], [0], [1]])
        return buffer
    else:
        for x in range(len(buffer)):
            if len(buffer[x][1]) > 2:
                buffer[x][1].pop(0)

        updated_lines = []
        for i in range(len(grouped_lines)):
            distances = []
            angles = []
            i_x1, i_y1, i_x2, i_y2, angle_i = grouped_lines[i]
            for j in range(len(buffer)):
                avg, lines, *_ = buffer[j]
                j_x1, j_y1, j_x2, j_y2, angle_j = avg
                distances.append(LineDistance((j_x1, j_y1), (j_x2, j_y2), (i_x1, i_y1)))
                angles.append(abs(max(angle_i, angle_j) - min(angle_i, angle_j)))

            index = distances.index(min(distances))
            dist = distances[index]
            angle = angles[index]

            if (angle <= 5 and dist < 40) or (angle < 15 and angle > 5 and dist < 55):
                buffer[index][1].append(grouped_lines[i])
                avg_x1 = sum([e[0] for e in buffer[index][1]]) / len(buffer[index][1])
                avg_y1 = sum([e[1] for e in buffer[index][1]]) / len(buffer[index][1])
                avg_x2 = sum([e[2] for e in buffer[index][1]]) / len(buffer[index][1])
                avg_y2 = sum([e[3] for e in buffer[index][1]]) / len(buffer[index][1])

                angle_avg = np.degrees(np.arctan2(avg_y1 - avg_y2, avg_x1 - avg_x2))
                buffer[index][0] = [avg_x1, avg_y1, avg_x2, avg_y2, angle_avg]
                updated_lines.append(index)
            else:
                buffer.append([grouped_lines[i], [grouped_lines[i]], [0], [1]])
        updated_lines = np.unique(np.array(updated_lines))

        new_buffer = []
        for i in range(len(buffer)):
            if i in updated_lines:
                buffer[i][3].append(1)
            else:
                buffer[i][3].append(0)
            if len(buffer[i][3]) > 3:
                buffer[i][3].pop(0)
            if np.all(np.array(buffer[i][3])==0):
                pass
            else:
                new_buffer.append(buffer[i])
        return new_buffer

def update_center(lines, img, x_l=400, y_l=200):
    if len(lines) < 2:
        return 0, img
    lines = sorted(lines, key = lambda x: x[0])
    distances = []
    for x in range(len(lines)):
        x1, y1, x2, y2, angle = lines[x]
        distances.append(LineDistance((x1,y1), (x2,y2), (x_l, y_l)))
    indices = np.argsort(np.array(distances))

    h, w = img.shape[:2]
    x1, y1, x2, y2, _ = lines[indices[0]]
    x1, y1, x2, y2 = LengthenLine(x1, y1, x2, y2, w, h)
    if y1 < y2:
        l1 = [x1, y1, x2, y2]
    else:
        l1 = [x2, y2, x1, y1]
    x1, y1, x2, y2, _ = lines[indices[1]]
    x1, y1, x2, y2 = LengthenLine(x1, y1, x2, y2, w, h)
    if y1 < y2:
        l2 = [x1, y1, x2, y2]
    else:
        l2 = [x2, y2, x1, y1]
    roi_corners = np.array([[(l1[0], l1[1]), (l1[2], l1[3]), (l2[2], l2[3]), (l2[0], l2[1])]], dtype=np.int32)
    mask = np.zeros(img.shape, dtype=np.uint8)
    cv2.fillPoly(mask, roi_corners, (235,215,150))
    img = overlay_mask(img, mask)
    if lines[indices[0]][0] < lines[indices[1]][0]:
        return distances[indices[1]] - distances[indices[0]], img
    else:
        return distances[indices[0]] - distances[indices[1]], img

def classificate_lines(buffer, img):
    result = img.copy()
    height, width, c = img.shape
    for i in range(len(buffer)):
        img0 = np.zeros((height, width, c), np.uint8)
        x1, y1, x2, y2, angle = buffer[i][0]
        x1, y1, x2, y2 = LengthenLine(x1, y1, x2, y2, width, height)
        cv2.line(img0, (int(x1), int(y1)), (int(x2), int(y2)), (255,255,255), 12)
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        temp_img = cv2.bitwise_and(img, img, mask=img0)
        col = int(abs(y1 - y2) / 10)
        l = LineString([(int(x1), int(y1)), (int(x2), int(y2))])
        y = int(max(y1, y2))
        coords = []
        for j in range(col):
            y -= 10
            x_, _ = temp_img[y,:].nonzero()
            if len(x_):
                xmin = np.min(x_)
                xmax = np.max(x_)
                l2 = LineString([(0, y), (width, y)])
                p = l.intersection(l2)

                cv2.line(result, (int(p.x - 25), y),(int(p.x + 25), y), (100,200,80), 2)
                cv2.line(result, (int(xmin), y),(int(xmax), y), (180,150,255), 2)
                coords.append([int(xmin), int(xmax), y])

        brs = []
        for z in range(len(coords) - 1):
            ff = np.zeros((height, width, c), np.uint8)
            xmin1, xmax1, y1_ = coords[z]
            xmin2, xmax2, y2_ = coords[z+1]
            xmin = min(xmin1, xmin2)
            xmax = max(xmax1, xmax2)
            ff[y2_:y1_, xmin:xmax] = temp_img[y2_:y1_, xmin:xmax]
            part = temp_img[y2_:y1_, xmin:xmax]
            r, g, b = cv2.split(part)
            r = cv2.equalizeHist(r)
            g = cv2.equalizeHist(g)
            b = cv2.equalizeHist(b)
            r = r.flatten()
            g = g.flatten()
            b = b.flatten()
            r = np.sum(r[np.nonzero(r)]) / len(r[np.nonzero(r)])
            g = np.sum(g[np.nonzero(g)]) / len(g[np.nonzero(g)])
            b = np.sum(b[np.nonzero(b)]) / len(b[np.nonzero(b)])
            brightness = np.sum([r, g, b]) / 3
            brs.append(brightness)

        if len(brs) == 0:
            buffer[i][2].append(2)
            if len(buffer[i][2]) > 5:
                buffer[i][2].pop(0)
        else:
            if abs(min(brs) - max(brs)) > 15:
                buffer[i][2].append(1)
                if len(buffer[i][2]) > 5:
                    buffer[i][2].pop(0)
            else:
                buffer[i][2].append(0)
                if len(buffer[i][2]) > 5:
                    buffer[i][2].pop(0)

    for line_obj in buffer:
        x1, y1, x2, y2, angle = line_obj[0]
        x1, y1, x2, y2 = LengthenLine(x1, y1, x2, y2, width, height)
        classes = line_obj[2]
        cls0 = np.count_nonzero(np.array(classes)==0)
        cls1 = np.count_nonzero(np.array(classes)==1)
        cls2 = np.count_nonzero(np.array(classes)==2)
        classes = np.array([cls0, cls1, cls2])
        cls = np.argmax(classes)
        if cls == 1:
            cv2.line(result, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        elif cls == 0:
            cv2.line(result, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
        else:
            cv2.line(result, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)

    return result, buffer



def foo(crop, orig, img, buffer, x_l=400, y_l=200):
    h, w, c = img.shape
    img0 = np.zeros((h, w, c), np.uint8)

    img = cv2.Canny(img, 100, 200)
    _lines = cv2.HoughLinesP(img, rho=1, theta=np.pi/180, threshold=40, minLineLength=10, maxLineGap=300)
    lines = []
    if _lines is not None:
        for x in range(0, len(_lines)):
            x1, y1, x2, y2 = _lines[x][0];
            angle = np.degrees(np.arctan2(y1 - y2, x1 - x2))
            #if abs(angle) < 165 and abs(angle) > 100:
            #if abs(angle) < 140 and abs(angle) > 100:
            lines.append([x1, y1, x2, y2, angle])

    grouped_lines = GroupLines(lines)
    buffer = update_buffer(buffer, grouped_lines)
    grouped_lines = [x[0] for x in buffer]
    crop, buffer = classificate_lines(buffer, crop)
    dist, crop = update_center(grouped_lines, crop, x_l, y_l)
    if dist < 0:
        cv2.line(crop, (x_l, y_l),(x_l + dist, y_l), [0, 0, 255], 4)
    else:
        cv2.line(crop, (x_l, y_l),(x_l + dist, y_l), [0, 255, 0], 4)
    cv2.line(crop, (x_l + dist, y_l-30),(x_l + dist, y_l+30), [126, 126, 126], 3)
    return crop, buffer

def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean = np.array([104.00699, 116.66877, 122.67892])

    dataloader = DataLoader('E:/Autopilot/input/vc', 'E:/Autopilot/output/vc')
    model_path = "E:/Autopilot/pytorch-semseg-master/runs/39060/fcn8s_camvid_best_model.pkl"

    model_file_name = os.path.split(model_path)[1]
    model_name = model_file_name[: model_file_name.find("_")]

    model_dict = {"arch": model_name}
    model = get_model(model_dict, 2, version='camvid')
    state = convert_state_dict(torch.load(model_path)["model_state"])
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    buffer = []
    for img0, _, _, _, frame in dataloader:
        if frame == 1:
            buffer = []
        # x = 520
        # y = 770
        x = 550
        y = 680
        crop = img0[y:y+304, x:x+1085]
        img = preproc_img(crop, mean)
        img = img.to(device)
        outputs = model(img)

        pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
        decoded = decode_segmap(pred)

        res = overlay_mask(crop, decoded)
        res, buffer = foo(crop, res, decoded, buffer, x_l=455, y_l=180)

        img0[y:y+304, x:x+1085] = res
        dataloader.save_results(img0)
        cv2.imshow('123', res)
        if cv2.waitKey(1) == ord('q'):
            dataloader.release()
            break

run()
