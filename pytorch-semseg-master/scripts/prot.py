import os
import cv2
import sys
import math
import shapely
import statistics
import numpy as np
import pandas as pd
import tensorflow as tf

from collections import defaultdict
from collections import Counter
from math import atan2, degrees

from utils import label_map_util
from utils import visualization_utils as vis_util

from shapely.geometry import LineString
from scipy.spatial import distance
from skimage.transform import (hough_line, hough_line_peaks, probabilistic_hough_line)

sys.path.append("..")

CWD_PATH = os.getcwd()


VIDEO_NAME = '3rd.mp4'
#VIDEO_NAME = '2nd.mp4'
#VIDEO_NAME = '1st.mp4'
NUM_CLASSES = 1
TRESHOLD_player = 0.85
TRESHOLD_ball = 0.70
PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME)

#=========== for sess_ball

MODEL_NAME_ball = 'inference_graph_ball'

PATH_TO_CKPT_ball = os.path.join(CWD_PATH,MODEL_NAME_ball,'frozen_inference_graph.pb')
PATH_TO_LABELS_ball = os.path.join(CWD_PATH,'inference_graph_ball','labelmap.pbtxt')

label_map_ball = label_map_util.load_labelmap(PATH_TO_LABELS_ball)
categories_ball = label_map_util.convert_label_map_to_categories(label_map_ball, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index_ball = label_map_util.create_category_index(categories_ball)

detection_graph_ball = tf.Graph()
with detection_graph_ball.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT_ball, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess_ball = tf.Session(graph=detection_graph_ball)

image_tensor_ball = detection_graph_ball.get_tensor_by_name('image_tensor:0')
detection_boxes_ball = detection_graph_ball.get_tensor_by_name('detection_boxes:0')
detection_scores_ball = detection_graph_ball.get_tensor_by_name('detection_scores:0')
detection_classes_ball = detection_graph_ball.get_tensor_by_name('detection_classes:0')
num_detections_ball = detection_graph_ball.get_tensor_by_name('num_detections:0')



#=========== for sess_player

MODEL_NAME_player = 'inference_graph_player'

PATH_TO_CKPT_player = os.path.join(CWD_PATH,MODEL_NAME_player,'frozen_inference_graph.pb')
PATH_TO_LABELS_player = os.path.join(CWD_PATH,'inference_graph_player','labelmap.pbtxt')

label_map_player = label_map_util.load_labelmap(PATH_TO_LABELS_player)
categories_player= label_map_util.convert_label_map_to_categories(label_map_player, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index_player = label_map_util.create_category_index(categories_player)

detection_graph_player = tf.Graph()
with detection_graph_player.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT_player, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess_player= tf.Session(graph=detection_graph_player)

image_tensor_player= detection_graph_player.get_tensor_by_name('image_tensor:0')
detection_boxes_player = detection_graph_player.get_tensor_by_name('detection_boxes:0')
detection_scores_player = detection_graph_player.get_tensor_by_name('detection_scores:0')
detection_classes_player= detection_graph_player.get_tensor_by_name('detection_classes:0')
num_detections_player = detection_graph_player.get_tensor_by_name('num_detections:0')

accum_player_dist = []
accum_ball_coord = [[0,0]]
accum_nearest_dist = []

def AddElement(list, element, max_len):
    list.insert(0, element)
    if len(list) > max_len:
        list.pop(max_len)

def CustomViusalize(boxes, frame, color, thickness):
    for bndbox in boxes:
        pt1 = (bndbox[1], bndbox[0])
        pt2 = (bndbox[3], bndbox[2])
        cv2.rectangle(frame, pt1, pt2, color, thickness)
    return frame

def PreprocData(boxes, scores, classes, treshold, frame):
    width = np.size(frame, 1)
    height = np.size(frame, 0)

    prep_boxes = []
    prep_classes = []
    prep_scores = []

    for j in range(classes.size):
        if scores[0][j] > treshold:
            prep_scores.append(scores[0][j])
            prep_classes.append(classes[0][j])
            prep_boxes.append(boxes[0][j])

    for coord in prep_boxes:
        coord[0] = coord[0] * height
        coord[1] = coord[1] * width
        coord[2] = coord[2] * height
        coord[3] = coord[3] * width

    return prep_boxes, prep_classes, prep_scores

def BoxesToPoints(boxes, position):
    points = []
    if len(boxes) == 0:
        return [accum_ball_coord[len(accum_ball_coord) - 1]]

    if position == "center":
        for bndbox in boxes:
            x = bndbox[1] + abs(bndbox[3] - bndbox[1]) / 2
            y = bndbox[0] + abs(bndbox[2] - bndbox[0]) / 2
            points.append([x, y])
        return points
    elif position == "down":
        for bndbox in boxes:
            x = bndbox[1] + abs(bndbox[3] - bndbox[1]) / 2
            y = max(bndbox[2], bndbox[0])
            points.append([x, y])
        return points

def AccumData(boxes_b, boxes_p, frame):
    distances = []
    coord_b = BoxesToPoints(boxes_b, 'center')[0]
    coord_p = BoxesToPoints(boxes_p, 'down')

    for i in range(len(boxes_p)):
        dist = math.sqrt((coord_p[i][1] - coord_b[1])**2 + (coord_p[i][0] - coord_b[0])**2)
        distances.append(int(dist))

    distances.sort()

    AddElement(accum_player_dist, distances, 10)
    AddElement(accum_nearest_dist, min(distances), 10)
    AddElement(accum_ball_coord, coord_b, 10)

def AccumAnalyze():
    if PlayerOwn():
        return "player own"
    elif Fight():
        return "fight"
    else:
        return "unknown"

def PlayerOwn():
    accum_to_arr = np.asarray(accum_nearest_dist[:])
    equals = np.where(accum_to_arr < 150, 1, 0)
    res = np.count_nonzero(equals)

    return res >= 8 and accum_player_dist[len(accum_player_dist) - 1][1] > 200

def Fight():
    accum = 0
    for list in accum_player_dist:
        if list[2] < 200:
            accum += 1
    return accum >= 7




def HueMask(image):
    width = np.size(image, 1)
    height = np.size(image, 0)

    image = cv2.GaussianBlur(image, (3, 3), 0)
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    unique, counts = np.unique(hsv[:, :, 0], return_counts=True)
    hist_s = pd.Series(counts, unique)

    # Normalize hist
    hist_s = (hist_s / width / height) * 100
    h_argmax = hist_s.argmax()
    h_argright = h_argmax + 5
    h_argleft = h_argmax - 5
    img_slice = hsv[:, :, 0]

    # Change image pixel of 'H' channel by next conditions
    result_np1 = np.where((img_slice > h_argleft) & (img_slice < h_argright), 255, 0)
    result_np2 = np.where(hist_s[img_slice.flatten()] > 0.1, 255, 0).reshape(height, width)
    result_np = np.where(result_np1 == result_np2, result_np1, 0)
    result_np = np.uint8(result_np)

    return result_np

def LineDistance(line1_p1,line1_p2, line2_p):
    p1 = np.asarray((line1_p1[0], line1_p1[1]))
    p2 = np.asarray((line1_p2[0], line1_p2[1]))
    p3 = np.asarray((line2_p[0], line2_p[1]))
    distance = int(np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1))
    return distance

def SegmentLines(lines, **kwargs):
    if len(lines) < 2:
        return [lines]
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    _1, labels, _2 = cv2.kmeans(np.float32(lines)[:, 4], 2, None, criteria, attempts, flags)

    segments = [[], []]
    for x in range(len(lines)):
        segments[labels[x][0]].append(lines[x])

    avg_1st, _, _ = LineStatistics(segments[0])
    avg_2nd, _, _ = LineStatistics(segments[1])
    if avg_2nd < avg_1st and (avg_1st < 0 or avg_2nd < 0):
        temp = segments[0]
        segments[0] = segments[1]
        segments[1] = temp

    return segments

def LineStatistics(lines, kwarg="default"):
    if len(lines) > 1:
        if kwarg == "default":
            sliced = np.asarray(lines)[:, 4]
        else:
            sliced = np.asarray([abs(ele[4]) for ele in lines])[:,]
        avg = sum(sliced) / len(lines)
        std = statistics.stdev(sliced)
        dif = abs(max(sliced) - min(sliced))
        return avg, std, dif
    else:
        return abs(lines[0][4]), -1, -1

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

                if (dif_angle <= 2 and distance < 20) or (dif_angle < 8 and dif_angle > 2 and distance < 30):
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
        angle_avg = np.rad2deg(np.arctan2(average_y1 - average_y2, average_x1 - average_x2))

        if unique == True:
            output.append([average_x1,average_y1,average_x2,average_y2,angle_avg])
        else:
            for x in range(len(group)):
                output.append([average_x1,average_y1,average_x2,average_y2,angle_avg])
        group.clear()
        i += 1

    return output

def DetectPart(segments):
    angles_dif = []
    for i in range(len(segments[0])):
        angle_1st = abs(segments[0][i][4])
        for j in range(len(segments[1])):
            angle_2nd = abs(segments[1][j][4])
            angles_dif.append(abs(angle_1st - angle_2nd))
    avg_dif = sum(angles_dif) / len(angles_dif)

    if avg_dif < 25:
        avg_1st, _, _ = LineStatistics(segments[0], "absolute")
        avg_2nd, _, _ = LineStatistics(segments[1], "absolute")
        if avg_1st < avg_2nd:
            return "right"
        else:
            return "left"
    else:
        return "center"

def PreprocImage(image):
    field_mask = HueMask(image)

    field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    field_mask = cv2.dilate(field_mask, np.ones((60, 60), np.uint8),iterations = 1)
    field_mask = cv2.erode(field_mask, np.ones((60, 60), np.uint8),iterations = 1)

    image = cv2.bitwise_and(image, image, mask=field_mask)
    image = cv2.GaussianBlur(image, (7, 7), 0)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    mask = cv2.Canny(gray,35,90,apertureSize = 3)

    field_mask = cv2.erode(field_mask, np.ones((10, 10), np.uint8),iterations = 1)

    mask = cv2.bitwise_and(mask, mask, mask=field_mask)

    return mask

def LineDetection(image):

    message = "unknown"

    _lines = cv2.HoughLinesP(image,rho = 1,theta = 1*np.pi/180,threshold = 200,minLineLength = 200,maxLineGap = 30)
    if _lines is None:
        _lines = cv2.HoughLinesP(image,rho = 1,theta = 1*np.pi/180,threshold = 130,minLineLength = 130,maxLineGap = 50)

    if _lines is None:
        cv2.rectangle(origin, (10, 10), (260, 70), (255, 255, 255), cv2.FILLED)
        cv2.putText(origin, message, (50,50), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,0), 1)
        return origin

    lines = []
    for x in range(0, len(_lines)):
        x1, y1, x2, y2 = _lines[x][0];
        angle = np.rad2deg(np.arctan2(y1 - y2, x1 - x2))
        lines.append([x1, y1, x2, y2, angle])

    grouped_lines = GroupLines(lines)
    _, _, dif = LineStatistics(grouped_lines, "absolute")

    if dif > 5:
        segments = SegmentLines(grouped_lines)
        if len(segments) > 1:
            message = DetectPart(segments)

    return message



video = cv2.VideoCapture(PATH_TO_VIDEO)

message_act = 'unknown'
message_place = 'unknown'

count = 0
while(video.isOpened()):
    count += 1
    ret, frame = video.read()
    frame_expanded = np.expand_dims(frame, axis=0)

    (boxes_ball, scores_ball, classes_ball, num_ball) = sess_ball.run(
        [detection_boxes_ball, detection_scores_ball, detection_classes_ball, num_detections_ball],
        feed_dict={image_tensor_ball: frame_expanded})

    (boxes_player, scores_player, classes_player, num_player) = sess_player.run(
        [detection_boxes_player, detection_scores_player, detection_classes_player, num_detections_player],
        feed_dict={image_tensor_player: frame_expanded})

    prep_boxes_b, prep_classes_b, prep_scores_b = PreprocData(
    boxes_ball,
    scores_ball,
    classes_ball,
    TRESHOLD_ball,
    frame)

    prep_boxes_p, prep_classes_p, prep_scores_p = PreprocData(
    boxes_player,
    scores_player,
    classes_player,
    TRESHOLD_player,
    frame)

    AccumData(prep_boxes_b, prep_boxes_p, frame)

    if count % 5 == 0:
        message_act = AccumAnalyze()

    mask = PreprocImage(frame)
    message_place = LineDetection(mask)

    frame = CustomViusalize(prep_boxes_b, frame, [253,246,219], 2)
    frame = CustomViusalize(prep_boxes_p, frame, [220,92,93], 2)

    cv2.rectangle(frame, (280, 10), (530, 70), (255, 255, 255), cv2.FILLED)
    cv2.putText(frame, message_place, (320,50), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,0), 1)

    cv2.rectangle(frame, (10, 10), (260, 70), (255, 255, 255), cv2.FILLED)
    cv2.putText(frame, message_act, (50,50), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,0), 1)

    cv2.imshow('Object detector', frame)

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
