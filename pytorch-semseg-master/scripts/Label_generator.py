
import os
import glob
import cv2
import sys
import time
import numpy as np
import tensorflow as tf
import xml.etree.cElementTree as ET

sys.path.append("..")

# ПАРАМЕТРЫ

FRAME_STEP = 20
NUM_CLASSES = 3
TRESHOLD = 0.8
input_directory = "C:/examples/"
output_directory = "C:/results/"

# ИНИЦИАЛИЗАЦИЯ СЕТИ
#"""
from utils import label_map_util
from utils import visualization_utils as vis_util

CWD_PATH = os.getcwd()

MODEL_NAME = 'inference_graph'

PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

num_detections = detection_graph.get_tensor_by_name('num_detections:0')
#"""

def AddObject(tree, name, coord):
    object = ET.SubElement(tree, "object")
    ET.SubElement(object, "name").text = name
    ET.SubElement(object, "pose").text = "Unspecified"
    ET.SubElement(object, "truncated").text = "0"
    ET.SubElement(object, "difficult").text = "0"
    bndbox = ET.SubElement(object, "bndbox")
    ET.SubElement(bndbox, "xmin").text = str(int(coord[1]))
    ET.SubElement(bndbox, "ymin").text = str(int(coord[0]))
    ET.SubElement(bndbox, "xmax").text = str(int(coord[3]))
    ET.SubElement(bndbox, "ymax").text = str(int(coord[2]))

def CreateAnnotation(folder, filename, path, dimentions):
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = str(folder)
    ET.SubElement(annotation, "filename").text = str(filename)
    ET.SubElement(annotation, "path").text = str(path)
    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "Unknown"
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(dimentions[0])
    ET.SubElement(size, "height").text = str(dimentions[1])
    ET.SubElement(size, "depth").text = "3"
    ET.SubElement(annotation, "segmented").text = "0"
    return annotation

all_video = []
os.chdir(input_directory)
for file in glob.glob('*.mp4'):
    all_video.append(input_directory + str(file))

images_path = os.path.join(input_directory,'*g')
image_files = glob.glob(images_path)

image_list = []

#for file in image_files:
    #img = cv2.imread(file)
    #image_list.append(img)


#filelist = glob.glob(os.path.join(output_directory, "*.bak"))
#for f in filelist:
    #os.remove(f)

all_video_frames = []

if len(all_video) != 0:
    for video in all_video:
        vidcap = cv2.VideoCapture(video)
        success,frame = vidcap.read()
        count = 0

        video_frames = []
        while success:
            success,frame = vidcap.read()
            if frame is not None and count % FRAME_STEP == 0:
                video_frames.append(frame)
            count += 1
        all_video_frames.append(video_frames)

all_video_frames.append(image_list)

start = time.time()

vid_count = 0
for vid_frames in all_video_frames:
    vid_count += 1
    os.mkdir(output_directory + str(vid_count))
    frame_count = 0
    for frame in vid_frames:
        frame_count += 1

        image_expanded = np.expand_dims(frame, axis=0)
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})

        Width = np.size(frame, 1)
        Height = np.size(frame, 0)

        for a in boxes[0]:
            a[0] = a[0] * Height
            a[1] = a[1] * Width
            a[2] = a[2] * Height
            a[3] = a[3] * Width

        tresh_boxes = []
        tresh_classes = []
        tresh_scores = []

        for j in range(classes.size):
            if scores[0][j] > TRESHOLD:
                tresh_scores.append(scores[0][j])
                tresh_classes.append(classes[0][j])
                tresh_boxes.append(boxes[0][j])

        classes_name = [category_index.get(value) for index, value in enumerate(tresh_classes)]

        uniq_classes = []
        for i in range(len(tresh_boxes)):
            uniq_classes.append(classes_name[i].get("name"))
        uniq_classes = np.unique(uniq_classes)

        image_path = output_directory + str(vid_count) + '/' + str(frame_count) + '.jpg'
        xml_path = output_directory + str(vid_count) + '/' + str(frame_count) + '.xml'

        annotation = CreateAnnotation(str(vid_count), str(frame_count) + '.jpg', image_path.replace("/", "\\"), [Width, Height])
        for x in range(len(uniq_classes)):
            for i in range(len(tresh_boxes)):
                if classes_name[i].get("name") == uniq_classes[x]:
                    AddObject(annotation, classes_name[i].get("name"), tresh_boxes[i])
        ET.ElementTree(annotation).write(xml_path)
        cv2.imwrite(image_path, frame)

        print(vid_count, "/", len(all_video_frames), " video\t", frame_count, "/", len(vid_frames), " frame")

end = time.time()
print ("done in ", round((end - start), 2), "second")
