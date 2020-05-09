import cv2, time, os, sys, glob, shutil, random
import numpy as np
from datetime import datetime


class DataLoader():
    def __init__(self, input, output, ignore_nesting, frames_to_skip):
        self.input = input
        self.output = output
        self.nfskip = max(frames_to_skip, 0)
        self.ignore_nesting = ignore_nesting

        self.nImgs = 0
        self.nVids = 0
        self.cImg = 0
        self.cVid = 0
        self.fps = 0
        self.nframes = 0
        self.rvid = None
        self.svid = None
        self.type = ''

        self.__walk_tree__()

        self.last = self.nImgs + self.nVids - 1

    def __walk_tree__(self):
        self.data = [[], [], [], []]

        if os.path.exists(self.output):
            shutil.rmtree(self.output)

        for root, dirs, files in os.walk(self.input):
            root = root.replace('\\', '/')
            if not root.endswith('/'):
                root += '/'

            imgs = glob.glob(root+'*g')
            vids = glob.glob(root+'*.mp4')

            self.nImgs += len(imgs)
            self.nVids += len(vids)

            self.data[0] += [path.replace('\\', '/') for path in imgs]
            self.data[1] += ['image'] * len(imgs)

            self.data[0] += [path.replace('\\', '/') for path in vids]
            self.data[1] += ['video'] * len(vids)

            if not self.ignore_nesting:
                os.mkdir(root.replace(self.input, self.output))

        if self.ignore_nesting:
            self.data[2] = [str(x) for x in range(len(self.data[1]))]
            self.data[3] = [self.output for x in range(len(self.data[1]))]
            pass
        else:
            self.data[2] = [os.path.split(path)[1].split('.')[0] for path in self.data[0]]
            self.data[3] = [os.path.split(path.replace(self.input, self.output))[0] for path in self.data[0]]
            pass

        self.data = [list(item) for item in zip(*self.data)]

    def __init_rvid__(self):
        self.rvid = cv2.VideoCapture(self.data[self.iter][0])
        self.cVid += 1
        self.frame = 0
        self.nframes = int(self.rvid.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.rvid.get(cv2.CAP_PROP_FPS) / (self.nfskip+1))

    def __init_svid__(self, img):
        h, w = img.shape[:2]
        path = '{}/{}.mp4'.format(
        self.data[self.iter][3], self.data[self.iter][2])
        self.svid = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (w, h))

    def __release__(self):
        if self.rvid:
            self.rvid.release()
        if self.svid:
            self.svid.release()


    def __iter__(self):
        self.iter = -1
        self.frame = 0
        return self

    def __next__(self):
        if self.iter == self.last and self.frame + self.nfskip >= self.nframes:
            self.__release__()
            raise StopIteration
        else:
            self.type = self.data[max(self.iter, 0)][1][0]
            if self.type == 'image':
                self.iter += 1
            elif self.frame + self.nfskip >= self.nframes:
                self.__release__()
                self.iter += 1

            item = self.data[self.iter]
            self.type = item[1]
            if self.type == 'image':
                self.cImg += 1
                img = cv2.imread(item[0])
            else:
                if self.frame + self.nfskip >= self.nframes:
                    self.__init_rvid__()
                while self.frame % (self.nfskip+1) != 0:
                    self.frame += 1
                    _, img = self.rvid.read()
                self.frame += 1
                _, img = self.rvid.read()
            return img, item[1], item[2], item[3]


    def save_results(self, img):
        if self.type == 'video':
            if not self.svid:
                self.__init_svid__(img)
            self.svid.write(img)
        else:
            path = '{}/{}.jpg'.format(
            self.data[self.iter][3], self.data[self.iter][2])
            cv2.imwrite(path, img)

    def print_status(self):
        if self.type == 'video':
            print('Video: {}/{}   Frame: {}/{}'.format(
            self.cVid, self.nVids, self.frame, self.nframes), end='    ')
        else:
            print('Image: {}/{}'.format(self.cImg, self.nImgs), end='    ')


class WebcamLoader():
    def __init__(self, output):
        self.output = output

        if os.path.exists(self.output):
            shutil.rmtree(self.output)
        os.mkdir(self.output)

        self.rvid = cv2.VideoCapture(0)
        self.svid = None

    def __init_svid__(self, img):
        h, w = img.shape[:2]
        fps = self.rvid.get(cv2.CAP_PROP_FPS)
        path = self.output + str(datetime.fromtimestamp(time.time())).split('.')[0] + '.mp4'
        self.svid = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    def get_item(self):
        success, img = self.rvid.read()
        if success:
            return img
        else:
            return None

    def save_item(self, img):
        if not self.svid:
            self.__init_svid__(img)
        self.svid.write(img)

    def release_all(self):
        if self.svid:
            self.svid.release()
        if self.rvid:
            self.rvid.release()

class StreamLoader():
    def __init__(self):
        pass

# rebuild!!!
class DatasetLoader():
    def __init__(self, input, shuffle, names, type=None):
        self.input = input
        self.names = names
        self.type = type

        self.data = []

        imgs, xmls, txts = self.__walk_tree__()
        self.__preproc_data__(imgs, xmls, txts)

        self.iter = -1
        self.last = len(self.data) - 1

        if shuffle:
            self.indices = random.sample(range(0, self.last), len(self.data))
        else:
            self.indices = [x for x in range(len(self.data))]

    def __walk_tree__(self):
        imgs = []
        xmls = []
        txts = []

        if not os.path.exists(self.input):
            return imgs, xmls, txts

        for root, dirs, files in os.walk(self.input):
            if not root.endswith('/'):
                root += '/'
            if self.type:
                if self.type in root:
                    imgs.append(glob.glob(root+'*g'))
                    xmls.append(glob.glob(root+'*.xml'))
                    txts.append(glob.glob(root+'*.txt'))
            else:
                imgs.append(glob.glob(root+'*g'))
                xmls.append(glob.glob(root+'*.xml'))
                txts.append(glob.glob(root+'*.txt'))

        return imgs, xmls, txts

    def __preproc_data__(self, imgs, xmls, txts):
        temp_names = []
        count = 0
        for i in range(len(imgs)):
            xml_names = [os.path.splitext(os.path.basename(path.replace('\\', '/')))[0] for path in xmls[i]]
            txt_names = [os.path.splitext(os.path.basename(path.replace('\\', '/')))[0] for path in txts[i]]
            img_names = [os.path.splitext(os.path.basename(path.replace('\\', '/')))[0] for path in imgs[i]]

            for j in range(len(imgs[i])):
                img_path = imgs[i][j]

                try:
                    xml_index = xml_names.index(img_names[j])
                except:
                    xml_index = None
                try:
                    txt_index = txt_names.index(img_names[j])
                except:
                    txt_index = None

                if xml_index:
                    objects = annotation.parse_xml(xmls[i][xml_index])
                elif txt_index:
                    h, w = cv2.imread(img_path).shape[:2]
                    objects = annotation.parse_txt(txts[i][txt_index], self.names, [w, h])
                else:
                    objects = []

                self.data.append([img_path, objects])


    def get_item(self):
        if self.iter < self.last:
            self.iter += 1
            path, objects = self.data[self.indices[self.iter]]
            #return [cv2.imread(path, cv2.IMREAD_UNCHANGED), objects]
            return [cv2.imread(path), objects]
        else:
            return None
