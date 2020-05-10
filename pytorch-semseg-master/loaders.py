import cv2, time, os, sys, glob, shutil
from datetime import datetime

class DataLoader():
    def __init__(self, input, output):
        self.input = input
        self.output = output

        self.nImgs = 0
        self.nVids = 0
        self.cImg = 0
        self.cVid = 0
        self.fps = 0
        self.nframes = 0
        self.rvid = None
        self.svid = None
        self.type = ''

        self.CROSSROADS_MAP = ['oktrev', 'gavrilova', 'lenina']

        self.__walk_tree__()

        self.last = self.nImgs + self.nVids - 1

    def __walk_tree__(self):
        self.data = [[], []]

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

            os.mkdir(root.replace(self.input, self.output))

        self.data = [list(item) for item in zip(*self.data)]

    def __init_rvid__(self):
        self.rvid = cv2.VideoCapture(self.data[self.iter][0])
        self.cVid += 1
        self.frame = 0
        self.nframes = int(self.rvid.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.rvid.get(cv2.CAP_PROP_FPS)

    def __init_svid__(self, img):
        h, w = img.shape[:2]
        print(self.data[self.iter][0].replace(self.input, self.output))
        self.svid = cv2.VideoWriter(
        self.data[self.iter][0].replace(self.input, self.output),
        cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (w, h))

    def __release__(self):
        if self.rvid:
            self.rvid.release()
            self.rvid = None
        if self.svid:
            self.svid.release()
            self.svid = None

    def __iter__(self):
        self.iter = -1
        self.frame = 0
        return self

    def __next__(self):
        if self.iter == self.last and self.frame >= self.nframes:
            self.__release__()
            raise StopIteration
        else:
            self.type = self.data[max(self.iter, 0)][1][0]
            if self.type == 'image':
                self.iter += 1
            elif self.frame >= self.nframes:
                self.__release__()
                self.iter += 1

            item = self.data[self.iter]
            self.type = item[1]
            if self.type == 'image':
                self.cImg += 1
                img = cv2.imread(item[0])
            else:
                if self.frame >= self.nframes:
                    self.__init_rvid__()
                self.frame += 1
                _, img = self.rvid.read()
            return img


    def save_results(self, img):
        if self.type == 'video':
            if not self.svid:
                self.__init_svid__(img)
            self.svid.write(img)
        else:
            cv2.imwrite(self.data[self.iter][0].replace(self.input, self.output), img)
