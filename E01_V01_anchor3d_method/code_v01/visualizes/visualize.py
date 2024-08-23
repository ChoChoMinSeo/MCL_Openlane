import cv2
import numpy as np

from libs.utils import *
from visualizes.visualize_module import Runner
from visualizes.vis_config import cfg_vis

class Visualize(object):

    def __init__(self, cfg):
        self.cfg = cfg
        self.mean = np.array([cfg.mean], dtype=np.float32)
        self.std = np.array([cfg.std], dtype=np.float32)
        self.line = np.zeros((cfg.height, 3, 3), dtype=np.uint8)
        self.line[:, :, :] = 255

        self.show = dict()
        self.db = dict()
        self.runner = Runner(cfg_vis)

    def update_label(self, label, name='label'):
        label = to_np(label)
        self.show[name] = label

    def update_image_name(self, img_name):
        self.show['img_name'] = img_name

    def update_datalist(self, img_name, label, dir_name, file_name, img_idx):
        self.update_image_name(img_name)
        self.update_label(label,'label')

        self.dir_name = dir_name
        self.file_name = file_name
        self.img_idx = img_idx

    def save_datalist(self,data,file_name):
        if self.cfg.datalist_mode =='example':
            img = cv2.imread(f'{self.cfg.dir["dataset"]}/images/validation/{file_name}.jpg')
        else:
            img = cv2.imread(f'{self.cfg.dir["dataset"]}/images/{self.cfg.datalist_mode}/{file_name}.jpg')
        dir_name = f'{self.cfg.dir["out"]}/display/'
        self.runner.run(img = img,data = data,path=dir_name+file_name)