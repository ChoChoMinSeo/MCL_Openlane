import os
import torch

import numpy as np

class Config(object):
    def __init__(self):
        # --------basics-------- #
        self.setting_for_system()
        self.setting_for_path()
        self.setting_for_image_param()
        self.setting_for_dataloader()
        self.setting_for_visualization()
        self.setting_for_save()
        # --------others-------- #
        # self.setting_for_lane_detection()

    def setting_for_system(self):
        self.gpu_id = "0"
        self.seed = 123
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_id
        torch.backends.cudnn.deterministic = True

    def setting_for_path(self):
        self.pc = 'main'
        self.dir = dict()
        self.setting_for_dataset_path()  # dataset path
        self.dir['proj'] = os.path.dirname(os.getcwd())
        self.dir['head_proj'] = os.path.dirname(self.dir['proj'])
        self.dir['pre0'] = f'{self.dir["head_proj"]}/E00_V01_data_processing/output_v01_{self.datalist_mode}/pickle'
        self.dir['pre1_5'] = f'{self.dir["head_proj"]}/E01_V05_lane_merge_spline/output_v02_{self.datalist_mode}_for_supp/pickle'
        self.dir['pre1_6'] = f'{self.dir["head_proj"]}/E01_V06_parallel/output_v02_{self.datalist_mode}_for_supp/pickle'

        self.dir['out'] = f'{os.getcwd().replace("code", "output")}_{self.datalist_mode}'

    def setting_for_dataset_path(self):
        self.dataset = 'openlane'  # ['vil100']
        self.datalist_mode = 'example'  # ['training', 'testing', 'validation']

        # ------------------- need to modify -------------------
        self.dir['dataset'] = '--dataset path'
        # ------------------------------------------------------

    def setting_for_image_param(self):
        self.org_height = 1280
        self.org_width = 1920
        self.height = self.org_height // 4
        self.width = self.org_width // 4
        self.size = [self.width, self.height, self.width, self.height]
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.min_y = 3
        self.max_y = 103
        self.min_x = -10
        self.max_x = 10
        self.min_z = -10
        self.max_z = 10
        self.sampling_pts = range(5,106,5)

    def setting_for_dataloader(self):
        self.num_workers = 4

    def setting_for_visualization(self):
        self.display_all = True

    def setting_for_save(self):
        self.save_pickle = True