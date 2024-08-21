import os

class Config(object):
    def __init__(self):
        # --------basics-------- #
        self.setting_for_path()
        self.setting_for_image_param()

    def setting_for_path(self):
        self.pc = 'main'
        self.dir = dict()
        self.setting_for_dataset_path()  # dataset path

        self.dir['proj'] = os.path.dirname(os.getcwd())
        self.dir['head_proj'] = os.path.dirname(self.dir['proj'])
        # self.dir['pre0'] = f'{self.dir["head_proj"]}/E00_V01_data_processing/output_v01_{self.datalist_mode}/pickle'
        self.dir['pre0'] = f'{self.dir["head_proj"]}/E01_V08_lane_selection_after_als2/output_v02_{self.datalist_mode}/pickle'
        self.dir['backup'] = f'{self.dir["head_proj"]}/E01_V08_lane_selection_after_als2/output_v02_{self.datalist_mode}/pickle/results'
        self.dir['out'] = f'{os.getcwd().replace("code", "output")}_{self.datalist_mode}'
        self.dir['data_path'] = self.dir['out']+'/pickle/results'


    def setting_for_dataset_path(self):
        self.datalist_mode = 'validation'  # ['training', 'testing', 'validation','example']
        # ------------------- need to modify -------------------
        self.dir['dataset'] = '--dataset path'
        # ------------------------------------------------------

    def setting_for_image_param(self):
        self.org_height = 1280
        self.org_width = 1920
        self.height = self.org_height // 4
        self.width = self.org_width // 4

        self.max_x = 10
        self.min_x = -10
        self.max_y = 103
        self.min_y = 3

        self.org_h = 1280
        self.org_w = 1920
        self.resize_h = 360
        self.resize_w = 480
        self.crop_y = 0

        self.size = [self.width, self.height, self.width, self.height]
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]