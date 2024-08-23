import os
import torch

class Config(object):
    def __init__(self):
        # --------basics-------- #
        self.setting_for_system()
        self.setting_for_path()
        self.setting_for_save()
        self.setting_for_image_param()
        self.setting_for_dataloader()

    def setting_for_system(self):
        self.gpu_id = "0"
        self.seed = 123
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_id
        torch.backends.cudnn.deterministic = True

    def setting_for_path(self):
        self.dataset = 'openlane'  # ['vil100']
        self.datalist_mode = 'validation'  # ['training', 'testing', 'validation']

        self.pc = 'main'
        self.dir = dict()
        # ------------------- need to modify -------------------
        self.dir['dataset'] = '--dataset_dir'
        # ------------------------------------------------------
        
        self.dir['proj'] = os.path.dirname(os.getcwd())
        self.dir['datalist'] = f'/home/asdf/바탕화면/Minseo/openlane/3d_ALS/E00_V01_data_processing/output_v01_{self.datalist_mode}'
        self.dir['als'] = f'/home/asdf/바탕화면/Minseo/openlane/3d_ALS/E01_V07_ALS/output_v06_debug_{self.datalist_mode}/pickle/results'

        self.dir['out'] = f'{os.getcwd().replace("code_", "output_")}_{self.datalist_mode}'


    def setting_for_save(self):
        self.save_pickle = True

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
        self.min_x = -20
        self.max_x = 20
        self.min_z = -10
        self.max_z = 10

    def setting_for_dataloader(self):
        self.num_workers = 4
        self.target_sample = 5
        self.multiprocessing = True