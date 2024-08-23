import os
import torch

class Config(object):
    def __init__(self):
        # --------basics-------- #
        self.setting_for_system()
        self.setting_for_path()
        self.setting_for_save()

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
        print(self.dir['proj'])
        self.dir['out'] = f'{os.getcwd().replace("code_", "output_")}_{self.datalist_mode}'
        print(self.dir['out'])

    def setting_for_dataset_path(self):
        self.dataset = 'openlane'  # ['vil100']
        self.datalist_mode = 'training'  # ['training', 'testing', 'validation']

        # ------------------- need to modify -------------------
        self.dir['dataset'] = '--dataset_dir'
        # ------------------------------------------------------

    def setting_for_save(self):
        self.save_pickle = True
