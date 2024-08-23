import argparse
import os

parser = argparse.ArgumentParser()

"""User Arguments"""
parser.add_argument('--run_mode', type=str, default='test')

"""Convert"""
cfg_vis = parser.parse_args()

# """Path"""
# cfg.dir = dict()
"""Mode"""
cfg_vis.mode = "validation" # ["train", "validation"]
# Output
cfg_vis.dir = dict()
cfg_vis.dir['out'] = f'{os.getcwd().replace("code", "output")}_{cfg_vis.mode}'
"""Dataset"""
cfg_vis.extra_mode = 'linear'
cfg_vis.degree = 3

cfg_vis.org_h = 1280
cfg_vis.org_w = 1920
cfg_vis.resize_h = 360
cfg_vis.resize_w = 480
cfg_vis.crop_y = 0
cfg_vis.top_view_region = [[-10, 103], [10, 103], [-10, 3], [10, 3]]
cfg_vis.y_samples = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
# cfg_vis.y_samples = range(5,201,5)


cfg_vis.batch_size = 8
