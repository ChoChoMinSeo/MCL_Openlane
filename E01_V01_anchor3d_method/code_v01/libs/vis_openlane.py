import numpy as np
import cv2
import os
import os.path as ops
import copy
import math
import ujson as json
from scipy.interpolate import interp1d
import matplotlib
from tqdm import tqdm
import warnings
import pickle
import pdb
from mpl_toolkits.mplot3d import Axes3D

# matplotlib.use('Agg')
import matplotlib.pyplot as plt

from libs.utils_anchor3d import *
from libs.utils import *


# plt.rcParams['figure.figsize'] = (35, 30)
# plt.rcParams.update({'font.size': 25})
# plt.rcParams.update({'font.weight': 'semibold'})

color = [[0, 0, 255],  # red
         [0, 255, 0],  # green
         [255, 0, 255],  # purple
         [255, 255, 0]]  # cyan

vis_min_y = 5
vis_max_y = 80

class LaneVis(object):
    def __init__(self, db):
        self.resize_h = db.resize_h
        self.resize_w = db.resize_w
        self.H_crop = homography_crop_resize([db.org_h, db.org_w], db.crop_y, [db.resize_h, db.resize_w])
        self.top_view_region = db.top_view_region
        self.ipm_h = db.ipm_h
        self.ipm_w = db.ipm_w
        self.org_h = db.org_h
        self.org_w = db.org_w
        self.crop_y = db.crop_y
        self.x_min = db.top_view_region[0, 0]
        self.x_max = db.top_view_region[1, 0]
        self.y_min = db.top_view_region[2, 1]
        self.y_max = db.top_view_region[0, 1]
        self.y_samples = np.array([  5,  10,  15,  20,  30,  40,  50,  60,  80,  100], dtype=np.float32)
        self.dist_th = 1.5
        self.ratio_th = 0.75
        self.close_range = 40
        self.num_category = 21
        self.category_dict = {0: 'invalid',
                              1: 'white-dash',
                              2: 'white-solid',
                              3: 'double-white-dash',
                              4: 'double-white-solid',
                              5: 'white-ldash-rsolid',
                              6: 'white-lsolid-rdash',
                              7: 'yellow-dash',
                              8: 'yellow-solid',
                              9: 'double-yellow-dash',
                              10: 'double-yellow-solid',
                              11: 'yellow-ldash-rsolid',
                              12: 'yellow-lsolid-rdash',
                              13: 'fishbone',
                              14: 'others',
                              20: 'roadedge'}

    def vis(self, gt, save_dir, img_dir, img_name):
        img_path = os.path.join(img_dir, 'images', img_name)
        fig = plt.figure()
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133, projection='3d')

        # evaluate lanelines
        cam_extrinsics = np.array(gt['extrinsic'])
        # Re-calculate extrinsic matrix based on ground coordinate
        R_vg = np.array([[0, 1, 0],
                            [-1, 0, 0],
                            [0, 0, 1]], dtype=float)
        R_gc = np.array([[1, 0, 0],
                            [0, 0, 1],
                            [0, -1, 0]], dtype=float)
        cam_extrinsics[:3, :3] = np.matmul(np.matmul(
                                    np.matmul(np.linalg.inv(R_vg), cam_extrinsics[:3, :3]),
                                        R_vg), R_gc)

        cam_extrinsics[0:2, 3] = 0.0

        cam_intrinsics = gt['intrinsic']
        cam_intrinsics = np.array(cam_intrinsics)
        gt_lanes_packed = gt['lane_lines']

        gt_lanes, gt_visibility, gt_category = [], [], []
        for j, gt_lane_packed in enumerate(gt_lanes_packed):
            # A GT lane can be either 2D or 3D
            # if a GT lane is 3D, the height is intact from 3D GT, so keep it intact here too
            lane = np.array(gt_lane_packed['xyz'])
            lane_visibility = np.ones(len(lane[0]))

            lane = np.vstack((lane, np.ones((1, lane.shape[1]))))
            cam_representation = np.linalg.inv(
                                    np.array([[0, 0, 1, 0],
                                                [-1, 0, 0, 0],
                                                [0, -1, 0, 0],
                                                [0, 0, 0, 1]], dtype=float))

            lane = np.matmul(cam_extrinsics, np.matmul(cam_representation, lane))
            lane = lane[0:3, :].T

            gt_lanes.append(lane)
            gt_visibility.append(lane_visibility)
            gt_category.append(gt_lane_packed['category'])

        # only consider those gt lanes overlapping with sampling range
        gt_category = [gt_category[k] for k, lane in enumerate(gt_lanes)
                        if lane[0, 1] < self.y_samples[-1] and lane[-1, 1] > self.y_samples[0]]
        gt_lanes = [lane for lane in gt_lanes if lane[0, 1] < self.y_samples[-1] and lane[-1, 1] > self.y_samples[0]]

        gt_lanes = [prune_3d_lane_by_range(np.array(lane), self.x_min, self.x_max) for lane in gt_lanes]

        gt_category = [gt_category[k] for k, lane in enumerate(gt_lanes) if lane.shape[0] > 1]
        gt_lanes = [lane for lane in gt_lanes if lane.shape[0] > 1]

        cnt_gt = len(gt_lanes)

        P_g2im = projection_g2im_extrinsic(cam_extrinsics, cam_intrinsics)
        P_gt = np.matmul(self.H_crop, P_g2im)
        img = cv2.imread(img_path)
        img = cv2.warpPerspective(img, self.H_crop, (self.resize_w, self.resize_h))
        img = img.astype(np.float) / 255
        
        H_ipm2g = cv2.getPerspectiveTransform(
            np.float32([[0, 0], [self.ipm_w - 1, 0], [0, self.ipm_h - 1], [self.ipm_w - 1, self.ipm_h - 1]]), 
            np.float32(self.top_view_region))
        H_g2ipm = np.linalg.inv(H_ipm2g)
        H_g2im = homography_g2im_extrinsic(cam_extrinsics, cam_intrinsics)
        H_im2g = np.linalg.inv(H_g2im)
        H_im2ipm = np.linalg.inv(np.matmul(H_g2im, H_ipm2g))
        raw_img = cv2.imread(img_path)
        raw_img = raw_img.astype(np.float) / 255
        im_ipm = cv2.warpPerspective(raw_img, H_im2ipm, (self.ipm_w, self.ipm_h))
        im_ipm = np.clip(im_ipm, 0, 1)

        ax1.imshow(img[:, :, [2, 1, 0]])
        ax2.imshow(im_ipm[:, :, [2, 1, 0]])

        gt_visibility_mat = np.zeros((cnt_gt, 10))

        for i in range(cnt_gt):
            min_y = np.min(np.array(gt_lanes[i])[:, 1])
            max_y = np.max(np.array(gt_lanes[i])[:, 1])
            x_values, z_values, visibility_vec = resample_laneline_in_y(np.array(gt_lanes[i]), self.y_samples,
                                                                        out_vis=True)
            gt_lanes[i] = np.vstack([x_values, z_values]).T
            gt_visibility_mat[i, :] = np.logical_and(x_values >= self.x_min,
                                                     np.logical_and(x_values <= self.x_max,
                                                                    np.logical_and(self.y_samples >= min_y,
                                                                                   self.y_samples <= max_y)))
            gt_visibility_mat[i, :] = np.logical_and(gt_visibility_mat[i, :], visibility_vec)

        
        for i in range(cnt_gt):
            x_values = gt_lanes[i][:, 0]
            z_values = gt_lanes[i][:, 1]
            x_g = x_values[np.where(gt_visibility_mat[i, :])]
            if len(x_g) == 0:
                continue
            y_g = self.y_samples[np.where(gt_visibility_mat[i, :])]
            z_g = z_values[np.where(gt_visibility_mat[i, :])]
            fit1 = np.polyfit(y_g, x_g, 2)
            fit2 = np.polyfit(y_g, z_g,2)
            f_xy = np.poly1d(fit1)
            f_zy = np.poly1d(fit2)
            y_g = np.linspace(min(y_g), max(y_g), 5 * len(y_g))
            x_g = f_xy(y_g)
            z_g = f_zy(y_g)
            ax3.plot(x_g, y_g, z_g, lw=2, c='red', alpha=1, label='gt')

            x_2d, y_2d = projective_transformation(P_gt, x_g, y_g, z_g)
            valid_mask_2d = np.logical_and(np.logical_and(x_2d >= 0, x_2d < self.resize_w), np.logical_and(y_2d >= 0, y_2d < self.resize_h))
            x_2d = x_2d[valid_mask_2d]
            y_2d = y_2d[valid_mask_2d]
            ax1.plot(x_2d, y_2d, lw=2, c='red', alpha=1, label='gt')

            x_ipm, y_ipm = homographic_transformation(H_g2ipm, x_g, y_g)
            valid_mask_ipm = np.logical_and(np.logical_and(x_ipm >= 0, x_ipm < self.ipm_w), np.logical_and(y_ipm >= 0, y_ipm < self.ipm_h))
            x_ipm = x_ipm[valid_mask_ipm]
            y_ipm = y_ipm[valid_mask_ipm]
            ax2.plot(x_ipm, y_ipm, lw=2, c='red', alpha=1, label='gt')

        bottom, top = ax3.get_zlim()
        left, right = ax3.get_xlim()
        ax3.set_zlim(min(bottom, -0.1), max(top, 0.1))
        ax3.set_xlim(left, right)
        ax3.set_ylim(vis_min_y, vis_max_y)
        ax3.locator_params(nbins=5, axis='x')
        ax3.locator_params(nbins=10, axis='z')
        ax3.tick_params(pad=18, labelsize=15)
        print("save to", ops.join(save_dir, img_name.replace("/", "_").replace('jpg','png')))
        fig.savefig(ops.join(save_dir, img_name.replace("/", "_").replace('jpg','png')))
        plt.close(fig)


    def visualize(self,  gt_file, img_dir=None, save_dir=None, vis_step=20):
        mkdir(save_dir)
        json_gt = [json.loads(line) for line in open(gt_file).readlines()]
        gts = {l['file_path']: l for l in json_gt}
        for i, pred in tqdm(enumerate(json_gt)):
            if i % vis_step > 0:
                continue
            raw_file = pred['file_path']
            gt = gts[raw_file]
            self.vis(gt, save_dir, img_dir, raw_file)