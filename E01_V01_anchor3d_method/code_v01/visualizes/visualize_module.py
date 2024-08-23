import os
import numpy as np

import cv2
import matplotlib.pyplot as plt
import matplotlib

from visualizes.utils.utils import *

class Runner(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.H_crop = homography_crop_resize([self.cfg.org_h, self.cfg.org_w], self.cfg.crop_y, [self.cfg.resize_h, self.cfg.resize_w])

        self.cam_representation = np.linalg.inv(
            np.array([[0, 0, 1, 0],
                      [-1, 0, 0, 0],
                      [0, -1, 0, 0],
                      [0, 0, 0, 1]], dtype=float))
        self.R_vg = np.array([[0, 1, 0],
                         [-1, 0, 0],
                         [0, 0, 1]], dtype=float)
        self.R_gc = np.array([[1, 0, 0],
                         [0, 0, 1],
                         [0, -1, 0]], dtype=float)
        self.inv_R_vg = np.linalg.inv(self.R_vg)

        self.x_min = self.cfg.top_view_region[0][0] # -10
        self.x_max = self.cfg.top_view_region[1][0] # 10
        self.y_min = self.cfg.top_view_region[2][1] # 3
        self.y_max = self.cfg.top_view_region[0][1] # 103
        self.vis_init = np.ones((len(self.cfg.y_samples)),dtype=np.bool_)

    def recalculate_extrinsic(self, extrinsic):
        # Re-calculate extrinsic matrix based on ground coordinate
        extrinsic[:3, :3] = np.matmul(np.matmul(
            np.matmul(self.inv_R_vg, extrinsic[:3, :3]),
            self.R_vg), self.R_gc)

        extrinsic[0:2, 3] = 0.0
        return extrinsic
    
    def get_3dlanes(self, lane, extrinsic):
        lane = lane[:,:,0].T  # [3, n]
        ones = np.ones((1, lane.shape[1]))
        lane = np.vstack((lane, ones))  # [4, n], (x, y, z, 1)
        lane = np.matmul(extrinsic, np.matmul(self.cam_representation, lane))  # [4, n]
        lane = lane[0:3, :].T  # [n, 3]
        return lane
    
    def get_camera_coordinates(self, lane, extrinsic):
        # lane: 실제 좌표 [n, 3]
        # extrinsic: 외부 파라미터 행렬 [4, 4]

        # ones 추가하여 [n, 4] 형태로 만들기
        ones = np.ones((lane.shape[0], 1))
        lane = np.hstack((lane, ones))  # [n, 4]

        # extrinsic과 self.cam_representation의 역행렬 계산
        extrinsic_inv = np.linalg.inv(extrinsic)
        cam_representation_inv = np.linalg.inv(self.cam_representation)

        # 실제 좌표에서 카메라 좌표로 변환
        lane = np.matmul(cam_representation_inv, np.matmul(extrinsic_inv, lane.T))  # [4, n]
        lane = lane[0:3, :].T  # [n, 3]

        return lane

    def visualize_gt_3dlanes(self, data, img, path):    
        # Load data
        extrinsic, intrinsic = data['extrinsic'], data['intrinsic']
        lanes = data["lane3d"]

        # Recalculate extrinsic
        # Calculate projection matrix
        P_g2im = projection_g2im_extrinsic(extrinsic, intrinsic)
        P_gt = np.matmul(self.H_crop, P_g2im)

        # Recalculate GT lane pts
        lanes_r_org = []
        lanes_r_new = []
        for i, lane in enumerate(lanes['org_lane']):
            # lanes_r_org.append(self.get_3dlanes(lane, extrinsic))
            lanes_r_org.append(lane)

        for i, lane in enumerate(lanes['new_lane']):
            # lanes_r_new.append(self.get_3dlanes(lane, extrinsic))
            lanes_r_new.append(lane)

        # Sample x, z in y direction and calculate visibility matrix
        lanes_s_org = []
        lanes_s_new = []
        # min_y_list = []
        # max_y_list = []
        
        for i, lane in enumerate(lanes_r_org):
            # min_y = np.min(lane[:, 1])
            # max_y = np.max(lane[:, 1])
            # min_y_list.append(min_y.astype(np.float16))
            # max_y_list.append(max_y.astype(np.float16))
            # x, y, z = resample_laneline_in_y(lane, self.cfg.y_samples, mode = 'linear')
            x,y,z = lane[:,0],lane[:,1],lane[:,2]

            lanes_s_org.append(np.vstack((x, y, z)).T)

        # merged_idx = []
        for i, lane in enumerate(lanes_r_new):
            # min_y = np.min(lane[:, 1])
            # max_y = np.max(lane[:, 1])
            # min_y_idx = min_y_list.index(min_y.astype(np.float16))
            # max_y_idx = max_y_list.index(max_y.astype(np.float16))
            # if min_y_idx!= max_y_idx:
            #     merged_idx.append(i)
            x,y,z = lane[:,0],lane[:,1],lane[:,2]
            # x, y, z = resample_laneline_in_y(lane, self.cfg.y_samples, mode = 'linear')

            lanes_s_new.append(np.vstack((x, y, z)).T)
        
        # Set figure
        plt.ioff()
        fig = plt.figure(figsize=(10,5),constrained_layout=True)
        fig.subplots_adjust(hspace=0.3)
        ax1 = fig.add_subplot(231) # 2D projection
        ax1.set_title('ALS')
        ax2 = fig.add_subplot(232, projection='3d') # 3D graph
        ax2_2 = fig.add_subplot(233) # xy graph

        ax2.set_xlim3d(-20, 20)
        ax2.set_ylim3d(0, 110)
        ax2.set_zlim3d(-10, 10)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('z')

        ax2.set_title(f'{len(lanes_r_org)} lanes')
        ax2_2.set_xlim(-20, 20)
        ax2_2.set_ylim(0, 110)
        ax2_2.set_title('xy plane')
        ax2_2.set_xlabel('x')
        ax2_2.set_ylabel('y',rotation=0)

        ax3 = fig.add_subplot(234) # 2D projection
        ax3.set_title('anchor3d')
        ax4 = fig.add_subplot(235, projection='3d') # 3D graph
        ax4_2 = fig.add_subplot(236) # xy graph

        ax4.set_xlim3d(-20, 20)
        ax4.set_ylim3d(0, 110)
        ax4.set_zlim3d(-10, 10)
        ax4.set_xlabel('x')
        ax4.set_ylabel('y')
        ax4.set_zlabel('z')
        ax4.set_title(f'{len(lanes_r_new)} lanes')
        ax4_2.set_xlim(-20, 20)
        ax4_2.set_ylim(0, 110)
        ax4_2.set_xlabel('x')
        ax4_2.set_ylabel('y',rotation=0)


        # Crop/resize image and put it on 2D fig
        img = cv2.warpPerspective(img, self.H_crop, (self.cfg.resize_w, self.cfg.resize_h))
        img = img.astype(np.float32) / 255

        ax1.imshow(img[:, :, [2, 1, 0]],interpolation='bilinear',aspect='auto')
        ax3.imshow(img[:, :, [2, 1, 0]],interpolation='bilinear',aspect='auto')

        # Plot
        legend = list()
        for i, lane in enumerate(lanes_s_org):
            x, y, z = lane[:, 0], lane[:, 1], lane[:, 2]
            color = colors[i%20]
            ax2.plot(x, y, z, lw=2, c=color, alpha=1, label='gt')
            ax2_2.plot(x, y, lw=2, c=color, alpha=1, label='gt')

            # 2D plot
            x_2d, y_2d = projective_transformation(P_gt, x, y, z)
            valid_mask_2d = np.logical_and(np.logical_and(x_2d >= 0, x_2d < self.cfg.resize_w), np.logical_and(y_2d >= 0, y_2d < self.cfg.resize_h))
            x_2d = x_2d[valid_mask_2d]
            y_2d = y_2d[valid_mask_2d]
            ax1.plot(x_2d, y_2d, lw=2, c=color, alpha=1, label='gt')
            legend.append(i)
        ax2.legend(legend)
        # legend = list()
        for i, lane in enumerate(lanes_s_new):
            x, y, z = lane[:, 0], lane[:, 1], lane[:, 2]
            color = colors[i%20]
            ax4.plot(x, y, z, lw=2, c=color, alpha=1, label='gt')
            ax4_2.plot(x, y,lw=2, c=color, alpha=1, label='gt')

            # 2D plot
            x_2d, y_2d = projective_transformation(P_gt, x, y, z)
            valid_mask_2d = np.logical_and(np.logical_and(x_2d >= 0, x_2d < self.cfg.resize_w), np.logical_and(y_2d >= 0, y_2d < self.cfg.resize_h))
            x_2d = x_2d[valid_mask_2d]
            y_2d = y_2d[valid_mask_2d]
            ax3.plot(x_2d, y_2d, lw=2, c=color, alpha=1, label='gt')
            # legend.append(i)
        # ax4.legend(legend)
        # Save visualization result
        mkdir('/'.join(path.split('/')[:-1]))
        fig.savefig(path)
        plt.close(fig)
    def run(self,img,data,path):
        self.visualize_gt_3dlanes(data, img, path)