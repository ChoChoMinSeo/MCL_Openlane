import cv2
import numpy as np
import matplotlib.pyplot as plt
from ui_tools.utils.vis_utils import *
from mpl_toolkits.mplot3d import Axes3D

class Windows(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.H_crop = homography_crop_resize([self.cfg.org_h, self.cfg.org_w], self.cfg.crop_y, [self.cfg.resize_h, self.cfg.resize_w])
    def blank_window(self,win_name):
        blank = np.ones((300,300,1))*255
        cv2.imshow(win_name,blank)

    def progress_window(self,tot_scene,scene_idx,tot_img,img_idx):
        blank = np.ones((200,300,1))*255
        cv2.putText(blank,f'scene: {scene_idx}/{tot_scene}',(0,50),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0,0,0),thickness=2)
        cv2.putText(blank,f'image: {img_idx}/{tot_img}',(0,150),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0,0,0),thickness=2)
        cv2.imshow('progress',blank)
        cv2.setTrackbarPos('image','progress',img_idx-1)

    def draw_plane(self, line_idx, datas,xy=True):
        f = plt.figure(figsize=(5,5),constrained_layout=True)
        ax = plt.gca()
        ax.set_ylim(0, 110)
        ax.set_ylabel('y')
        if xy == True:
            ax.set_xlim(-20,20)
            win_name = 'xy plane'
            ax.set_xlabel('x')
            x = 0
        else:
            ax.set_xlim(-0.5, 0.5)
            win_name = 'yz plane'
            ax.set_xlabel('z')
            x = 2
        for i in range(len(datas)):
            if i == line_idx:
                color = (1,0,0)
            else:
                color = (0,0,0)
            cur_lane = datas[i]
            plt.plot(cur_lane[:,x],cur_lane[:,1],lw=2, c=color, alpha=1)
        f_arr = self.figure_to_array(f)
        f_arr = cv2.resize(f_arr,(500,500))
        plt.close()
        cv2.imshow(win_name,f_arr)

    def draw_3d(self,line_idx, datas):
        f = plt.figure(figsize=(3,3))
        ax = f.add_subplot(projection = '3d')
        ax.set_xlim3d(-20, 20)
        ax.set_ylim3d(0, 110)
        ax.set_zlim3d(-0.5, 0.5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        # Crop/resize image and put it on 2D fig
        for i in range(len(datas)):
            cur_lane = datas[i]
            x,y,z = cur_lane[:,0],cur_lane[:,1],cur_lane[:,2]
            if i == line_idx:
                color = (1,0,0)
            else:
                color = (0,0,0)
            ax.plot(x, y, z, lw=2, c=color, alpha=1, label='gt')
        f_arr = self.figure_to_array(f)
        plt.close()
        cv2.imshow('3d plane',f_arr)

    def draw_img(self,img,line_idx, datas,P_gt):
        f = plt.figure(constrained_layout=True)
        plt.axis('off')
        ax = f.gca()
        ax.imshow(img[:, :, [2, 1, 0]],interpolation='bilinear',aspect='auto')
        img_coords = []
        for i in range(len(datas)):
            cur_lane = datas[i]
            x,y,z = cur_lane[:,0],cur_lane[:,1],cur_lane[:,2]
            x_2d, y_2d = projective_transformation(P_gt, x, y, z)
            valid_mask_2d = np.logical_and(np.logical_and(x_2d >= 0, x_2d < self.cfg.resize_w), np.logical_and(y_2d >= 0, y_2d < self.cfg.resize_h))
            x_2d = x_2d[valid_mask_2d]
            y_2d = y_2d[valid_mask_2d]
            if i == line_idx:
                color = (1,0,0)
            else:
                color = (0,0,0)
            ax.plot(x_2d, y_2d, lw=2, c=color, alpha=1, label='gt')
            img_coords.append(np.concatenate([x_2d.reshape(-1,1),y_2d.reshape(-1,1)],axis=1))
        f_arr = self.figure_to_array(f)
        f_arr = cv2.resize(f_arr,(640,480))
        plt.close()
        cv2.imshow('image',f_arr)
        return img_coords,f_arr.shape[:2]


    def figure_to_array(self,fig):
        fig.canvas.draw()
        return np.array(fig.canvas.renderer._renderer)