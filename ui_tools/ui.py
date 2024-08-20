import cv2
import os
from ui_tools.utils.utils import *
from ui_tools.utils.vis_utils import *
from ui_tools.window import Windows
from copy import deepcopy
from scipy.interpolate import interp1d

class Editor(object):
    def __init__(self,cfg):
        self.cfg = cfg
        self.H_crop = homography_crop_resize([self.cfg.org_h, self.cfg.org_w], self.cfg.crop_y, [self.cfg.resize_h, self.cfg.resize_w])
        self.plane_flag = False
        self.modified = False
        self.move = 0

    def interpolate(self,axis):
        self.datas[self.cur_img_idx][self.cur_lane_idx][self.y_idx,axis]
        y = self.datas[self.cur_img_idx][self.cur_lane_idx][:,1]
        y_idx = int(self.y_idx)
        lower_bound = y_idx-8
        upper_bound = y_idx+8
        temp_lane = np.zeros((0,3))
        if lower_bound>0:
            temp_lane = np.concatenate([temp_lane, self.datas[self.cur_img_idx][self.cur_lane_idx][:lower_bound,:]],axis=0)
        temp_lane = np.concatenate([temp_lane, self.datas[self.cur_img_idx][self.cur_lane_idx][y_idx:y_idx+1,:]],axis=0)
        if upper_bound<len(self.datas[self.cur_img_idx][self.cur_lane_idx]):
            temp_lane = np.concatenate([temp_lane, self.datas[self.cur_img_idx][self.cur_lane_idx][upper_bound:,:]],axis=0)
        f_x = interp1d(temp_lane[:,1],temp_lane[:,axis],fill_value='extrapolate')
        new_x = f_x(y)
        self.datas[self.cur_img_idx][self.cur_lane_idx][:,axis] = new_x

    def on_click_img(self,win_name):
        def click_img(event,x,y,flags,param):
            y = y*self.cfg.resize_h/self.img_h
            x = x*self.cfg.resize_w/self.img_w
            if event == cv2.EVENT_LBUTTONDOWN:
                min_dist = float('inf')
                min_idx = -1
                for i in range(len(self.img_coords)):
                    temp_lane = self.img_coords[i].copy()
                    temp_lane[:,1] = abs(temp_lane[:,1] - y)
                    y_idx = np.argmin(temp_lane[:,1])
                    dist = abs(temp_lane[y_idx,0]-x)
                    if dist < min_dist:
                        min_dist = dist
                        min_idx = i
                if min_dist<30:
                    self.cur_lane_idx = min_idx
                    self.window.draw_plane(self.cur_lane_idx, datas = self.datas[self.cur_img_idx], xy = True)
                    self.window.draw_plane(self.cur_lane_idx, datas = self.datas[self.cur_img_idx], xy = False)
                    self.window.draw_3d(line_idx=self.cur_lane_idx, datas = self.datas[self.cur_img_idx])
                    self.img_coords,(self.img_h,self.img_w) = self.window.draw_img(img = self.imgs[self.cur_img_idx],line_idx=self.cur_lane_idx, datas = self.datas[self.cur_img_idx], P_gt=self.P_gt)
                else:
                    print('lane not found!')
        cv2.setMouseCallback(win_name, click_img)

    def on_click_2d(self,win_name):
        if win_name == 'xy plane':
            origin = 276
            pt = 62
            step = 20
            xy = True
            axis = 0
            thr = 2
        else:
            origin = 280
            pt = 61
            step = 0.5
            xy = False
            axis = 2
            thr = 0.08
        def click_2d(event,x,y,flags,param):
            x = (x-origin)*step/(origin-pt)
            y = (455-y)*100/409
            y = int(y)
            if event == cv2.EVENT_LBUTTONDOWN:
                self.modified = True
                min_dist = float('inf')
                min_idx = -1
                for i in range(len(self.datas[self.cur_img_idx])):
                    temp_lane = self.datas[self.cur_img_idx][i].copy()
                    temp_lane[:,1] = abs(temp_lane[:,1] - y)
                    y_idx = np.argmin(temp_lane[:,1])
                    dist = abs(temp_lane[y_idx,axis]-x)
                    if dist < min_dist:
                        min_dist = dist
                        min_idx = i
                if min_dist<thr:
                    self.cur_lane_idx = min_idx
                    self.window.draw_plane(self.cur_lane_idx, datas = self.datas[self.cur_img_idx], xy = True)
                    self.window.draw_plane(self.cur_lane_idx, datas = self.datas[self.cur_img_idx], xy = False)
                    self.window.draw_3d(line_idx=self.cur_lane_idx, datas = self.datas[self.cur_img_idx])
                    self.img_coords,(self.img_h,self.img_w) = self.window.draw_img(img = self.imgs[self.cur_img_idx],line_idx=self.cur_lane_idx, datas = self.datas[self.cur_img_idx], P_gt=self.P_gt)
                else:
                    print('lane not found!')
                    return 0
                self.y_idx = np.where(self.datas[self.cur_img_idx][self.cur_lane_idx][:,1]==y)[0]
                if len(self.y_idx)==0:
                    print('y not found')
                    return 0
                self.plane_flag = True
                self.x0 = x
            elif self.plane_flag and event == cv2.EVENT_MOUSEMOVE:
                self.move = x-self.x0
                datas = deepcopy(self.datas[self.cur_img_idx])
                datas[self.cur_lane_idx][self.y_idx,axis]+=self.move
                self.window.draw_plane(self.cur_lane_idx, datas = datas, xy = xy)

            elif self.plane_flag and event == cv2.EVENT_LBUTTONUP:
                self.plane_flag = False
                self.datas[self.cur_img_idx][self.cur_lane_idx][self.y_idx,axis]+=self.move
                # interpolate near points
                self.interpolate(axis)

                self.window.draw_plane(self.cur_lane_idx, datas = self.datas[self.cur_img_idx], xy = xy)
                self.window.draw_3d(line_idx=self.cur_lane_idx, datas = self.datas[self.cur_img_idx])
                self.img_coords,(self.img_h,self.img_w) = self.window.draw_img(img = self.imgs[self.cur_img_idx],line_idx=self.cur_lane_idx, datas = self.datas[self.cur_img_idx], P_gt=self.P_gt)
                self.move = 0
        cv2.setMouseCallback(win_name, click_2d)

    def on_click_progress(self):
        def click_pr(event,x,y,flags,param):
            if event == cv2.EVENT_LBUTTONDOWN:
                print('# scene 입력: ')
                try:
                    pos = int(input())-1
                except:
                    print('취소')
                    return 0
                self.scene_trackbar_on_change(pos)
                cv2.setTrackbarPos('scene','progress',pos)
        cv2.setMouseCallback('progress',click_pr)

    def scene_trackbar_on_change(self,pos):
        self.save_data()
        self.cur_scene_idx = pos
        self.cur_img_idx = 0
        self.cur_lane_idx = 0
        self.load_img()
        self.load_data()
        self.window.draw_plane(self.cur_lane_idx, datas = self.datas[self.cur_img_idx], xy = True)
        self.window.draw_plane(self.cur_lane_idx, datas = self.datas[self.cur_img_idx], xy = False)
        self.window.draw_3d(line_idx=self.cur_lane_idx, datas = self.datas[self.cur_img_idx])
        self.img_coords,(self.img_h,self.img_w) = self.window.draw_img(img = self.imgs[self.cur_img_idx],line_idx=self.cur_lane_idx, datas = self.datas[self.cur_img_idx], P_gt=self.P_gt)
        self.window.progress_window(len(self.datalist_scene),self.cur_scene_idx+1,len(self.datas),self.cur_img_idx+1)

    def img_trackbar_on_change(self,pos):
        self.save_data()
        self.cur_img_idx = pos
        self.cur_lane_idx = 0
        self.window.draw_plane(self.cur_lane_idx, datas = self.datas[self.cur_img_idx], xy = True)
        self.window.draw_plane(self.cur_lane_idx, datas = self.datas[self.cur_img_idx], xy = False)
        self.window.draw_3d(line_idx=self.cur_lane_idx, datas = self.datas[self.cur_img_idx])
        self.img_coords,(self.img_h,self.img_w) = self.window.draw_img(img = self.imgs[self.cur_img_idx],line_idx=self.cur_lane_idx, datas = self.datas[self.cur_img_idx], P_gt=self.P_gt)
        self.window.progress_window(len(self.datalist_scene),self.cur_scene_idx+1,len(self.datas),self.cur_img_idx+1)

    def launch_ui(self):
        self.window = Windows(self.cfg)
        self.window.blank_window('image')
        self.window.blank_window('xy plane')
        self.window.blank_window('yz plane')
        self.window.blank_window('3d plane')
        self.window.blank_window('progress')

        self.on_click_img('image')
        self.on_click_2d('xy plane')
        self.on_click_2d('yz plane')
        self.on_click_progress()
        self.cur_lane_idx = 0
        self.window.progress_window(len(self.datalist_scene),self.cur_scene_idx+1,len(self.datas),self.cur_img_idx+1)
        self.load_img()
        self.window.draw_plane(self.cur_lane_idx, datas = self.datas[self.cur_img_idx], xy = True)
        self.window.draw_plane(self.cur_lane_idx, datas = self.datas[self.cur_img_idx], xy = False)
        self.window.draw_3d(line_idx=self.cur_lane_idx, datas = self.datas[self.cur_img_idx])
        self.img_coords,(self.img_h,self.img_w) = self.window.draw_img(img = self.imgs[self.cur_img_idx],line_idx=self.cur_lane_idx, datas = self.datas[self.cur_img_idx], P_gt=self.P_gt)
        cv2.createTrackbar('scene','progress',self.cur_scene_idx,len(self.datalist_scene)-1,self.scene_trackbar_on_change)
        cv2.createTrackbar('image','progress',self.cur_img_idx,len(self.datas)-1,self.img_trackbar_on_change)

    def init(self):
        self.datalist = list()
        self.datalist_error = list()

    def generate_video_datalist(self):
        datalist_org = load_pickle(f'{self.cfg.dir["pre0"]}/datalist')
        datalist_out = dict()
        for i in range(len(datalist_org)):
            name = datalist_org[i]
            dirname = os.path.dirname(name)
            if dirname not in datalist_out.keys():
                datalist_out[dirname] = list()
            datalist_out[dirname].append(name)
            print(f'{i} ==> {name} done')
        save_pickle(f'{self.cfg.dir["out"]}/pickle/datalist_video', datalist_out)
        save_pickle(f'{self.cfg.dir["out"]}/pickle/datalist', datalist_out)

    def load_data(self):
        self.datas = []
        scene_name = self.datalist_scene[self.cur_scene_idx]
        for img_name in (self.datalist_video[scene_name]):
            data = load_pickle(f'{self.cfg.dir["data_path"]}/{img_name}')
            self.datas.append(data['lane3d']['new_lane'])
        self.extrinsic = data['extrinsic']
        self.intrinsic = data['intrinsic']
        P_g2im = projection_g2im_extrinsic(self.extrinsic, self.intrinsic)
        self.P_gt = np.matmul(self.H_crop, P_g2im)

    def save_data(self):
        if self.modified:
            scene_name = self.datalist_scene[self.cur_scene_idx]
            img_name = self.datalist_video[scene_name][self.cur_img_idx]
            path = f'{self.cfg.dir["out"]}/pickle/results/{img_name}'
            out = {
                'lane3d':{
                    'new_lane':self.datas[self.cur_img_idx]
                },
                'extrinsic': self.extrinsic,
                'intrinsic': self.intrinsic
            }
            save_pickle(path, out)

    def load_img(self):
        self.imgs = []
        for i in range(len(self.datalist_video[self.datalist_scene[self.cur_scene_idx]])):
            img_name = self.datalist_video[self.datalist_scene[self.cur_scene_idx]][i]
            # SAVE IMG
            if self.cfg.datalist_mode =='example':
                img = cv2.imread(f'{self.cfg.dir["dataset"]}/images/validation/{img_name}.jpg')
            else:
                img = cv2.imread(f'{self.cfg.dir["dataset"]}/images/{self.cfg.datalist_mode}/{img_name}.jpg')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.warpPerspective(img, self.H_crop, (self.cfg.resize_w, self.cfg.resize_h))
            img = img.astype(np.float32) / 255
            self.imgs.append(img)

    def run(self):
        self.init()
        self.generate_video_datalist()
        self.datalist_video = load_pickle(f'{self.cfg.dir["out"]}/pickle/datalist_video')
        self.datalist_scene = list(self.datalist_video)

        self.cur_scene_idx = 0
        self.cur_img_idx = 0
        # try:
        self.load_data()
        self.launch_ui()
        try:
            while True:
                key = cv2.waitKeyEx(0)
                if key == 27:
                    self.save_data()
                    break
                elif key == 65361:
                    # l arrow
                    self.save_data()
                    self.cur_img_idx = max(self.cur_img_idx-1,0)
                    self.cur_lane_idx = 0
                    self.window.draw_plane(self.cur_lane_idx, datas = self.datas[self.cur_img_idx], xy = True)
                    self.window.draw_plane(self.cur_lane_idx, datas = self.datas[self.cur_img_idx], xy = False)
                    self.window.draw_3d(line_idx=self.cur_lane_idx, datas = self.datas[self.cur_img_idx])
                    self.img_coords,(self.img_h,self.img_w) = self.window.draw_img(img = self.imgs[self.cur_img_idx],line_idx=self.cur_lane_idx, datas = self.datas[self.cur_img_idx], P_gt=self.P_gt)
                    self.window.progress_window(len(self.datalist_scene),self.cur_scene_idx+1,len(self.datas),self.cur_img_idx+1)
                elif key == 65363:
                    # r arrow
                    self.save_data()
                    self.cur_img_idx = min(self.cur_img_idx+1,len(self.datalist_video[self.datalist_scene[self.cur_scene_idx]])-1)
                    self.cur_lane_idx = 0
                    self.window.draw_plane(self.cur_lane_idx, datas = self.datas[self.cur_img_idx], xy = True)
                    self.window.draw_plane(self.cur_lane_idx, datas = self.datas[self.cur_img_idx], xy = False)
                    self.window.draw_3d(line_idx=self.cur_lane_idx, datas = self.datas[self.cur_img_idx])
                    self.img_coords,(self.img_h,self.img_w) = self.window.draw_img(img = self.imgs[self.cur_img_idx],line_idx=self.cur_lane_idx, datas = self.datas[self.cur_img_idx], P_gt=self.P_gt)
                    self.window.progress_window(len(self.datalist_scene),self.cur_scene_idx+1,len(self.datas),self.cur_img_idx+1)
                elif key == 65362:
                    # up arrow
                    self.save_data()
                    self.cur_scene_idx = min(self.cur_scene_idx+1,len(self.datalist_scene)-1)
                    self.cur_img_idx = 0
                    self.cur_lane_idx = 0
                    self.load_img()
                    self.load_data()
                    self.window.draw_plane(self.cur_lane_idx, datas = self.datas[self.cur_img_idx], xy = True)
                    self.window.draw_plane(self.cur_lane_idx, datas = self.datas[self.cur_img_idx], xy = False)
                    self.window.draw_3d(line_idx=self.cur_lane_idx, datas = self.datas[self.cur_img_idx])
                    self.img_coords,(self.img_h,self.img_w) = self.window.draw_img(img = self.imgs[self.cur_img_idx],line_idx=self.cur_lane_idx, datas = self.datas[self.cur_img_idx], P_gt=self.P_gt)
                    self.window.progress_window(len(self.datalist_scene),self.cur_scene_idx+1,len(self.datas),self.cur_img_idx+1)
                elif key == 65364:
                    # down arrow
                    self.save_data()
                    self.cur_scene_idx = max(self.cur_scene_idx-1,0)
                    self.cur_img_idx = 0
                    self.cur_lane_idx = 0
                    self.load_img()
                    self.load_data()
                    self.window.draw_plane(self.cur_lane_idx, datas = self.datas[self.cur_img_idx], xy = True)
                    self.window.draw_plane(self.cur_lane_idx, datas = self.datas[self.cur_img_idx], xy = False)
                    self.window.draw_3d(line_idx=self.cur_lane_idx, datas = self.datas[self.cur_img_idx])
                    self.img_coords,(self.img_h,self.img_w) = self.window.draw_img(img = self.imgs[self.cur_img_idx],line_idx=self.cur_lane_idx, datas = self.datas[self.cur_img_idx], P_gt=self.P_gt)
                    self.window.progress_window(len(self.datalist_scene),self.cur_scene_idx+1,len(self.datas),self.cur_img_idx+1)
                elif key == 65535:
                    # delete
                    del self.datas[self.cur_img_idx][self.cur_lane_idx]
                    self.window.draw_plane(self.cur_lane_idx, datas = self.datas[self.cur_img_idx], xy = True)
                    self.window.draw_plane(self.cur_lane_idx, datas = self.datas[self.cur_img_idx], xy = False)
                    self.window.draw_3d(line_idx=self.cur_lane_idx, datas = self.datas[self.cur_img_idx])
                    self.img_coords,(self.img_h,self.img_w) = self.window.draw_img(img = self.imgs[self.cur_img_idx],line_idx=self.cur_lane_idx, datas = self.datas[self.cur_img_idx], P_gt=self.P_gt)
                elif key == 65379:
                    # insert
                    print('l')
                else:
                    print(key)
        except:
            self.save_data()
            cv2.destroyAllWindows()
        # save
        cv2.destroyAllWindows()