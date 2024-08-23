# Ver.
import cv2
import torch.nn.functional as F

import math
import time

from scipy.optimize import linear_sum_assignment
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy.interpolate import UnivariateSpline
from sklearn.neighbors import LocalOutlierFactor
from scipy.ndimage import gaussian_filter1d

# from scipy.signal import savgol_filter
from scipy.signal import medfilt

from libs.utils import *
from visualizes.visualize_module import Runner
from visualizes.vis_config import cfg_vis
from visualizes.utils.utils import *
from libs.extrapolation_modes import do_extrapolation

class Preprocessing(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.vis_runner = Runner(cfg_vis)
    
    def get_min_distance(self,lane1, lane2):
        min_dist = float('inf')
        for i in range(len(lane1)//2):
            for j in range(1,len(lane2)//2):
                dist = self.get_distance(lane1[i],lane2[-j])
                min_dist = min(min_dist, dist)
                if dist<0.2:
                    return True
        else: 
            return False
        
    def refine_lane(self,lane,direction):
        if direction == 'inc':
            lane_idx = np.where(self.signs==1,1,self.signs)
            lane_idx = np.where(self.signs==-1,0,self.signs)
            lane_idx = np.append(np.array([1]),lane_idx).astype(np.bool_)
        else:
            lane_idx = np.where(self.signs==-1,1,self.signs)
            lane_idx = np.where(self.signs==1,0,self.signs)
            lane_idx = np.append(np.array([1]),lane_idx).astype(np.bool_)

        lane = lane[lane_idx]

        return lane 
    def get_slope(self,lane):
        lane = np.array(lane)
        
        # x와 y 좌표를 각각 분리
        x = lane[:, 0]
        y = lane[:, 1]
        # x와 y 좌표의 차이 계산
        delta_x = np.diff(x)
        delta_y = np.diff(y)
        
        # 기울기 계산 (delta_y / delta_x)
        slopes = delta_y / delta_x
        # 기울기의 부호 계산
        self.signs = np.sign(slopes).astype(np.int8)

    def calculate_slope_signs(self,lane):
        self.get_slope(lane)
        # check fluctuation then refine
        if np.sum(self.signs ==1)>len(self.signs)*0.6:
            lane = self.refine_lane(lane,'inc')
            for iter in range(10):
                self.get_slope(lane)
                lane = self.refine_lane(lane,'inc')
            return lane
        elif np.sum(self.signs ==-1)>len(self.signs)*0.6:
            lane = self.refine_lane(lane,'dec')
            for iter in range(10):
                self.get_slope(lane)
                lane = self.refine_lane(lane,'dec')
            return lane
        else:
            return None
    
    def check_gradient(self,lane):
        # x값 최대 최소가 비슷하면 수직선이다.
        if max(lane[:,0])-min(lane[:,0])<3:
            # direction = 'vertical'
            return lane
        else:
            # 좌표들 사이 기울기로 투표해서 증가/감소 판정
            # + check fluctuation
            return self.calculate_slope_signs(lane)
        
    def short_lane(self,lane):
        # by number of points
        if len(lane)<5:
            return True
        min_y, max_y = min(lane[:,1]),max(lane[:,1])
        min_x, max_x = min(lane[:,0]),max(lane[:,0])
        # by physical length
        if max_y - min_y < 10 and max_x - min_x < 3:
            return True
        return False
    
    def check_duplicates(self,lanes):
        pts_map = np.zeros(((self.cfg.max_y-self.cfg.min_y),(self.cfg.max_x-self.cfg.min_x)),dtype=np.bool_)
        for lane_idx,lane in enumerate(lanes):
            del_idx = []
            for idx,pt in enumerate(lane):
                x,y = pt[0],pt[1]
                x = int((x+self.cfg.max_x)*2)
                y = int(y-self.cfg.min_y)
                try:
                    if pts_map[y][x]:
                        del_idx.append(idx)
                    else:
                        pts_map[y][x] = True
                except:
                    0
            lanes[lane_idx] = np.delete(lane,del_idx,axis=0)
        return lanes

    def convert_lane_y(self,lane):
        lane = lane[lane[:, 1] < self.cfg.max_y+1]
        lane = lane[lane[:, 1] > self.cfg.min_y-1]
        lane[:, 1] = np.round(lane[:, 1])
        unique_idx = np.sort(np.unique(lane[:, 1], return_index=True)[1])
        lane = lane[unique_idx]
        return lane
    
    def crop_lane(self,lane):
        lane = lane[lane[:, 1] <= self.cfg.max_y]
        lane = lane[lane[:, 1] >= self.cfg.min_y]
        lane = lane[lane[:, 0] <= self.cfg.max_x]
        lane = lane[lane[:, 0] >= self.cfg.min_x]
        return lane

    def lane_smoothing(self,lane):
        lof = LocalOutlierFactor(n_neighbors=5)
        labels = lof.fit_predict(lane)

        # 노이즈 포인트 식별
        # noise_points = lane[labels == -1]
        lane = lane[labels != -1]
        return lane
    
    def z_smoothing(self,lane):
        z = lane[:,2]
        new_z = medfilt(z, kernel_size=5)
        new_z = gaussian_filter1d(new_z,sigma=3)
        new_x = gaussian_filter1d(lane[:,0],sigma=1)
        lane[:,0] = new_x
        # y= lane[:,1]
        # input_lane = lane[::max(len(lane)//3,1),:]
        # _,_,new_z = do_extrapolation(input_lane,y,mode = 'b-spline')
        lane[:,2] = new_z
        return lane

    def interpolation(self,lane):
        sampling_pts = np.array(range(lane[:,1][0].astype(np.int8),lane[:,1][-1].astype(np.int8)))
        x,y,z = do_extrapolation(lane,sampling_pts,mode = 'cubic_spline')
        return np.concatenate([x.reshape(-1,1),y.reshape(-1,1),z.reshape(-1,1)],axis = 1)
    
    def visualize_data(self, img_name,data):
        if self.cfg.datalist_mode =='example':
            img_ori = cv2.imread(f'{self.cfg.dir["dataset"]}/images/validation/{img_name}.jpg')
        else:
            img_ori = cv2.imread(f'{self.cfg.dir["dataset"]}/images/{self.cfg.datalist_mode}/{img_name}.jpg')
        cur_data = {
            'lane3d' :{
                'org_lane': data['lane3d']['org_lane'],
                'new_lane': data['lane3d']['new_lane']
            },
            'extrinsic': np.array(data['extrinsic']),
            'intrinsic': np.array(data['intrinsic']),
        }
        dir_name = f'{self.cfg.dir["out"]}/display/'
        self.vis_runner.run(img=img_ori,data = cur_data, path=dir_name+img_name)

    # filter
    def filter(self,datalist):
        print(f'construct lane matrix')
        # 같은 영상에 있는 frame들에 대해 수행
        for i in range(len(datalist)):
            img_name = datalist[i]
            dir_name = img_name.split('/')[0]
            # img_detailed_name = img_name.split('/')[1]
            # GET LANE DATA
            data = load_pickle(f'{self.cfg.dir["pre0"]}/{img_name}')
            print(f'{i} ==> load {img_name}')
            lanes = data['lane3d']['new_lane'].copy()
            data['lane3d']['org_lane'] = data['lane3d']['new_lane'].copy()

            # Do filtering
            del_idx = []
            for idx, lane in enumerate(lanes):
                lane = self.lane_smoothing(lane)

                lane = self.convert_lane_y(lane)
                # 점 개수, 길이 짧으면 제거
                if self.short_lane(lane):
                    del_idx.append(idx)
                    continue
                # 단조 증가/감소하게끔
                lane = self.check_gradient(lane)

                if lane is None:
                    del_idx.append(idx)
                    continue
                # x쪽도 원하는 범위 내 죄표만 남게
                lane = self.crop_lane(lane)
                if self.short_lane(lane):
                    del_idx.append(idx)
                    continue
                lanes[idx] = lane
                # smooth z
                lane = self.z_smoothing(lane)
                lane = self.interpolation(lane)
            for idx in range(len(del_idx)-1,-1,-1):
                del lanes[del_idx[idx]]

            # 겹치는 포인트가 하나에만 속하도록
            # lanes = self.check_duplicates(lanes.copy())
            # del_idx = []
            # for idx, lane in enumerate(lanes):
            #     # 점 개수, 길이 짧으면 제거
            #     if self.short_lane(lane):
            #         del_idx.append(idx)
            #         continue
            # for idx in range(len(del_idx)-1,-1,-1):
            #     del lanes[del_idx[idx]]

            # save filtered lane
            if len(data['lane3d']['new_lane']) == len(lanes):
                print('same')
            else:
                print(len(data['lane3d']['new_lane']),'->',len(lanes),'merged')
            data['lane3d']['new_lane'] = lanes.copy()     

            if self.cfg.save_pickle == True:
                path = f'{self.cfg.dir["out"]}/pickle/{img_name}'
                save_pickle(path=path, data=data)
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # SAVE IMG
            if self.cfg.display_all == True :
                self.visualize_data(img_name, data)
        print("Finished Filtering!")

    def run_for_videos(self):
        start_time = time.time()
        print('total_frames: ',len(self.datalist))
        if self.cfg.multiprocess:
            import multiprocessing
            start_time = time.time()
            procs = []
            quarter = len(self.datalist)//4
            for i in range(4):
                if i!=3:
                    datalist = self.datalist[quarter*i:quarter*(i+1)]
                else:
                    datalist = self.datalist[quarter*i:]
                p = multiprocessing.Process(target=self.filter,args = (datalist,))
                p.start()
                procs.append(p)
            for p in procs:
                p.join()
        else:
            self.filter(self.datalist)
        print("time: ", time.time() - start_time)

    def init(self):
        self.datalist = list()

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

    def run(self):
        print('start')
        self.init()
        self.generate_video_datalist()
        datalist_video = load_pickle(f'{self.cfg.dir["out"]}/pickle/datalist_video')
        datalist_scene = list(datalist_video)
        self.datalist = list()
        # datalist_scene = [
        #     'segment-17065833287841703_2980_000_3000_000_with_camera_labels',
        #     'segment-89454214745557131_3160_000_3180_000_with_camera_labels',
        #     'segment-271338158136329280_2541_070_2561_070_with_camera_labels',
        #     'segment-346889320598157350_798_187_818_187_with_camera_labels',
        #     'segment-967082162553397800_5102_900_5122_900_with_camera_labels',
        #     'segment-1071392229495085036_1844_790_1864_790_with_camera_labels',
        #     'segment-1457696187335927618_595_027_615_027_with_camera_labels'
        #     ]
        # datalist_scene= ['segment-12358364923781697038_2232_990_2252_990_with_camera_labels']
        # datalist_scene= ['segment-89454214745557131_3160_000_3180_000_with_camera_labels']
        # datalist_scene=['segment-346889320598157350_798_187_818_187_with_camera_labels']


        for i in range(len(datalist_scene)):
            self.video_idx = i
            self.video_name = datalist_scene[i]
            self.datalist_video = datalist_video[self.video_name]
            self.datalist += self.datalist_video
        self.run_for_videos()