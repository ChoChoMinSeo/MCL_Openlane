# Ver.
import cv2
import time

from libs.utils import *
from visualizes.visualize_module import Runner
from visualizes.vis_config import cfg_vis
from visualizes.utils.utils import *
from tqdm import tqdm

class Preprocessing(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.vis_runner = Runner(cfg_vis)

    def short_lane(self,lanes):
        checklist = [False]*len(lanes)
        for idx,lane in enumerate(lanes):
            if lane[-1,1]-lane[0,1]<30:
                checklist[idx] = True
        return checklist
    
    def complex_lane(self,lanes):
        checklist = [False]*len(lanes)
        for idx,lane in enumerate(lanes):
            x,y = lane[:,0],lane[:,1]
            dx_dy = np.gradient(x, y)
            d2x_dy2 = np.gradient(dx_dy, y)
            if max(abs(d2x_dy2))>0.5:
                checklist[idx] = True
        return checklist
    
    def get_angle(self,a, b, c):
        # a[0]/= self.x_std
        # a[1]/= self.y_std
        # b[0]/= self.x_std
        # b[1]/= self.y_std
        # c[0]/= self.x_std
        # c[1]/= self.y_std
        a = a[:2]
        b = b[:2]
        c = c[:2]

        ba = a-b
        bc = c-b
        dot_product = np.dot(ba,bc)
        cos_theta = dot_product/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-8)
        angle = np.arccos(cos_theta)
        return angle
    
    def complex_lane_angle(self,lanes):
        checklist = [False]*len(lanes)
        for idx,lane in enumerate(lanes):
            if len(lane)<30:
                checklist[idx] = False
                continue
            x,y = lane[:,0],lane[:,1]
            self.x_std = x.std()
            self.y_std = y.std()
            angle_arr = list()
            for pt_idx in range(2,len(lane)-2):
                cur_angle = self.get_angle(lane[pt_idx-2].copy(),lane[pt_idx].copy(),lane[pt_idx+2].copy())
                if cur_angle < 2.5:
                    checklist[idx] = True
                    break
            #     angle_arr.append(cur_angle)
            # angle_arr = np.array(angle_arr)
            # da_dy = np.gradient(angle_arr, y[3:-3:3])
            # d2a_dy2 = np.gradient(da_dy, y[3:-3:3])
            # if max(abs(d2a_dy2))>0.4:
            #     checklist[idx] = True
        return checklist
    
    def select_lane(self,datalist):
        print(f'construct lane matrix')
        # 같은 영상에 있는 frame들에 대해 수행
        for i in range(len(datalist)):
            img_name = datalist[i]
            dir_name = img_name.split('/')[0]
            # img_detailed_name = img_name.split('/')[1]
            # from merged data
            # data = load_pickle(f'{self.cfg.dir["pre5"]}/{img_name}')
            # from 'parallel' data
            # data = load_pickle(f'{self.cfg.dir["pre6_interpolate"]}/{img_name}')
            data = load_pickle(f'{self.cfg.dir["pre6_parallel"]}/{img_name}')
            lanes = data['lane3d']['new_lane'].copy()
            data['lane3d']['org_lane'] = data['lane3d']['new_lane'].copy()
            print(f'{i} ==> load {img_name}')
            data['checklist'] = dict()
            # start job here
            data['checklist']['short'] = self.short_lane(lanes)
            data['checklist']['complex'] = self.complex_lane_angle(lanes)
            new_lanes = []
            for i in range(len(lanes)):
                if not data['checklist']['short'][i] and not data['checklist']['complex'][i]:
                    new_lanes.append(lanes[i].copy())
            # statistics
            self.statistics[dir_name]['total']+=len(lanes)
            self.statistics[dir_name]['short']+=np.array(data['checklist']['short']).sum()
            self.statistics[dir_name]['complex']+=np.array(data['checklist']['complex']).sum()
            remain_arr = np.array([a or b for a, b in zip(data['checklist']['short'], data['checklist']['complex'])])
            self.statistics[dir_name]['remain'] +=(len(lanes)-remain_arr.sum()
)
            if np.array(data['checklist']['complex']).sum() !=0:
                self.vis_datalist.append(img_name)
            # UPDATE
            # update dataset with merged data
            data['lane3d']['new_lane'] = new_lanes.copy()
            if self.cfg.save_pickle == True:
                path = f'{self.cfg.dir["out"]}/pickle/{img_name}'
                save_pickle(path=path, data=data)
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def statistics_sum(self):
        for video_name in self.datalist_scene:
            self.statistics['total']+=self.statistics[video_name]['total']
            self.statistics['short']+=self.statistics[video_name]['short']
            self.statistics['complex']+=self.statistics[video_name]['complex']
            self.statistics['remain']+=self.statistics[video_name]['remain']

        save_dict_to_txt(self.statistics,f'{self.cfg.dir["out"]}/statistics.txt')
        print("Finished Selection!")

    def error_lane(self,org_data,remain_idx):
        remain_lane = list()
        for idx in range(len(remain_idx)):
            if remain_idx[idx] == True:
                remain_lane.append(org_data['lane3d']['new_lane'][idx])
        org_data['lane3d']['new_lane'] = remain_lane
        return org_data
      
    def visualize_datalist(self, datalist):
        for i in tqdm(range(len(datalist))):
            img_name = datalist[i]
            dir_name = img_name.split('/')[0]
            data = load_pickle(f'{self.cfg.dir["out"]}/pickle/{img_name}')
            # SAVE IMG
            if self.cfg.datalist_mode =='example':
                img_ori = cv2.imread(f'{self.cfg.dir["dataset"]}/images/validation/{img_name}.jpg')
            else:
                img_ori = cv2.imread(f'{self.cfg.dir["dataset"]}/images/{self.cfg.datalist_mode}/{img_name}.jpg')
            checklist = data['checklist']
            cur_data = {
                'lane3d' :{
                    'org_lane': data['lane3d']['org_lane'],
                    'new_lane': data['lane3d']['new_lane']
                },
                'extrinsic': np.array(data['extrinsic']),
                'intrinsic': np.array(data['intrinsic']),
            }
            # short lane
            # if np.array(checklist['short']).sum()!=0:
            #     dir_name = f'{self.cfg.dir["out"]}/short/'
            #     self.vis_runner.run(img=img_ori,data = self.error_lane(cur_data.copy(),checklist['short']), path=dir_name+img_name)
            if np.array(checklist['complex']).sum()!=0:
                dir_name = f'{self.cfg.dir["out"]}/complex/'
                # self.vis_runner.run(img=img_ori,data = self.error_lane(cur_data.copy(),checklist['complex']), path=dir_name+img_name)
                self.vis_runner.run(img=img_ori,data = cur_data, path=dir_name+img_name)


    def run_for_videos(self):
        print('total_frames: ',len(self.datalist))
        start_time = time.time()
        # select good lanes
        self.select_lane(self.datalist)
        # generate statistics txt
        self.statistics_sum()
        print("time: ", time.time() - start_time)

        # visualize
        if self.cfg.display_all == True:
            if self.cfg.multiprocess:
                import multiprocessing
                import parmap
                quarter = len(self.vis_datalist)//self.cfg.num_workers
                input_datalist = []
                for i in range(self.cfg.num_workers):
                    if i!=self.cfg.num_workers-1:
                        temp = self.vis_datalist[quarter*i:quarter*(i+1)]
                    else:
                        temp = self.vis_datalist[quarter*i:]
                    input_datalist.append(temp)
                result = parmap.map(self.visualize_datalist, input_datalist, pm_pbar=True, pm_processes=self.cfg.num_workers)
                
            else:
                self.visualize_datalist(self.vis_datalist)
        
        print("time: ", time.time() - start_time)

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

    def run(self):
        print('start')
        self.init()
        self.generate_video_datalist()
        datalist_video = load_pickle(f'{self.cfg.dir["out"]}/pickle/datalist_video')
        self.datalist_scene = list(datalist_video)
        self.datalist = list()
        self.vis_datalist = list()
        # datalist_scene = [
        #     # 'segment-17065833287841703_2980_000_3000_000_with_camera_labels',
        #     'segment-89454214745557131_3160_000_3180_000_with_camera_labels',
        #     # 'segment-271338158136329280_2541_070_2561_070_with_camera_labels',
        #     # 'segment-346889320598157350_798_187_818_187_with_camera_labels',
        #     # 'segment-967082162553397800_5102_900_5122_900_with_camera_labels',
        #     # 'segment-1071392229495085036_1844_790_1864_790_with_camera_labels',
        #     # 'segment-1457696187335927618_595_027_615_027_with_camera_labels'
        #     ]
        # self.datalist_scene= ['segment-2367305900055174138_1881_827_1901_827_with_camera_labels']
        self.statistics = dict()
        for i in range(len(self.datalist_scene)):
            video_name = self.datalist_scene[i]
            self.statistics[video_name]={'total':0,'short':0,'complex':0,'remain':0}
            temp_datalist = datalist_video[video_name]
            self.datalist += temp_datalist
        self.statistics['total'] = 0
        self.statistics['short'] = 0
        self.statistics['complex'] = 0
        self.statistics['remain'] = 0
        self.run_for_videos()