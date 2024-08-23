# Ver.
import cv2
import torch.nn.functional as F

import time

from scipy import interpolate
from scipy.interpolate import UnivariateSpline
from numpy.polynomial import Polynomial
from scipy.interpolate import CubicSpline
from scipy.interpolate import InterpolatedUnivariateSpline
from libs.utils import *
from visualizes.visualize_module import Runner
from visualizes.vis_config import cfg_vis
from visualizes.utils.utils import *
from libs.extrapolation_modes import do_extrapolation

class Preprocessing(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.vis_runner = Runner(cfg_vis)
    
    def interpolate(self,lanes):
        for idx,lane in enumerate(lanes):
            lanes[idx] = lane[np.unique(lane[:,1],return_index=True)[1],:]

        for idx,lane in enumerate(lanes):
            original_idx = lane[:,1]
            interpolate_y = np.array(range(min(lane[:,1]).astype(np.int8),max(lane[:,1]).astype(np.int8)+1))
            if len(original_idx) != len(interpolate_y):
                original_idx = np.where(np.isin(interpolate_y,original_idx))[0]
                f_z = interp1d(lane[:,1],lane[:,2], kind='linear')
                z_new = f_z(interpolate_y)
                # f_x = InterpolatedUnivariateSpline(lane[:,1].astype(np.float32),lane[:,0].astype(np.float32),k=3)
                f_x = interp1d(lane[:,1],lane[:,0], kind='linear')

                x_new = f_x(interpolate_y)
                new_lane = np.concatenate([x_new.reshape(-1,1),interpolate_y.reshape(-1,1),z_new.reshape(-1,1)],axis=1)
                new_lane[original_idx,:] = lane
                lanes[idx] = new_lane
            
        return lanes
    
    def refine_lane(self,datalist):
        print(f'construct lane matrix')
        # 같은 영상에 있는 frame들에 대해 수행
        for i in range(len(datalist)):
            img_name = datalist[i]
            dir_name = img_name.split('/')[0]
            # img_detailed_name = img_name.split('/')[1]
            # from merged data
            data = load_pickle(f'{self.cfg.dir["pre5"]}/{img_name}')
            lanes = data['lane3d']['new_lane'].copy()
            data['lane3d']['org_lane'] = data['lane3d']['new_lane'].copy()
            print(f'{i} ==> load {img_name}')

            lanes = self.interpolate(lanes)
            
            # UPDATE
            # update dataset with merged data
            data['lane3d']['new_lane'] = lanes.copy()            
            if self.cfg.save_pickle == True:
                path = f'{self.cfg.dir["out"]}/pickle/{img_name}'
                save_pickle(path=path, data=data)
            # # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # # SAVE IMG
            # if self.cfg.display_all == True :
            #     if self.cfg.datalist_mode =='example':
            #         img_ori = cv2.imread(f'{self.cfg.dir["dataset"]}/images/validation/{img_name}.jpg')
            #     else:
            #         img_ori = cv2.imread(f'{self.cfg.dir["dataset"]}/images/{self.cfg.datalist_mode}/{img_name}.jpg')
            #     cur_data = {
            #         'lane3d' :{
            #             'org_lane': data['lane3d']['org_lane'],
            #             'new_lane': data['lane3d']['new_lane']
            #         },
            #         'extrinsic': np.array(data['extrinsic']),
            #         'intrinsic': np.array(data['intrinsic']),
            #     }
            #     if flag == False:
            #         dir_name = f'{self.cfg.dir["out"]}/same/'
            #     else:
            #         dir_name = f'{self.cfg.dir["out"]}/modified/'
            #         self.vis_runner.run(img=img_ori,data = cur_data, path=dir_name+img_name)
        print("Finished Sampling!")

    def run_for_videos(self):
        print('total_frames: ',len(self.datalist))
        start_time = time.time()

        if self.cfg.multiprocess:
            import multiprocessing
            procs = []
            quarter = len(self.datalist)//self.cfg.num_workers
            for i in range(self.cfg.num_workers):
                if i!=self.cfg.num_workers-1:
                    datalist = self.datalist[quarter*i:quarter*(i+1)]
                else:
                    datalist = self.datalist[quarter*i:]
                p = multiprocessing.Process(target=self.refine_lane,args = (datalist,))
                p.start()
                procs.append(p)
            for p in procs:
                p.join()
        else:
            self.refine_lane(self.datalist)
        
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
        # if os.path.exists(f'{self.cfg.dir["out"]}/pickle/datalist_video.pickle') == False:
            # no datalist
        self.generate_video_datalist()
        datalist_video = load_pickle(f'{self.cfg.dir["out"]}/pickle/datalist_video')
        datalist_scene = list(datalist_video)
        self.datalist = list()
        # datalist_scene = [
        #     # 'segment-17065833287841703_2980_000_3000_000_with_camera_labels',
        #     'segment-89454214745557131_3160_000_3180_000_with_camera_labels',
        #     # 'segment-271338158136329280_2541_070_2561_070_with_camera_labels',
        #     # 'segment-346889320598157350_798_187_818_187_with_camera_labels',
        #     # 'segment-967082162553397800_5102_900_5122_900_with_camera_labels',
        #     # 'segment-1071392229495085036_1844_790_1864_790_with_camera_labels',
        #     # 'segment-1457696187335927618_595_027_615_027_with_camera_labels'
        #     ]
        # datalist_scene= ['segment-6161542573106757148_585_030_605_030_with_camera_labels']
        # with open(f'{self.cfg.dir["out"]}/scenelist.txt','w') as f:
        #     for i in datalist_scene:
        #         f.write(str(i)+'\n')
        for i in range(len(datalist_scene)):
            self.video_idx = i
            self.video_name = datalist_scene[i]
            self.datalist_video = datalist_video[self.video_name]
            self.datalist += self.datalist_video
        self.run_for_videos()