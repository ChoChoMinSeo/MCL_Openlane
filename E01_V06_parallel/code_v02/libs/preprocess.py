# Ver.
import cv2
import torch.nn.functional as F
import time

from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import InterpolatedUnivariateSpline
from libs.utils import *
from datasets.dataset_openlane import *
from visualizes.visualize_module import Runner
from visualizes.vis_config import cfg_vis
from visualizes.utils.utils import *

class Preprocessing(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.vis_runner = Runner(cfg_vis)

    def duplicated_pts(self,lanes):
        # get point space, erase duplicated pts
        point_space = np.zeros((self.cfg.max_y+1,(self.cfg.max_x-self.cfg.min_x)+1),dtype=np.bool_)
        new_lanes = []
        for lane in lanes:
            temp_lane = lane.copy()
            temp_lane[:,0] -= self.cfg.min_x
            temp_lane[:,0] = np.round(temp_lane[:,0])
            keep_idx = []
            for pt_idx,pt in enumerate(temp_lane):
                x,y,_ = pt.astype(np.int8)
                if not point_space[y][x]:
                    keep_idx.append(pt_idx)
                    point_space[y][x] = True
            new_lanes.append(lane[keep_idx].copy())
        
        return new_lanes
    
    def curve_only(self,input_x, input_y, sampling_pts):
        dx_dy = np.gradient(input_x, input_y)
        d2x_dy2 = np.gradient(dx_dy, input_y)
        t_idx = np.argmin(np.abs(d2x_dy2))
        if t_idx < 2 or t_idx > len(input_y)-3:
            t_idx = len(input_y)//2
        x0,y0 = input_x[0],input_y[0]
        xn,yn = input_x[-1],input_y[-1]
        xt1,yt1 = input_x[t_idx-1],input_y[t_idx-1]
        xt2,yt2 = input_x[t_idx+1],input_y[t_idx+1]
        m = (yt2-yt1)/(xt2-xt1)
        k = yt1-m*xt1
        a1,b1,c1 = np.matmul(np.linalg.inv([[xt1**2,xt1,1],[x0**2,x0,1],[2*xt1,1,0]]),np.array([yt1,y0,m]).T).T
        a2,b2,c2 = np.matmul(np.linalg.inv([[xt2**2,xt2,1],[xn**2,xn,1],[2*xt2,1,0]]),np.array([yt2,yn,m]).T).T
        def f1(new_x):
            return a1*(new_x**2)+b1*new_x+c1
        def f2(new_x):
            return a2*(new_x**2)+b2*new_x+c2
            
        sampling_pts_1 = sampling_pts[sampling_pts<input_x[t_idx]]
        sampling_pts_2 = sampling_pts[sampling_pts>=input_x[t_idx]]
        new_y1 = f1(sampling_pts_1)
        new_y2 = f2(sampling_pts_2)
        new_y = np.append(new_y1,new_y2).reshape(-1,1)

        return new_y    
    
    def extrapolate_parallel_lane(self,lanes):
        new_lanes = []
        flag = False
        for idx, lane in enumerate(lanes):
            if len(lane)<30:
                basis_idx = -1
                dist = float('inf')
                basis_dist_arr=np.array([])
                for jdx,lane2 in enumerate(lanes):
                    if idx != jdx and len(lane2)>30:
                        common_idx1 = np.where(np.isin(lane[:,1],lane2[:,1]))[0]
                        if len(common_idx1)<len(lane)//2:
                            continue
                        common_idx2 = np.where(np.isin(lane2[:,1],lane[:,1]))[0]

                        dist_arr = lane2[common_idx2,0]-lane[common_idx1,0]
                        temp_dist = np.mean(lane2[common_idx2,0]-lane[common_idx1,0])
                        if abs(temp_dist)<abs(dist):
                            dist = temp_dist
                            basis_idx = jdx
                            basis_dist_arr = dist_arr
                            basis_commmon_idx = common_idx2
                # FIT dist_arr
                # exponential
                if len(basis_dist_arr) == 0:
                    new_lanes.append(lane)
                    continue
                sign = -1 if basis_dist_arr[0]<0 else 1
                min_y = min(lanes[basis_idx][basis_commmon_idx,1])
                try:
                    def func(x,a, b):
                        return a*np.exp(-b * x)
                    p0 = (10,0.1)
                    popt,pcov = curve_fit(func,xdata=lanes[basis_idx][basis_commmon_idx,1],ydata=basis_dist_arr*sign,p0=p0)
                    a,b= popt
                    dist_arr = func(lanes[basis_idx][:,1],a,b)
                except:
                    print('not fitted')
                    f_dist = UnivariateSpline(lanes[basis_idx][basis_commmon_idx,1],basis_dist_arr*sign,k=2)
                    dist_arr=f_dist(lanes[basis_idx][:,1])
                new_lane = lanes[basis_idx].copy()
                new_lane[:,0]-=(dist_arr*sign)
                new_lane = np.concatenate([lane,new_lane])
                new_lane = new_lane[np.unique(new_lane[:,1],return_index=True)[1]]
                new_lanes.append(new_lane)
                flag = True
            else:
                new_lanes.append(lane)
        return new_lanes, flag
    
    def fill_lanes(self,lanes):
        for idx,lane in enumerate(lanes):
            lanes[idx] = lane[np.unique(lane[:,1],return_index=True)[1],:]

        for idx,lane in enumerate(lanes):
            original_idx = lane[:,1]
            interpolate_y = np.array(range(min(lane[:,1]).astype(np.int8),max(lane[:,1]).astype(np.int8)+1))
            if len(original_idx) != len(interpolate_y):
                original_idx = np.where(np.isin(interpolate_y,original_idx))[0]
                f_z = interp1d(lane[:,1],lane[:,2],bounds_error=False, fill_value = (lane[:,2][0],lane[:,2][-1]), kind='linear')
                z_new = f_z(interpolate_y)
                f_x = InterpolatedUnivariateSpline(lane[:,1].astype(np.float32),lane[:,0].astype(np.float32),k=1)
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
            lanes = self.fill_lanes(lanes)
            flag = False
            if len(lanes)>1:
                lanes,flag = self.extrapolate_parallel_lane(lanes)
            # UPDATE
            # update dataset with merged data
            data['lane3d']['new_lane'] = lanes.copy()            
            if self.cfg.save_pickle == True:
                path = f'{self.cfg.dir["out"]}/pickle/{img_name}'
                save_pickle(path=path, data=data)
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # SAVE IMG
            if self.cfg.display_all == True :
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
                if flag == False:
                    dir_name = f'{self.cfg.dir["out"]}/same/'
                else:
                    dir_name = f'{self.cfg.dir["out"]}/modified/'
                    self.vis_runner.run(img=img_ori,data = cur_data, path=dir_name+img_name)
        print("Finished Merging!")

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