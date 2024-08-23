import torch
import torch.nn.functional as F

from libs.utils import *
import shutil
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from scipy.interpolate import splrep,splev
from numpy.polynomial import Polynomial
from sklearn.linear_model import Ridge
from tqdm import tqdm

class Preprocessing(object):
    def __init__(self, cfg, dict_DB):
        self.cfg = cfg
        self.visualize = dict_DB['visualize']
        self.shape_mode = ['straight', 'curve']
    
    def convert_lane_pts(self, lane):
        lane = lane[lane[:, 1] < self.cfg.max_y]

        unique_idx = np.sort(np.unique(lane[:, 1], return_index=True)[1])
        lane = lane[unique_idx]
        lane = lane[lane[:, 0]<self.cfg.max_x]
        lane = lane[lane[:, 0]>self.cfg.min_x]
        lane = lane[lane[:, 2]<self.cfg.max_z]
        lane = lane[lane[:, 2]>self.cfg.min_z]
        lane[:,0] += self.cfg.max_x
        lane[:,2] += self.cfg.max_z

        return lane
    
    def straight_or_curve(self,lane):
        min_x = min(lane[:,0])
        max_x = max(lane[:,0])

        if max_x-min_x<1:
            return 0
        else:
            return 1
    # 1)
    def construct_lane_matrix(self):
        print(f'construct lane matrix')
        self.mat = dict()
        self.mat['x_pts'] = torch.zeros((self.cfg.max_y, 0), dtype=torch.float32).cuda()
        self.mat['z_pts'] = torch.zeros((self.cfg.max_y, 0), dtype=torch.float32).cuda()

        self.mat['idx'] = torch.zeros((1, 0), dtype=torch.int32).cuda()
        self.mat['lane_idx'] = torch.zeros((1, 0), dtype=torch.int32).cuda()
        self.mat['lane_type'] = torch.zeros((1, 0), dtype=torch.int32).cuda()
        self.mat['img_name'] = list()
        for i in range(len(self.datalist)):
            img_name = self.datalist[i]
            # data = load_pickle(f'{self.cfg.dir["pre1_5"]}/{img_name}')
            data = load_pickle(f'{self.cfg.dir["pre1_6"]}/{img_name}')

            lanes_data = data['lane3d']['new_lane']
            self.extrinsic = data['extrinsic']
            self.intrinsic = data['intrinsic']
            for j in range(len(lanes_data)):
                if len(lanes_data[j]) == 0:
                    continue

                lane_pts = self.convert_lane_pts(lanes_data[j])
                self.mat['lane_type'] = torch.cat((self.mat['lane_type'],to_tensor(np.array([self.straight_or_curve(lane_pts)]).reshape(-1,1))),dim=1)
                col_mat_x = np.zeros((self.cfg.max_y, 1), dtype=np.float32)
                col_mat_x[np.int32(lane_pts[:, 1]), 0] = lane_pts[:, 0]
                col_mat_z = np.zeros((self.cfg.max_y, 1), dtype=np.float32)
                col_mat_z[np.int32(lane_pts[:, 1]), 0] = lane_pts[:, 2]

                self.mat['x_pts'] = torch.cat((self.mat['x_pts'], to_tensor(col_mat_x)), dim=1)
                self.mat['z_pts'] = torch.cat((self.mat['z_pts'], to_tensor(col_mat_z)), dim=1)
                
                self.mat['idx'] = torch.cat((self.mat['idx'], to_tensor(np.int32([i])).reshape(-1, 1)), dim=1)
                self.mat['lane_idx'] = torch.cat((self.mat['lane_idx'], to_tensor(np.int32([j])).reshape(-1, 1)), dim=1)
                self.mat['img_name'].append(img_name)

            if i % 10 == 0:
                print(f'{self.video_idx} : {self.video_name} --> img idx {i} done!')

        if self.cfg.save_pickle == True:
            for key in self.mat.keys():
                try:
                    self.mat[key] = to_np(self.mat[key])
                except:
                    self.mat[key] = np.array(self.mat[key]).reshape(1, -1)
            save_pickle(f'{self.cfg.dir["out"]}/pickle/data/matrix', data=self.mat)
            
    # 2)
    def divide_lane_matrix(self):
        print(f'refine lane matrix')
        mat_all = load_pickle(f'{self.cfg.dir["out"]}/pickle/data/matrix')
        for mode in self.shape_mode:
            out = dict()
            flag = 0 if mode == 'straight' else 1
            idx = (mat_all['lane_type'] == flag)
            for key in mat_all.keys():
                out[key] = mat_all[key][:, idx[0]]
            if self.cfg.save_pickle == True:
                save_pickle(f'{self.cfg.dir["out"]}/pickle/data/matrix_{mode}', data=out)

    def determine_matrix_height(self, mat_x,mat_z):
        h_idx = [self.cfg.min_y, self.cfg.max_y]
        mat_x = np.float32(mat_x[h_idx[0]:h_idx[1]])
        mat_z = np.float32(mat_z[h_idx[0]:h_idx[1]])
        y_pts = np.int32(np.arange(0, self.cfg.max_y))[h_idx[0]:h_idx[1]][:, np.newaxis]
        return mat_x,mat_z, y_pts
    
    # 3)                
    def refine_lane_matrix(self):
        print(f'refine lane matrix')
        for mode in self.shape_mode:
            mat = load_pickle(f'{self.cfg.dir["out"]}/pickle/data/matrix_{mode}')
            mat['x_pts'],mat['z_pts'], y_pts = self.determine_matrix_height(mat['x_pts'],mat['z_pts'])

            if self.cfg.save_pickle == True:
                save_pickle(f'{self.cfg.dir["out"]}/pickle/data/matrix_{mode}', data=mat)
                save_pickle(f'{self.cfg.dir["out"]}/pickle/data/y_pts_{mode}', data=y_pts)

    def partial_curve(self,input_x, input_y, sampling_pts):
        dx_dy = np.gradient(input_x, input_y)
        d2x_dy2 = np.gradient(dx_dy, input_y)
        t_idx = np.argmin(np.abs(d2x_dy2))
        if t_idx == 0 or t_idx == len(input_y)-1:
            t_idx = len(input_y)//2
        x0,y0 = input_x[0],input_y[0]
        xn,yn = input_x[-1],input_y[-1]
        xt,yt = input_x[t_idx],input_y[t_idx]
        thr = 0.01
        try:
            gr1 = np.abs(np.mean(np.gradient(np.gradient(input_y[:t_idx],input_x[:t_idx]),input_x[:t_idx])))
            # first_g1 = np.gradient(input_x[:t_idx],input_y[:t_idx])
            # first_g1_ = first_g1[~np.isnan(first_g1)]
            # gr1 = np.abs(np.mean(np.gradient(first_g1_,input_y[:t_idx][~np.isnan(first_g1)])))
        except:
            gr1 = 0
        try:
            gr2 = np.abs(np.mean(np.gradient(np.gradient(input_y[t_idx:],input_x[t_idx:]),input_x[t_idx:] )))

            # first_g2 = np.gradient(input_x[t_idx:],input_y[t_idx:])
            # first_g2_ = first_g2[~np.isnan(first_g2)]
            # print(np.gradient(first_g2_,input_y[t_idx:][~np.isnan(first_g2)]))
            # gr2 = np.abs(np.mean(np.gradient(first_g2_,input_y[t_idx:][~np.isnan(first_g2)])))
        except:
            gr2 = 0
        if gr1<=thr:
            m1 = (yt-y0)/(xt-x0)
            k1 = y0-m1*x0
            if gr2>thr:
                a2,b2,c2 = np.matmul(np.linalg.inv([[xt**2,xt,1],[xn**2,xn,1],[2*xt,1,0]]),np.array([yt,yn,m1]).T).T
        if gr2<=thr:
            m2 = (yn-yt)/(xn-xt)
            k2 = yt-m2*xt
            if gr1>thr:
                a1,b1,c1 = np.matmul(np.linalg.inv([[xt**2,xt,1],[x0**2,x0,1],[2*xt,1,0]]),np.array([yt,y0,m2]).T).T
        if gr1>thr and gr2>thr:
            # A = [[xt**2,xt,1,0,0,0],
            #      [x0**2,x0,1,0,0,0],
            #      [0,0,0,xt**2,xt,1],
            #      [0,0,0,xn**2,xn,1],
            #      [2*xt,1,0,-2*xt,-1,0],
            #      [1,0,0,0,0,0]
            #      ]
            # B = np.array([yt,y0,yt,yn,0,0]).T
            # a1,b1,c1,a2,b2,c2 = np.matmul(np.linalg.inv([A]),B).T
            a1,b1,c1,a2,b2,c2 = self.curve_only_param(input_x,input_y)

        if gr1<=thr:
            def f1(new_x):
                return m1*new_x+k1
        else:
            def f1(new_x):
                return a1*(new_x**2)+b1*new_x+c1
        if gr2<=thr:
            def f2(new_x):
                return m2*new_x+k2
        else:
            def f2(new_x):
                return a2*(new_x**2)+b2*new_x+c2
            
        sampling_pts_1 = sampling_pts[sampling_pts<input_x[t_idx]]
        sampling_pts_2 = sampling_pts[sampling_pts>=input_x[t_idx]]
        new_y1 = f1(sampling_pts_1)
        new_y2 = f2(sampling_pts_2)
        new_y = np.append(new_y1,new_y2).reshape(-1,1)

        return new_y
    
    def curve_only_param(self,input_x, input_y):
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
        return a1,b1,c1,a2,b2,c2
    
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

    # 4)
    def complete_lane_matrix(self):
        print(f'complete lane matrix')
        for mode in self.shape_mode:
            mat = load_pickle(f'{self.cfg.dir["out"]}/pickle/data/matrix_{mode}')
            y_pts = load_pickle(f'{self.cfg.dir["out"]}/pickle/data/y_pts_{mode}')

            complete_matrix = np.zeros((mat['x_pts'].shape[0],2,mat['x_pts'].shape[1]))
            for idx in tqdm(range(mat['z_pts'].shape[1])):
                pts_idx = (mat['z_pts'][:,idx]!=0)
                x_pts_ = mat['x_pts'][pts_idx,idx]
                z_pts_ = mat['z_pts'][pts_idx,idx]
                y_pts_ = y_pts[pts_idx,0]
                if len(y_pts_)>4:
                    # for z
                    f_z = interp1d(y_pts_, z_pts_,bounds_error=False, fill_value = (z_pts_[0],z_pts_[-1]), kind='linear')
                    z_new = f_z(y_pts)

                    # for x
                    interpolate_y = np.array(range(min(y_pts_),max(y_pts_)))
                    f_x = interp1d(y_pts_,x_pts_)
                    x_new = f_x(interpolate_y)
                    if len(x_new)>4:
                        # x_new = self.curve_only(interpolate_y,x_new,y_pts)
                        # x_new = self.partial_curve(interpolate_y,x_new,y_pts)
                        f_x = CubicSpline(interpolate_y,x_new,  bc_type='natural',extrapolate='True')
                        x_new = f_x(y_pts)

                    else:
                        f_x = interp1d(interpolate_y,x_new, fill_value="extrapolate")
                        x_new = f_x(y_pts)


                    # if mode == 'straight':
                    #     f_x = interp1d(y_pts_, x_pts_,bounds_error=False, fill_value = 'extrapolate', kind='linear')
                    #     x_new = f_x(y_pts)
                    # else:
                    #     # interpolate with cubic spline
                    #     # f_x = CubicSpline(y_pts_,  bc_type='not-a-knot',extrapolate='True')
                    # np.concatenate([x_pts_.reshape(-1,1),z_pts_.reshape(-1,1)],axis=1),
                    # np.concatenate([x_pts_.reshape(-1,1),z_pts_.reshape(-1,1)],axis=1),
                    #     # x_new = f_x(y_pts)

                    #     # cubic spline with defined knots
                    #     # div = 3
                    #     # knots = np.array([y_pts_[y_pts_.shape[0]//div*i] for i in range(1,div)])
                    #     # tck_x = splrep(y_pts_,x_pts_,t = knots,k=3)
                    #     # x_new = splev(y_pts,tck_x)
                        
                    #     ## with polynomial
                    #     f_x = Polynomial.fit(y_pts_,x_pts_,deg=2)
                    #     x_new = f_x(y_pts)

                    #     # # extrapolate with ridge regression
                    #     # interpolate_y = np.array(range(min(y_pts_),max(y_pts_)))
                    #     # x_new = f_x(interpolate_y)

                    #     ## ridge
                    #     # ridge = Ridge().fit(interpolate_y.reshape(-1,1),x_new.reshape(-1,1))
                    #     # x_under = ridge.predict(np.array(range(self.cfg.min_y,min(y_pts_))).reshape(-1,1))
                    #     # x_upper = ridge.predict(np.array(range(max(y_pts_),self.cfg.max_y)).reshape(-1,1))
                    #     # x_new = np.append(x_under.reshape(-1,1),x_new.reshape(-1,1),axis=0)
                    #     # x_new = np.append(x_new.reshape(-1,1),x_upper.reshape(-1,1),axis=0)
                    
                    complete_matrix_ = np.concatenate([np.expand_dims(x_new-self.cfg.max_x,1), np.expand_dims(z_new-self.cfg.max_z,1)],axis=1).squeeze(-1)

                    complete_matrix[:,:,idx] = complete_matrix_

            if self.cfg.save_pickle == True:
                save_pickle(f'{self.cfg.dir["out"]}/pickle/data/complete_matrix_{mode}', data=complete_matrix)
            print(f'{mode} mode done!')
    # 6)
    def save_results(self):
        # copy
        datalist = load_pickle(f'{self.cfg.dir["out"]}/pickle/datalist')
        for i in range(len(datalist)):
            img_name = datalist[i]
            data = load_pickle(f'{self.cfg.dir["pre1_5"]}/{img_name}')
            data['lane3d']['org_lane'] = data['lane3d']['new_lane'].copy()
            save_pickle(f'{self.cfg.dir["out"]}/pickle/results/{img_name}',data)

        for mode in self.shape_mode:
            print(f'save results for {mode}')
            mat = load_pickle(f'{self.cfg.dir["out"]}/pickle/data/matrix_{mode}')
            y_pts = load_pickle(f'{self.cfg.dir["out"]}/pickle/data/y_pts_{mode}')
            complete_mat = load_pickle(f'{self.cfg.dir["out"]}/pickle/data/complete_matrix_{mode}')

            current_idx = -1
            mat_idx = mat['idx']
            mat_lane_idx = mat['lane_idx']
            for i in range(complete_mat.shape[2]):
                x_pts = complete_mat[:,0, i]
                z_pts = complete_mat[:,1, i]

                lane_pts = np.concatenate((x_pts[:, np.newaxis], y_pts,z_pts[:, np.newaxis]), axis=1)
                idx = mat_idx[0, i]
                lane_idx = mat_lane_idx[0, i]

                if current_idx != idx or (i == complete_mat.shape[1] - 1):
                    if current_idx != -1:
                        path = f'{self.cfg.dir["out"]}/pickle/results/{img_name}'
                        if self.cfg.save_pickle == True:
                            save_pickle(path, out)

                    img_name = self.datalist[idx]
                    path = f'{self.cfg.dir["out"]}/pickle/results/{img_name}'
                    if os.path.exists(f'{path}.pickle'):
                        out = load_pickle(path)
                        if 'check' not in out.keys():
                            out['check'] = list(np.zeros(len(out['lane3d']['new_lane']), dtype=np.int32))
                    else:
                        print('empty file error !!!')

                out['lane3d']['new_lane'][lane_idx] = lane_pts
                if mode == 'straight':
                    out['check'][lane_idx] = 1
                elif mode == 'curve':
                    out['check'][lane_idx] = 2
                current_idx = idx

    def plot_img(self,datalist):
        for i in tqdm(range(len(datalist))):
            img_name = datalist[i]
            path = f'{self.cfg.dir["out"]}/pickle/results/{img_name}'
            data = load_pickle(path)
            self.visualize.save_datalist(data= data,file_name=img_name)
            # print(f'{img_name} done')
    # 7)
    def plot_lanes(self):
        print('plot start')
        datalist = load_pickle(f'{self.cfg.dir["out"]}/pickle/datalist')
        import multiprocessing
        import parmap
        # procs = []
        quarter = len(datalist)//self.cfg.num_workers
        input_datalist = []
        for i in range(self.cfg.num_workers):
            if i!=self.cfg.num_workers-1:
                temp = datalist[quarter*i:quarter*(i+1)]
            else:
                temp = datalist[quarter*i:]
            input_datalist.append(temp)
        result = parmap.map(self.plot_img, input_datalist, pm_pbar=True, pm_processes=self.cfg.num_workers)
        #     p = multiprocessing.Process(target=self.plot_img,args = (temp,))
        #     p.start()
        #     procs.append(p)
        # for p in procs:
        #     p.join()

    def run_for_videos(self):
        self.construct_lane_matrix()

        self.divide_lane_matrix()

        self.refine_lane_matrix()

        self.complete_lane_matrix()

        self.save_results()

        self.plot_lanes()

    def run(self):
        print('start')
        self.init()
        # if os.path.exists(f'{self.cfg.dir["out"]}/pickle/datalist_video.pickle') == False:
        self.generate_video_datalist()
        datalist_video = load_pickle(f'{self.cfg.dir["out"]}/pickle/datalist_video')
        datalist_scene = list(datalist_video)
        # datalist_scene = ['segment-1071392229495085036_1844_790_1864_790_with_camera_labels']
        # datalist_scene = [
        #     'segment-17065833287841703_2980_000_3000_000_with_camera_labels',
        #     'segment-89454214745557131_3160_000_3180_000_with_camera_labels',
        #     'segment-271338158136329280_2541_070_2561_070_with_camera_labels',
        #     'segment-346889320598157350_798_187_818_187_with_camera_labels',
        #     'segment-967082162553397800_5102_900_5122_900_with_camera_labels',
        #     'segment-1071392229495085036_1844_790_1864_790_with_camera_labels',
        #     'segment-1457696187335927618_595_027_615_027_with_camera_labels'
        # ]
        self.datalist = list()
        for i in range(len(datalist_scene)):
            # frame index
            self.video_idx = i
            self.video_name = datalist_scene[i]
            # frame
            self.datalist_video = datalist_video[self.video_name]
            self.datalist += self.datalist_video
        self.run_for_videos()

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
        save_pickle(f'{self.cfg.dir["out"]}/pickle/datalist', datalist_org)

        import csv
        datalist_scene = list(datalist_out)
        path = f'{self.cfg.dir["out"]}/data/video_list_{self.cfg.datalist_mode}.csv'
        mkdir(os.path.dirname(path))
        with open(path, 'w', newline='') as csvfile:
            fieldnames = ['video_name']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(len(datalist_scene)):
                writer.writerow({'video_name': datalist_scene[i]})