import torch

from libs.utils import *
import shutil

class Preprocessing(object):
    def __init__(self, cfg, dict_DB):
        self.cfg = cfg
        self.visualize = dict_DB['visualize']
        self.als = dict_DB['als']
        self.shape_mode = ['straight', 'l_curve', 'r_curve']
        # self.shape_mode = ['straight']
        self.target_sample = self.cfg.target_sample

        self.cam_representation = np.linalg.inv(
            np.array([[0, 0, 1, 0],
                      [-1, 0, 0, 0],
                      [0, -1, 0, 0],
                      [0, 0, 0, 1]], dtype=float))
        self.y_pts = np.array(range(self.cfg.min_y,self.cfg.max_y+1))
    
    def convert_lane_pts(self, lane):
        lane[:, 1] = np.round(lane[:, 1])
        lane = lane[lane[:, 1] < self.cfg.max_y+1]

        unique_idx = np.sort(np.unique(lane[:, 1], return_index=True)[1])
        lane = lane[unique_idx]
        lane = lane[lane[:, 0]<self.cfg.max_x]
        lane = lane[lane[:, 0]>self.cfg.min_x]
        lane = lane[lane[:, 2]<self.cfg.max_z]
        lane = lane[lane[:, 2]>self.cfg.min_z]
        lane[:,0] += (self.cfg.max_x)
        lane[:,2] += (self.cfg.max_z)

        return lane
    # 1)
    def construct_lane_matrix(self):
        print(f'construct lane matrix')
        self.mat = dict()
        self.mat['x_pts'] = torch.zeros((self.cfg.max_y+1, 0), dtype=torch.float32).cuda()
        self.mat['z_pts'] = torch.zeros((self.cfg.max_y+1, 0), dtype=torch.float32).cuda()

        self.mat['idx'] = torch.zeros((1, 0), dtype=torch.int32).cuda()
        self.mat['lane_idx'] = torch.zeros((1, 0), dtype=torch.int32).cuda()
        self.mat['img_name'] = list()
        self.mat['trainable'] = torch.zeros((1,0),dtype = torch.bool).cuda()
        for i in range(len(self.datalist)):
            img_name = self.datalist[i]
            data = load_pickle(f'{self.cfg.dir["pre6"]}/{img_name}')

            lanes_data = data['lane3d']['org_lane']
            self.extrinsic = data['extrinsic']
            self.intrinsic = data['intrinsic']
            for j in range(len(lanes_data)):
                if len(lanes_data[j]) == 0:
                    continue

                lane_pts = self.convert_lane_pts(lanes_data[j])
                col_mat_x = np.zeros((self.cfg.max_y+1, 1), dtype=np.float32)
                col_mat_x[np.int32(lane_pts[:, 1]), 0] = lane_pts[:, 0]
                col_mat_z = np.zeros((self.cfg.max_y+1, 1), dtype=np.float32)
                col_mat_z[np.int32(lane_pts[:, 1]), 0] = lane_pts[:, 2]

                self.mat['x_pts'] = torch.cat((self.mat['x_pts'], to_tensor(col_mat_x)), dim=1)
                self.mat['z_pts'] = torch.cat((self.mat['z_pts'], to_tensor(col_mat_z)), dim=1)

                if not data['checklist']['short'][j] and not data['checklist']['complex'][j]:
                    self.mat['trainable'] = torch.cat((self.mat['trainable'], to_tensor(np.int32([1])).reshape(-1, 1)), dim=1)
                else:
                    self.mat['trainable'] = torch.cat((self.mat['trainable'], to_tensor(np.int32([0])).reshape(-1, 1)), dim=1)

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
    
    def define_curve_type(self,x_pts):
        x_pts = x_pts[x_pts!=0]
        min_x_idx = np.argmin(x_pts)
        max_x_idx = np.argmax(x_pts)
        if x_pts[max_x_idx]-x_pts[min_x_idx]<1:
            return 0
        elif max_x_idx<min_x_idx:
            return 1
        else:
            return 2

    # 2)
    def divide_lane_matrix(self):
        print(f'divide lane matrix')
        mat_all = load_pickle(f'{self.cfg.dir["out"]}/pickle/data/matrix')
        lane_type = np.zeros((1,0))
        for i in range(mat_all['x_pts'].shape[1]):
            lane_type = np.concatenate([lane_type,np.array([self.define_curve_type(mat_all['x_pts'][:,i].copy())]).reshape(1,1)],axis=1)
        data = dict()
        for mode in range(len(self.shape_mode)):
            cur_mode = (lane_type[0]==mode)
            for key in mat_all.keys():
                data[key] = mat_all[key][:,cur_mode]
            
            if self.cfg.save_pickle == True:
                save_pickle(f'{self.cfg.dir["out"]}/pickle/data/matrix_{self.shape_mode[mode]}', data=data)
    # 3)
    def set_data_sampling(self):
        print(f'set data sampling')
        self.sample_idx = dict()
        for mode in self.shape_mode:
            print(f'set data sampling for {mode}')
            mat = load_pickle(f'{self.cfg.dir["out"]}/pickle/data/matrix_{mode}')
            if mode == 'straight':
                self.sample_idx[mode] = np.int32(np.linspace(0, len(mat['idx'][0]), 10))
            elif mode == 'l_curve':
                self.sample_idx[mode] = np.int32(np.linspace(0, len(mat['idx'][0]), 10))
            elif mode == 'r_curve':
                self.sample_idx[mode] = np.int32(np.linspace(0, len(mat['idx'][0]), 10))

    def determine_matrix_height(self, mat_x,mat_z):
        h_idx = [self.cfg.min_y, self.cfg.max_y+1]
        mat_x = np.float32(mat_x[h_idx[0]:h_idx[1]])
        mat_z = np.float32(mat_z[h_idx[0]:h_idx[1]])
        return mat_x,mat_z
    
    def filter_out_data2(self, mat):
        checklist = (np.sum((mat['x_pts'] != 0), axis=0) > mat['x_pts'].shape[0] // 30)
        for key in mat.keys():
            mat[key] = mat[key][:, checklist]
        return mat
    # 4)                
    def refine_lane_matrix(self):
        for mode in self.shape_mode:
            print(f'refine {mode} lane matrix')
            mat = load_pickle(f'{self.cfg.dir["out"]}/pickle/data/matrix_{mode}')
            for k in range(0, len(self.sample_idx[mode]) - 1):
                # if k != self.target_sample:
                #     continue
                # sampling
                mat_sampled = dict()
                for key in mat.keys():
                    mat_sampled[key] = mat[key][:, self.sample_idx[mode][k]:self.sample_idx[mode][k+1]]
                mat_sampled['x_pts'],mat_sampled['z_pts'] = self.determine_matrix_height(mat_sampled['x_pts'],mat_sampled['z_pts'])
                mat_sampled = self.filter_out_data2(mat_sampled)

                if self.cfg.save_pickle == True:
                    save_pickle(f'{self.cfg.dir["out"]}/pickle/data/matrix_{mode}_{k}', data=mat_sampled)
    # 5)
    def complete_lane_matrix(self):
        print(f'complete lane matrix via ALS')
        for mode in self.shape_mode:
            print(f'ALS for {mode}')

            for k in range(0, len(self.sample_idx[mode]) - 1):
                # if k != self.target_sample:
                #     continue
                mat = load_pickle(f'{self.cfg.dir["out"]}/pickle/data/matrix_{mode}_{k}')

                self.als.mode = mode

                train_idx = (mat['trainable'][0]==1)
                train_x,test_x = self.als.convert_data_to_csr_matrix(mat['x_pts'],train_idx)
                train_df = self.als.convert_data_to_dataframe(train_x)
                test_df = self.als.convert_data_to_dataframe(test_x)

                self.als.ALS(train_df, test_df, mat['z_pts'])

                # restore to original lane shape
                complete_matrix = np.concatenate([np.expand_dims(self.als.M_pred_x-self.cfg.max_x,1), np.expand_dims(self.als.M_pred_z-self.cfg.max_z,1)],axis=1)

                if self.cfg.save_pickle == True:
                    save_pickle(f'{self.cfg.dir["out"]}/pickle/data/complete_matrix_{mode}_{k}', data=complete_matrix)
                print(f'step {k} done!')
    # 6)
    def save_results(self):
        # copy
        datalist = load_pickle(f'{self.cfg.dir["out"]}/pickle/datalist')
        
        for i in range(len(datalist)):
            img_name = datalist[i]
            src_path = f'{self.cfg.dir["pre6"]}/{img_name}'
            tgt_path = f'{self.cfg.dir["out"]}/pickle/results/{img_name}'
            data = load_pickle(src_path)
            data['lane3d']['new_lane'] = data['lane3d']['org_lane'].copy()
            save_pickle(tgt_path,data)
            # mkdir(os.path.dirname(tgt_path))
            # shutil.copy(src_path, tgt_path)

        for mode in self.shape_mode:
            print(f'save results for {mode}')

            for k in range(0, len(self.sample_idx[mode]) - 1):
                # if k != self.target_sample:
                #     continue
                mat = load_pickle(f'{self.cfg.dir["out"]}/pickle/data/matrix_{mode}_{k}')
                complete_mat = load_pickle(f'{self.cfg.dir["out"]}/pickle/data/complete_matrix_{mode}_{k}')

                current_idx = -1
                mat_idx = mat['idx']
                mat_lane_idx = mat['lane_idx']
                for i in range(complete_mat.shape[2]):
                    print(f'sample num {k} ==> load {i}  ==> {complete_mat.shape[1]}')
                    x_pts = complete_mat[:,0, i]
                    z_pts = complete_mat[:,1, i]

                    lane_pts = np.concatenate((x_pts[:, np.newaxis], self.y_pts[:, np.newaxis],z_pts[:, np.newaxis]), axis=1)
                    idx = mat_idx[0, i]
                    lane_idx = mat_lane_idx[0, i]

                    if current_idx != idx or (i == complete_mat.shape[2] - 1):
                        if current_idx != -1:
                            path = f'{self.cfg.dir["out"]}/pickle/results/{img_name}'
                            if self.cfg.save_pickle == True:
                                save_pickle(path, out)

                        img_name = self.datalist[idx]
                        path = f'{self.cfg.dir["out"]}/pickle/results/{img_name}'
                        if os.path.exists(f'{path}.pickle'):
                            out = load_pickle(path)
                            # out['lane3d']['new_lane'] = out['lane3d']['org_lane'].copy()
                            if 'check' not in out.keys():
                                out['check'] = list(np.zeros(len(out['lane3d']['new_lane']), dtype=np.int32))
                        else:
                            print('empty file error !!!')

                    out['lane3d']['new_lane'][lane_idx] = lane_pts
                    if mode == 'straight':
                        out['check'][lane_idx] = 1
                    elif mode == 'l_curve':
                        out['check'][lane_idx] = 2
                    elif mode == 'r_curve':
                        out['check'][lane_idx] = 3
                    current_idx = idx

    def plot_img(self,datalist):
        for i in range(len(datalist)):
            img_name = datalist[i]
            path = f'{self.cfg.dir["out"]}/pickle/results/{img_name}'
            data = load_pickle(path)
            self.visualize.save_datalist(data= data,file_name=img_name)
            print(f'{img_name} done')
    # 7)
    def plot_lanes(self):
        print('plot start')
        datalist = load_pickle(f'{self.cfg.dir["out"]}/pickle/datalist')
        if self.cfg.multiprocessing:
            import multiprocessing
            procs = []
            quarter = len(datalist)//self.cfg.num_workers
            for i in range(self.cfg.num_workers):
                if i!=self.cfg.num_workers-1:
                    temp = datalist[quarter*i:quarter*(i+1)]
                else:
                    temp = datalist[quarter*i:]
                p = multiprocessing.Process(target=self.plot_img,args = (temp,))
                p.start()
                procs.append(p)
            for p in procs:
                p.join()
        else:
            self.plot_img(datalist)
        # # visualize
        # if mode == 'curve' or mode == 'straight':
        #     self.out_f = {
        #         'extrinsic': self.extrinsic,
        #         'intrinsic': self.intrinsic,
        #         'lanes': self.als.M_pred
        #     }
        #     self.visualize.save_datalist(self.out_f,self.file_name)

    def run_for_videos(self):
        self.construct_lane_matrix()

        self.divide_lane_matrix()

        self.set_data_sampling()

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