from libs.utils import *
import tqdm
import os
from libs.vis_openlane import LaneVis

class Preprocessing(object):

    def __init__(self, cfg, dict_DB):
        self.cfg = cfg
        self.visualize = dict_DB['visualize']

        # self.visualizer = LaneVis(self.cfg)
    def to_xyz(self,lanes):
        y_steps = np.linspace(1, 100, 100)
        new_lanes = []
        for lane in lanes:
            xs, zs, vises = lane[5:105].copy(), lane[205:305].copy(), lane[405:505].copy()
            xs = xs[vises > 0.5]
            zs = zs[vises > 0.5]
            ys = y_steps[vises > 0.5]
            temp = np.concatenate([xs.reshape(-1,1),ys.reshape(-1,1),zs.reshape(-1,1)],axis=1)
            new_lanes.append(temp)
        return new_lanes
    
    def plot_img(self,datalist):
        for i in range(len(datalist)):
            img_name = datalist[i]
            path_anchor = f'{self.cfg.dir["dataset"]}/cache_dense/validation/{img_name}.pkl'
            data_anchor = load_pkl(path_anchor)
            path_als = f'{self.cfg.dir["als"]}/{img_name}'
            data_als = load_pickle(path_als)
            vis_data = {
                'extrinsic' : data_anchor['gt_camera_extrinsic'],
                'intrinsic' : data_anchor['gt_camera_intrinsic'],
                'lane3d':{
                    'new_lane': self.to_xyz(data_anchor['gt_3dlanes']),
                    'org_lane':data_als['lane3d']['new_lane']
                }
            }
            self.visualize.save_datalist(data= vis_data,file_name=img_name)
            print(f'{img_name} done')

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
    def init(self):
        self.datalist = list()
        self.datalist_error = list()

    def get_datalist(self):
        datalist = list()
        path = f'{self.cfg.dir["dataset"]}/lane3d_1000/{self.cfg.datalist_mode}'
        scene_list = os.listdir(path)
        for i in range(len(scene_list)):
            scene_path = f'{path}/{scene_list[i]}'
            file_list = os.listdir(scene_path)
            for j in range(len(file_list)):
                datalist.append(f'{scene_list[i]}/{file_list[j].replace(".json", ".jpg")}')

        # save_pickle(path=f'{self.cfg.dir["out"]}/pickle/datalist', data=datalist)
        datalist = sorted(datalist)
        return datalist
    
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
        self.plot_lanes()
        
    def generate_video_datalist(self):
        datalist_org = load_pickle(f'{self.cfg.dir["datalist"]}/pickle/datalist')
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