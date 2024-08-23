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
        self.H_crop = homography_crop_resize([self.cfg.org_h, self.cfg.org_w], self.cfg.crop_y, [self.cfg.resize_h, self.cfg.resize_w])

    def real_to_img_coordinate(self,lanes):
        P_g2im = projection_g2im_extrinsic(self.extrinsic, self.intrinsic)
        self.P_gt = np.matmul(self.H_crop, P_g2im)
        new_lanes = []
        self.scales = []
        for idx,lane in enumerate(lanes):
            x,y,z = lane[:,0],lane[:,1],lane[:,2]
            self.scales.append(y.reshape(1,-1))
            x_2d, y_2d = projective_transformation(self.P_gt, x, y, z)
            # valid_mask_2d = np.logical_and(np.logical_and(x_2d >= 0, x_2d < self.cfg.resize_w), np.logical_and(y_2d >= 0, y_2d < self.cfg.resize_h))
            # x_2d = x_2d[valid_mask_2d]
            # y_2d = y_2d[valid_mask_2d]
            new_lanes.append(np.concatenate([x_2d.reshape(-1,1),y_2d.reshape(-1,1),z.reshape(-1,1)],axis=1))
        return new_lanes

    def img_to_real_coordinate(self,lanes):
        new_lanes = []
        for idx,lane in enumerate(lanes):
            x_2d, y_2d,_z = lane[:,0], lane[:,1],lane[:,2]
            x,y,z = inverse_projective_transformation2(self.P_gt,x_2d,y_2d,sampling_pts=cfg_vis.y_samples)
            new_lanes.append(np.concatenate([x.reshape(-1,1),y.reshape(-1,1),z.reshape(-1,1)],axis=1))
        return new_lanes

    def visualize_datalist(self, datalist):
        org_dir_name = ''
        for i in tqdm(range(len(datalist))):
            img_name = datalist[i]
            dir_name = img_name.split('/')[0]
            # if dir_name != org_dir_name:
            #     org_dir_name = dir_name
            #     print(dir_name)
            # else:
            #     continue

            # data = load_pickle(f'{self.cfg.dir["pre7_als"]}/{img_name}')
            data = load_pickle(f'{self.cfg.dir["pre0"]}/{img_name}')

            self.extrinsic = np.array(data['extrinsic'])
            self.intrinsic = np.array(data['intrinsic'])
            # SAVE IMG
            if self.cfg.datalist_mode =='example':
                img_ori = cv2.imread(f'{self.cfg.dir["dataset"]}/images/validation/{img_name}.jpg')
            else:
                img_ori = cv2.imread(f'{self.cfg.dir["dataset"]}/images/{self.cfg.datalist_mode}/{img_name}.jpg')
            # checklist = data['checklist']
            cur_data = {
                'lane3d' :{
                    'org_lane': data['lane3d']['new_lane'],
                    'new_lane': self.img_to_real_coordinate(self.real_to_img_coordinate(data['lane3d']['new_lane']))
                },
                'extrinsic': np.array(data['extrinsic']),
                'intrinsic': np.array(data['intrinsic']),
                # 'checklist': checklist.copy()
            }

            dir_name = f'{self.cfg.dir["out"]}/display/'
            self.vis_runner.run(img=img_ori,data = cur_data, path=dir_name+img_name)


    def run_for_videos(self):
        print('total_frames: ',len(self.datalist))
        start_time = time.time()
        # visualize
        if self.cfg.display_all == True:
            if self.cfg.multiprocess:
                import multiprocessing
                import parmap
                quarter = len(self.datalist)//self.cfg.num_workers
                input_datalist = []
                for i in range(self.cfg.num_workers):
                    if i!=self.cfg.num_workers-1:
                        temp = self.datalist[quarter*i:quarter*(i+1)]
                    else:
                        temp = self.datalist[quarter*i:]
                    input_datalist.append(temp)
                result = parmap.map(self.visualize_datalist, input_datalist, pm_pbar=True, pm_processes=self.cfg.num_workers)
            else:
                self.visualize_datalist(self.datalist)
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
        self.datalist_scene =['segment-191862526745161106_1400_000_1420_000_with_camera_labels']
        # self.datalist_scene = ['segment-10203656353524179475_7625_000_7645_000_with_camera_labels']
        for i in range(len(self.datalist_scene)):
            video_name = self.datalist_scene[i]
            temp_datalist = datalist_video[video_name]
            self.datalist += temp_datalist
        self.run_for_videos()