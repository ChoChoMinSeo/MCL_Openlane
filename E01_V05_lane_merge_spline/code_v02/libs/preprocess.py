# Ver.
import cv2
import torch.nn.functional as F

import math
import time

from scipy.optimize import linear_sum_assignment
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy.interpolate import UnivariateSpline

from libs.utils import *
from datasets.dataset_openlane import *
from visualizes.visualize_module import Runner
from visualizes.vis_config import cfg_vis
from visualizes.utils.utils import *
from libs.extrapolation_modes import do_extrapolation

class Preprocessing(object):
    def __init__(self, cfg):
        self.cfg = cfg
        # extrapolation 모드 변경
        # cfg_vis.extra_mode = 'custom_spline'
        cfg_vis.extra_degree = self.cfg.extra_degree
        
        self.vis_runner = Runner(cfg_vis)
    def get_angle(self,a, b, c):
        a[0]/= self.x_std
        a[1]/= self.y_std
        a[2]/= self.z_std
        b[0]/= self.x_std
        b[1]/= self.y_std
        b[2]/= self.z_std
        c[0]/= self.x_std
        c[1]/= self.y_std
        c[2]/= self.z_std
        a = a[:2]
        b = b[:2]
        c = c[:2]

        ba = a-b
        bc = c-b
        dot_product = np.dot(ba,bc)
        cos_theta = dot_product/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-8)
        angle = np.arccos(cos_theta)
        return angle
    
    def lane_smoothing(self,lane):
        x,y,z = lane[:,0],lane[:,1],lane[:,2]
        
        if len(lane)>5:
            spl = UnivariateSpline(y, x,s=3)
            x_smooth = spl(y)
        else:
            A = np.vstack([y, np.ones(len(y))]).T
            m, c = np.linalg.lstsq(A, x, rcond=None)[0]

            # 직선 방정식 y = mx + c
            x_smooth = m * y + c
        return np.vstack((x_smooth,y,z)).T

    def get_std(self,lanes):
        scene_x = np.array([])
        scene_y = np.array([])
        scene_z = np.array([])
        for lane in lanes:
            scene_x = np.append(scene_x,lane[:,0])
            scene_y = np.append(scene_y,lane[:,1])
            scene_z = np.append(scene_z,lane[:,2])

        self.x_std = scene_x.std()
        self.y_std = scene_y.std()
        self.z_std = scene_z.std()


    def measure_angles(self,org_lane,new_lane):
        if len(new_lane)<5:
            return 999
        
        if org_lane[:,1][-1]>=new_lane[:,1][-1]>=org_lane[:,1][0] or new_lane[:,1][-1]>=org_lane[:,1][-1]>=new_lane[:,1][0]:
            if len(org_lane)<len(new_lane):
                crop_idx = max(1,len(org_lane)//4)
                org_lane = org_lane[crop_idx:-crop_idx]
            else:
                crop_idx = max(1,len(new_lane)//4)
                new_lane = new_lane[crop_idx:-crop_idx]
            if org_lane[:,1][-1]>=new_lane[:,1][-1]>=org_lane[:,1][0] or new_lane[:,1][-1]>=org_lane[:,1][-1]>=new_lane[:,1][0]:
                return 999
            if len(new_lane)<5 or len(org_lane)<5:
                return 999
        
        new_y = new_lane[:,1]
        org_y = org_lane[:,1]

        if org_y[-1]>new_y[-1]:
            flag = 1
        else:
            flag = 2
        crop_idx = max(1,len(org_lane)//8)
        cropped_org_lane = org_lane[crop_idx:-crop_idx]
        crop_idx = max(1,len(new_lane)//8)
        cropped_new_lane = new_lane[crop_idx:-crop_idx]

        if flag == 1:
            angle1 = self.get_angle(org_lane[max(2,len(org_lane)//4)].copy(),org_lane[0].copy(),new_lane[-1].copy())
            angle2 = self.get_angle(org_lane[0].copy(),new_lane[-1].copy(),new_lane[-max(2,len(new_lane)//4)].copy())

            angle1_c = self.get_angle(cropped_org_lane[max(2,len(cropped_org_lane)//4)].copy(),cropped_org_lane[0].copy(),cropped_new_lane[-1].copy())
            angle2_c = self.get_angle(cropped_org_lane[0].copy(),cropped_new_lane[-1].copy(),cropped_new_lane[-max(2,len(cropped_new_lane)//4)].copy())
        else:
            angle1 = self.get_angle(new_lane[max(2,len(new_lane)//4)].copy(),new_lane[0].copy(),org_lane[-1].copy())
            angle2 = self.get_angle(new_lane[0].copy(),org_lane[-1].copy(),org_lane[-max(2,len(org_lane)//4)].copy())

            angle1_c = self.get_angle(cropped_new_lane[max(2,len(cropped_new_lane)//4)].copy(),cropped_new_lane[0].copy(),cropped_org_lane[-1].copy())
            angle2_c = self.get_angle(cropped_new_lane[0].copy(),cropped_org_lane[-1].copy(),cropped_org_lane[-max(2,len(cropped_org_lane)//4)].copy())
        angle1 = max(angle1, angle1_c)
        angle2 = max(angle2, angle2_c)

        dist1 = math.pi-angle1
        dist2 = math.pi-angle2
        # print(dist1, dist2)
        if dist1<0.5 and dist2<0.5:
            return min(dist1+dist2,0.849)
        # elif (dist1<0.7 and dist2<1) or (dist1<1 and dist2<0.7):
        #     if flag==1:
        #         distance_y = org_lane[0][1]-new_lane[-1][1]
        #         distance_x = abs(org_lane[0][0]-new_lane[-1][0])*5
        #     else:
        #         distance_y = new_lane[0][1]-org_lane[-1][1]
        #         distance_x = abs(new_lane[0][0]-org_lane[-1][0])*5
        #     # print(distance_y,distance_x)
        #     if distance_y<2 or distance_x<2:
        #         return 0.849
        #     return 999
        else:
            return 999

    def linear_sum_assignments_with_inf(self,mat):
        nan = np.isnan(mat).any()
        INF = 1e+5
        if nan:
            mat[np.isnan(mat)] = INF
        return linear_sum_assignment(mat)
    
    def merge_lanes(self, lane1, lane2, lanes, idx1, idx2):
        max_y1, max_y2 = max(lane1[:,1]),max(lane2[:,1])
        min_y1, min_y2 = min(lane1[:,1]),min(lane2[:,1])
        # # check upper lane
        if max_y1 > max_y2:
            # upper = 1
            # if overlap
            if min_y1 < max_y2:
                # crop lane2                
                lane2 = lane2[(lane2[:,1]<min_y1),:]
                if len(lane2)==0:
                    return lane1,False
            new_lane = np.concatenate([lane2,lane1],dtype=np.float16)
            y1, y2 = lane2[-1][1], lane1[0][1]
            x1, x2 = lane2[-1][0], lane1[0][0]
        else:
            # upper = 2
            if min_y2 < max_y1:
                # crop lane 2
                lane2 = lane2[(lane2[:,1]>max_y1),:]
                if len(lane2)==0:
                    return lane1,False
            new_lane = np.concatenate([lane1,lane2],dtype=np.float16)
            y1, y2 = lane1[-1][1], lane2[0][1]
            x1, x2 = lane1[-1][0], lane2[0][0]
        
        # CHECK if there's overlapping lane
        m = (x2 - x1) / (y2 - y1)
        # y 절편 계산
        c = x1 - m * y1
        overlap = False
        for i in range(len(lanes)):
            if i != idx1 and i != idx2:
                temp = lanes[i]
                temp = temp[(temp[:,1]>y1)&(temp[:,1]<y2),:]
                temp = temp[np.argsort(temp[:,1]),:]
                direction = 0
                if len(temp)>1:
                    for pt in temp:
                        x = m*pt[1]+c
                        if x>pt[0]:
                            if direction == 2:
                                overlap = True
                                break
                            direction = 1
                        else:
                            if direction == 1:
                                overlap = True
                                break
                            direction = 2
                if overlap:
                    break
        if not overlap:
            return new_lane, overlap
        else:
            return None, overlap
            
    # merge
    def merge(self,datalist):
        print(f'construct lane matrix')
        # 같은 영상에 있는 frame들에 대해 수행
        for i in range(len(datalist)):
            img_name = datalist[i]
            dir_name = img_name.split('/')[0]
            # img_detailed_name = img_name.split('/')[1]

            # GET LANE DATA
            # from original data
            # data = load_pickle(f'{self.cfg.dir["pre0"]}/{img_name}')
            # from filtered data
            data = load_pickle(f'{self.cfg.dir["pre4"]}/{img_name}')
            lanes = data['lane3d']['new_lane'].copy()

            # REMOVE short or out of ranged lanes
            dot_idx = []
            for idx, lane in enumerate(lanes):
                if len(lane) < 5:
                    dot_idx.append(idx)
            for idx in range(len(dot_idx)-1,-1,-1):
                del lanes[dot_idx[idx]]
            # x,y,z 표준편차를 1로 다 정규화
            self.get_std(lanes)
            # LOAD current frame data
            data['lane3d']['org_lane'] = data['lane3d']['new_lane'].copy()
            print(f'{i} ==> load {img_name}')
            for iteration in range(2):
                print(f'iteration ==> {iteration}')
                # if less than 1 lane, BREAK
                if len(lanes) == 0 or len(lanes) == 1:
                    path = f'{self.cfg.dir["out"]}/pickle/{img_name}'
                    save_pickle(path=path, data=data)
                    break
                # VAR SETTING
                dist_arr = np.full((len(lanes),len(lanes)), 999, dtype=float)
                threshold = 0.85 # custom / dist threshold
                # GET N*N ARRAY
                # Calculate dist for every lane
                final_merged_idx = {
                    i:i for i in range(len(lanes))
                }
                for i_idx in range(len(lanes)):
                    for j_idx in range(i_idx,len(lanes)):
                        # print('idx',i_idx,'j',j_idx)
                        if i_idx == j_idx:
                            dist_arr[i_idx][j_idx] = 999
                        else:
                            dist = self.measure_angles(lanes[i_idx].copy(), lanes[j_idx].copy())
                            if dist>threshold:
                                dist = 999
                            dist_arr[i_idx][j_idx] = dist
                            dist_arr[j_idx][i_idx] = dist

                
                # HUNGARIAN
                row_idx, col_idx = self.linear_sum_assignments_with_inf(dist_arr.copy())
                # HUNGARIAN: GET ORDERED PAIR
                # 합칠 index 설정
                for (lane1_idx, lane2_idx) in zip(row_idx,col_idx):
                    if dist_arr[lane1_idx][lane2_idx] < threshold and final_merged_idx[lane1_idx]==lane1_idx:
                        final_merged_idx[lane2_idx] = lane1_idx
                # MERGE
                # Merge lanes with full_merged_idx and select indexes that need to be deleted.
                # print(final_merged_idx)
                for i in range(len(lanes)):
                    dest_idx = final_merged_idx[i]
                    if dest_idx != i:
                        # unique한 y값을 원본 기준으로 남김
                        new_lane, overlap = self.merge_lanes(lanes[dest_idx],lanes[i],lanes,dest_idx, i)
                        if not overlap:
                            lanes[dest_idx] = new_lane.copy()
                        else:
                            final_merged_idx[i] = i

                # DELETE
                # 뒤에서부터 합쳐진 lane이면 lanes에서 삭제
                for i in range(len(lanes)-1,-1,-1):
                    if final_merged_idx[i] != i:
                        del lanes[i]
                # UPDATE
                # update dataset with merged data
                if len(data['lane3d']['new_lane']) == len(lanes):
                    print('same')
                    data['lane3d']['new_lane'] = lanes.copy()
                    break
                else:
                    print(len(data['lane3d']['new_lane']),'->',len(lanes),'merged')
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
                if len(data['lane3d']['new_lane'])==len(data['lane3d']['org_lane']):
                    dir_name = f'{self.cfg.dir["out"]}/same/'
                else:
                    dir_name = f'{self.cfg.dir["out"]}/merged/'
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
                p = multiprocessing.Process(target=self.merge,args = (datalist,))
                p.start()
                procs.append(p)
            for p in procs:
                p.join()
        else:
            self.merge(self.datalist)
        # self.merge(datalist)
        
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
        # datalist_scene= ['segment-2367305900055174138_1881_827_1901_827_with_camera_labels']
        # with open(f'{self.cfg.dir["out"]}/scenelist.txt','w') as f:
        #     for i in datalist_scene:
        #         f.write(str(i)+'\n')
        for i in range(len(datalist_scene)):
            self.video_idx = i
            self.video_name = datalist_scene[i]
            self.datalist_video = datalist_video[self.video_name]
            self.datalist += self.datalist_video
        self.run_for_videos()