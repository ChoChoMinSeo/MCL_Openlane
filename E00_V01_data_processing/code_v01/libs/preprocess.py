from libs.utils import *

class Preprocessing(object):

    def __init__(self, cfg, dict_DB):
        self.cam_representation = np.linalg.inv(
            np.array([[0, 0, 1, 0],
                      [-1, 0, 0, 0],
                      [0, -1, 0, 0],
                      [0, 0, 0, 1]], dtype=float))
        self.R_vg = np.array([[0, 1, 0],
                         [-1, 0, 0],
                         [0, 0, 1]], dtype=float)
        self.R_gc = np.array([[1, 0, 0],
                         [0, 0, 1],
                         [0, -1, 0]], dtype=float)
        self.inv_R_vg = np.linalg.inv(self.R_vg)
        self.cfg = cfg

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
    def get_3dlanes(self, lane, extrinsic):
        lane = lane[:,:,0].T  # [3, n]
        ones = np.ones((1, lane.shape[1]))
        lane = np.vstack((lane, ones))  # [4, n], (x, y, z, 1)
        lane = np.matmul(extrinsic, np.matmul(self.cam_representation, lane))  # [4, n]
        lane = lane[0:3, :].T  # [n, 3]
        return lane
    
    def get_camera_coordinates(self, lane, extrinsic):
        # lane: 실제 좌표 [n, 3]
        # extrinsic: 외부 파라미터 행렬 [4, 4]

        # ones 추가하여 [n, 4] 형태로 만들기
        ones = np.ones((lane.shape[0], 1))
        lane = np.hstack((lane, ones))  # [n, 4]

        # extrinsic과 self.cam_representation의 역행렬 계산
        extrinsic_inv = np.linalg.inv(extrinsic)
        cam_representation_inv = np.linalg.inv(self.cam_representation)

        # 실제 좌표에서 카메라 좌표로 변환
        lane = np.matmul(cam_representation_inv, np.matmul(extrinsic_inv, lane.T))  # [4, n]
        lane = lane[0:3, :].T  # [n, 3]

        return lane
    def recalculate_extrinsic(self, extrinsic):
        # Re-calculate extrinsic matrix based on ground coordinate
        extrinsic[:3, :3] = np.matmul(np.matmul(
            np.matmul(self.inv_R_vg, extrinsic[:3, :3]),
            self.R_vg), self.R_gc)

        extrinsic[0:2, 3] = 0.0
        return extrinsic
    
    def run(self):
        print('start')
        datalist = self.get_datalist()
        # datalist = load_pickle(f'{self.cfg.dir["out"]}/pickle/datalist')
        self.datalist = list()
        total_lanes = 0
        for i in range(len(datalist)):
            img_name = datalist[i]
            json_name = f'{img_name.replace(".jpg", ".json")}'
            json_file_path = f'{self.cfg.dir["dataset"]}/lane3d_1000/{self.cfg.datalist_mode}/{json_name}'
            with open(json_file_path, 'r') as j:
                data = json.loads(j.read())

            out = dict()
            out['lane3d'] = {
                'new_lane':list(),
                'org_lane':list()
            }
            # out['track_id'] = list()
            # out['attribute'] = list()
            # out['category'] = list()
            self.extrinsic = self.recalculate_extrinsic(np.array(data['extrinsic']))

            out['extrinsic']=self.extrinsic
            out['intrinsic']=data['intrinsic']
            total_lanes += len(data['lane_lines'])
            for j in range(len(data['lane_lines'])):
                x, y, z = data['lane_lines'][j]['xyz']

                x = np.array(x).reshape(-1, 1).astype(np.float16)
                y = np.array(y).reshape(-1, 1).astype(np.float16)
                z = np.array(z).reshape(-1, 1).astype(np.float16)

                lane = np.expand_dims(np.concatenate((x,y,z),axis=1),-1)
                lane = self.get_3dlanes(lane,self.extrinsic)
                x,y,z = lane[:,0],lane[:,1],lane[:,2]

                unique_idx = np.unique(y,return_index=True)[1]
                y = y[unique_idx].reshape(-1, 1)
                x = x[unique_idx].reshape(-1, 1)
                z = z[unique_idx].reshape(-1, 1)

                sorted_idx = y.argsort(axis=0)
                y = y[sorted_idx].reshape(-1, 1)
                x = x[sorted_idx].reshape(-1, 1)
                z = z[sorted_idx].reshape(-1, 1)

                lane_pts = np.concatenate((x,y,z),axis=1)
            #     # lane_pts = np.expand_dims(self.get_camera_coordinates(lane,data['extrinsic']),-1)
            #     # lane_pts = np.expand_dims(lane,-1)

            #     # lane_pts = np.concatenate((x, y, z), axis=1)
            #     # lane_pts = np.ndarray.tolist(lane_pts)

            #     # for pt in range(len(lane_pts)):
            #     #     # print(lane_pts[pt,0,0],lane_pts[pt,1,0])
            #     #     if -10<lane_pts[pt,0,0]<10 and 3<lane_pts[pt,1,0]<103:
            #     #         temp = lane_pts[pt:,:,:]
            #     #         print(pt)
            #     #         break
                if len(lane_pts)>3:
                    out['lane3d']['new_lane'].append(lane_pts)
                # out['track_id'].append(data['lane_lines'][j]['track_id'])
                # out['attribute'].append(data['lane_lines'][j]['attribute'])
                # out['category'].append(data['lane_lines'][j]['category'])
            if self.cfg.save_pickle == True:
                path = f'{self.cfg.dir["out"]}/pickle/{img_name.replace("/images/", "").replace(".jpg", "")}'
                save_pickle(path=path, data=out)

            self.datalist.append(img_name.replace("/images/", "").replace(".jpg", ""))

            print('i : {}, : image name : {} done'.format(i, img_name))
        print(len(self.datalist),'total_lanes: ',total_lanes)

        if self.cfg.save_pickle == True:
            path = f'{self.cfg.dir["out"]}/pickle/datalist'
            save_pickle(path, data=self.datalist)