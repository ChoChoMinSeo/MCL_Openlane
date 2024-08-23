import cv2
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from PIL import Image
from libs.utils import *

class Dataset_OpenLane(Dataset):
    def __init__(self, cfg, video_name=None):
        self.cfg = cfg
        if video_name is None:
            self.datalist = load_pickle(f'{cfg.dir["pre0"]}/datalist')
        else:
            self.datalist = load_pickle(f'{cfg.dir["out"]}/pickle/datalist_video')[video_name]
        # image transform
        self.transform = transforms.Compose([transforms.Resize((cfg.height, cfg.width), interpolation=2), transforms.ToTensor()])
        self.normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)

    def get_label(self, idx):
        data = load_pickle(f'{self.cfg.dir["pre0"]}/{self.datalist[idx]}')

        seg_label = np.zeros((self.cfg.node_num,2,len(data['lane']) ), dtype=np.float32)

        seg_label = np.ascontiguousarray(seg_label)
        lane_pts = list()
        for i in range(len(data['lane'])):
            pts_org = np.float32(data['lane'][i])
            lane_pts.append(pts_org)

            pts_org = pts_org.reshape((-1, 1, 3))

            pts_org = pts_org[np.where(pts_org[:,0,0]>self.cfg.min_x_coord)]
            pts_org = pts_org[np.where(pts_org[:,0,0]<self.cfg.max_x_coord)]
            pts_org = pts_org[np.where(pts_org[:,0,1]>self.cfg.min_y_coord)]
            pts_org = pts_org[np.where(pts_org[:,0,1]<self.cfg.max_y_coord)]
            seg_label[:pts_org.shape[0],:,i] = pts_org[:,0,:2]

        return {'label': np.float32(seg_label),
                'lane_pts': lane_pts,
                'extrinsic': data['extrinsic'],
                'intrinsic': data['intrinsic']
                }

    def __getitem__(self, idx):
        out = dict()
        out['img_name'] = self.datalist[idx]
        out.update(self.get_label(idx))
        return out

    def __len__(self):
        return len(self.datalist)