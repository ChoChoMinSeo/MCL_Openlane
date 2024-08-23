from datasets.dataset_openlane import *

def prepare_dataloader(cfg, dict_DB):
    dataset = Dataset_OpenLane(cfg=cfg,video_name='segment-2367305900055174138_1881_827_1901_827_with_camera_labels')
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=cfg.batch_size,
                                             shuffle=False,
                                             num_workers=cfg.num_workers,
                                             pin_memory=False)

    dict_DB['dataloader'] = dataloader

    return dict_DB

