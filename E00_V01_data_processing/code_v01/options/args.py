import argparse

def parse_args(cfg):
    parser = argparse.ArgumentParser(description='Hello')
    # parser.add_argument('--dataset_dir', default='/media/dkjin/4fefb28c-5de9-4abd-a935-aa2d61392048/Dataset/OpenLane', help='dataset dir')
    # parser.add_argument('--dataset_dir', default='/home/dkjin/Project/Dataset/OpenLane', help='dataset dir')
    parser.add_argument('--dataset_dir', default='/home/asdf/2TB_2/openlane',help='dataset dir')

    args = parser.parse_args()

    cfg = args_to_config(cfg, args)
    return cfg

def args_to_config(cfg, args):
    if args.dataset_dir is not None:
        cfg.dir['dataset'] = args.dataset_dir
    return cfg