import os
try:
    import pickle5 as pickle
except:
    import pickle
import numpy as np
import random
import torch
import shutil

global global_seed

global_seed = 123
torch.manual_seed(global_seed)
torch.cuda.manual_seed(global_seed)
torch.cuda.manual_seed_all(global_seed)
np.random.seed(global_seed)
random.seed(global_seed)

def _init_fn(worker_id):

    seed = global_seed + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    return

# convertor
def to_tensor(data):
    return torch.from_numpy(data).cuda()

def to_tensor2(data):
    return torch.from_numpy(data)

def to_np(data):
    return data.cpu().numpy()

def to_np2(data):
    return data.detach().cpu().numpy()

def to_3D_np(data):
    return np.repeat(np.expand_dims(data, 2), 3, 2)

def logger(text, LOGGER_FILE):  # write log
    with open(LOGGER_FILE, 'a') as f:
        f.write(text)
        f.close()


# directory & file
def mkdir(path):
    if os.path.exists(path) == False:
        os.makedirs(path)


def rmfile(path):
    if os.path.exists(path):
        os.remove(path)

def rmdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
# pickle
def save_pickle(path, data):

    '''
    :param file_path: ...
    :param data:
    :return:
    '''
    mkdir(os.path.dirname(path))
    with open(path + '.pickle', 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(file_path):
    with open(file_path + '.pickle', 'rb') as f:
        data = pickle.load(f)

    return data

def save_dict_to_txt(dictionary, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for key, value in dictionary.items():
            # 키와 값 사이에 콜론(:)과 공백을 넣고 한 줄에 하나씩 저장
            file.write(f"{key}: {value}\n")