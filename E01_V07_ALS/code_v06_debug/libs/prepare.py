from visualizes.visualize import *
from libs.als import *


def prepare_visualization(cfg, dict_DB):
    dict_DB['visualize'] = Visualize(cfg)
    return dict_DB

def prepare_als(cfg, dict_DB):
    dict_DB['als'] = ALS_pyspark(cfg)
    return dict_DB
