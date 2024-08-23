import os
from libs.prepare import *

from options.config import Config
from options.args import *
from libs.preprocess import *

def run(cfg, dict_DB):
    preprocessor = Preprocessing(cfg, dict_DB)
    preprocessor.run()

def main():
    # option
    cfg = Config()
    cfg = parse_args(cfg)

    # prepare
    dict_DB = dict()
    dict_DB = prepare_visualization(cfg, dict_DB)

    # run
    run(cfg, dict_DB)

if __name__ == '__main__':
    main()
