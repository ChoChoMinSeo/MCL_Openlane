from options.config import Config
from options.args import *
from libs.prepare import *
from libs.preprocess import *

def run_preprocessing(cfg):
    preprocess = Preprocessing(cfg)
    preprocess.run()

def main():

    # option
    cfg = Config()
    cfg = parse_args(cfg)

    # run
    run_preprocessing(cfg)

if __name__ == '__main__':
    main()
