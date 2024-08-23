from options.config import Config
from options.args import *
from ui_tools.ui import Editor

def run_editor(cfg):
    editor = Editor(cfg)
    editor.run()

def main():

    # option
    cfg = Config()
    cfg = parse_args(cfg)

    # run
    run_editor(cfg)

if __name__ == '__main__':
    main()
