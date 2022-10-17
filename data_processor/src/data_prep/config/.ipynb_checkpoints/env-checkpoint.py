import os
BASE_PATH = os.environ.get("BASE_PATH",'/home/resifis/Desktop/kaustcode/Packages/data_processor/src/data_prep')


class ConfigData ():
    def __init__(self,window_size,target_len):
        self.window_size = window_size
        self.target_len = target_len