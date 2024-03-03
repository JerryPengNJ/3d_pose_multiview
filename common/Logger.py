import logging
import os
import inspect
class MyLogger(logging.Logger):
    def __init__(self, filename=None, mode='w'):
        super().__init__(self)
        self.setLevel(level=logging.INFO)
        if filename is None:
            caller_frame = inspect.currentframe().f_back
            filename = inspect.getframeinfo(caller_frame).filename.split('.')[0]
            self.filename = filename + '.log'
        else:
            self.filename = filename + '.log'
        self.mode = mode
        formatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler = logging.FileHandler(self.filename, mode=self.mode)
        file_handler.setLevel(level=logging.INFO)
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level=logging.INFO)
        stream_handler.setFormatter(formatter)
        self.addHandler(stream_handler)
        self.addHandler(file_handler)