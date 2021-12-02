# __init__.py
from .gdf_filegen import gdf_LO
from .pot_filegen import pot_ctrlfile

# Timing utilities
import time

class Timer(object):
    def __init__(self, task):
        self.task = task

    def __enter__(self):
        print('STARTING {}'.format(self.task))
        self.t_start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        t = time.time() - self.t_start
        print('DONE with {} in {:.4f} seconds'.format(self.task, t))