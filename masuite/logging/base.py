import os
from abc import ABCMeta, abstractproperty
import pandas as pd
from masuite import sweep

SAFE_SEP = '-'
INITIAL_SEP = '__'
MASUITE_PREFIX = 'masuite_id' + INITIAL_SEP

def create_params_str(params: dict):
    params_str = ''
    for key, value in params.items():
        params_str += f'{key}{value}_' 
    return params_str


def create_checkpoint_file(safe_masuite_id: str, results_dir: str, params_str=None):
    if params_str:
        checkpoint_file_name = f'{safe_masuite_id}_{params_str}_checkpoints.csv'
    else:
        checkpoint_file_name = f'{safe_masuite_id}_checkpoints.csv'
    checkpoint_save_path = os.path.join(results_dir, checkpoint_file_name)

    return checkpoint_save_path
    

class Logger(metaclass=ABCMeta):
    @abstractproperty
    def write(self, data: dict):
        raise NotImplementedError
    
    def write_checkpoint(self, params: dict):
        self.checkpoint_data.append(params)
        df = pd.DataFrame(self.checkpoint_data)
        df.to_csv(self.checkpoint_save_path, index=False)
