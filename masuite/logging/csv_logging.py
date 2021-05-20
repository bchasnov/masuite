import os
from typing import Mapping, Any
import pandas as pd
from masuite import sweep
from masuite import environments
from masuite.logging import base
from masuite.utils import wrappers

SAFE_SEP = '-'
INITIAL_SEP = '__'
MASUITE_PREFIX = 'masuite_id' + INITIAL_SEP


class Logger(base.Logger):
    """
    Saves data to a CSV file via Pandas

    """
    def __init__(self,
        masuite_id: str,
        results_dir: str= '/tmp/masuite',
        overwrite:bool=False
    ):
        
        if not os.path.exists(results_dir):
            try:
                os.makedirs(results_dir)
            except OSError:
                pass

        safe_masuite_id = masuite_id.replace(sweep.SEP, SAFE_SEP)
        log_filename = f'{MASUITE_PREFIX}{safe_masuite_id}.csv'
        checkpoint_file_name = f'{MASUITE_PREFIX}{safe_masuite_id}_checkpoints.csv'
        log_save_path = os.path.join(results_dir, log_filename)
        checkpoint_save_path = os.path.join(results_dir, checkpoint_file_name)

        if os.path.exists(log_save_path) and not overwrite:
            raise ValueError(
                f'File {log_save_path} already exists. Specify a different '
                'directory, or set overwrite=True to overwrite existing data.'
            )
        if os.path.exists(checkpoint_save_path) and not overwrite:
            raise ValueError(
                f'File {checkpoint_save_path} already exists. Specify a different '
                'directory, or set overwrite=True to overwrite existing data.'
            )
            

        self.data, self.checkpoint_data = [], []
        self.log_save_path = log_save_path
        self.checkpoint_save_path = checkpoint_save_path


    def write(self, data: Mapping[str, Any]):
        self.data.append(data)
        df = pd.DataFrame(self.data)
        df.to_csv(self.log_save_path, index=False)


    def write_checkpoint(self, params: dict):
        self.checkpoint_data.append(params)
        df = pd.DataFrame(self.checkpoint_data)
        print(df.head())
        df.to_csv(self.checkpoint_save_path, index=False)