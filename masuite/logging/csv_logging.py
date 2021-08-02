import os
from typing import Mapping, Any
import pandas as pd
from masuite import sweep
from masuite.logging import base




class CSVLogger(base.Logger):
    """
    Saves data to a CSV file via Pandas

    """
    def __init__(self,
        filename: str,
        results_dir: str= '/tmp/masuite',
        overwrite:bool =False,
        log_checkpoints: bool=False,
        params: dict=None
    ):
        if not os.path.exists(results_dir):
            try:
                os.makedirs(results_dir)
            except OSError:
                pass
        
        safe_filename = filename.replace(sweep.SEP, base.SAFE_SEP)
        if params is not None:
            params_str = base.create_params_str(params)
        else:
            params_str = None
        
        if params_str:
            log_filename = f'{safe_filename}_{params_str}.csv'
        else:
            log_filename = f'{safe_filename}.csv'
        
        if log_checkpoints:
            self.checkpoint_save_path = base.create_checkpoint_file(
                safe_filename,
                results_dir,
                params_str
            )
            print(f"Logging agent checkpoints to file: {self.checkpoint_save_path}")
            if os.path.exists(self.checkpoint_save_path) and not overwrite:
                raise ValueError(
                    f'File {self.checkpoint_save_path} already exists. Specify a different '
                    'directory, or set overwrite=True to overwrite existing data.'
                )
        
        log_save_path = os.path.join(results_dir, log_filename)
        print(f"Logging to file: {log_save_path}")

        if os.path.exists(log_save_path) and not overwrite:
            raise ValueError(
                f'File {log_save_path} already exists. Specify a different '
                'directory, or set overwrite=True to overwrite existing data.'
            )
            

        self.data, self.checkpoint_data = [], []
        self.log_save_path = log_save_path


    def write(self, data: Mapping[str, Any]):
        self.data.append(data)
        df = pd.DataFrame(self.data)
        df.to_csv(self.log_save_path, index=False)
