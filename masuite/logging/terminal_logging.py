import os
from typing import Mapping, Any
from masuite import sweep
from masuite.logging import base

class TerminalLogger(base.Logger):
    def __init__(self,
        masuite_id: str,
        results_dir: str,
        overwrite:bool = False,
        log_checkpoints:bool = False,
        params:dict = None
    ):
        self.masuite_id = masuite_id
        if log_checkpoints:
            if not os.path.exists(results_dir):
                try:
                    os.makedirs(results_dir)
                except OSError:
                    pass
        
            safe_masuite_id = masuite_id.replace(sweep.SEP, base.SAFE_SEP)
            if params is not None:
                params_str = base.create_params_str(params)
            else:
                params_str = None
            
            self.checkpoint_save_path = base.create_checkpoint_file(
                safe_masuite_id,
                results_dir,
                params_str
            )

            if os.path.exists(self.checkpoint_save_path) and not overwrite:
                raise ValueError(
                    f'File {self.checkpoint_save_path} already exists. Specify a different '
                    'directory, or set overwrite=True to overwrite existing data.'
                )

            self.checkpoint_data = []
            self.log_save_path = None
            
    
    def write(self, data: Mapping[str, Any]):
        print(f'[{self.masuite_id}] {data}')

