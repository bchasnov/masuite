import os
from typing import Mapping, Any
import pandas as pd
from masuite import environments
from masuite.logging import base
from masuite.utils import wrappers

BAD_SEP = '/'
SAFE_SEP = '-'
INITIAL_SEP = '_-_'
MASUITE_PREFIX = 'masuite_id' + INITIAL_SEP

def wrap_environment(env: environments.Environment,
                     masuite_id: str,
                     results_dir: str,
                     overwrite: bool=False,
                     log_by_step: bool=False)->environments.Environment:
    """
    Returns a wrapped logging environment that logs to CSV
    """
    logger = Logger(masuite_id, results_dir, overwrite)
    return wrappers.Logging(env, logger, log_by_step=log_by_step)


class Logger(base.Logger):
    """
    Saves data to a CSV file via Pandas

    """
    def __init__(self,
                 masuite_id: str,
                 results_dir: str= '/tmp/masuite',
                 overwrite:bool=False):
        
        if not os.path.exists(results_dir):
            try:
                os.makedirs(results_dir)
            except OSError:
                pass

        safe_masuite_id = masuite_id.replace(BAD_SEP, SAFE_SEP)
        filename = f'{MASUITE_PREFIX}{safe_masuite_id}.csv'
        save_path = os.path.join(results_dir, filename)

        if os.path.exists(save_path) and not overwrite:
            raise ValueError(
                f'File {save_path} already exists. Specify a different '
                'directory, or set overwrite=True to overwrite existing data.'
            )

        self.data = []
        self.save_path = save_path

    def write(self, data: Mapping[str, Any]):
        self.data.append(data)
        df = pd.DataFrame(self.data)
        df.to_csv(self.save_path, index=False)
