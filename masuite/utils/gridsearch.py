from functools import partial, reduce
from itertools import product
import operator
from typing import Mapping, Iterable, Union

import numpy as np

from masuite.agents.base import Agent
from masuite.algos.utils import run
# from masuite.algos.base import Algorithm

class ParamGrid:
    """Grid of parameters used to iterate over all parameter combinations in a
    parameter grid (based on sklearn's param grid"""
    def __init__(self, param_grid: Union[Mapping, Iterable]):
        if isinstance(param_grid, Mapping):
            param_grid = [param_grid]
        
        for grid in param_grid:
            if not isinstance(grid, dict):
                raise TypeError(f"Parameter grid is not a dict ({grid})")
            
            for key in grid:
                if not isinstance(grid[key], Iterable):
                    raise TypeError(f"Parameter grid value is not an iterable ({grid[key]}")
        
        self.param_grid = param_grid
    

    def __iter__(self):
        """Iterate over the points in the grid."""
        for param in self.param_grid:
            items = sorted(param.items()) # for reproducablility
            if not items:
                yield {}
            else:
                keys, values = zip(*items)
                for val in product(*values):
                    params = dict(zip(keys, val))
                    yield params
    

    def __len__(self):
        """Number of points on the grid."""
        # Product function that can handle iterables (np.product can't).
        product = partial(reduce, operator.mul)
        return sum(product(len(v) for v in p.values()) if p else 1
                   for p in self.param_grid)
    

    def __getitem__(self, ind):
        """Get the parameters that would be ``ind``th in iteration
        Parameters
        ----------
        ind : int
            The iteration index
        Returns
        -------
        params : dict of str to any
            Equal to list(self)[ind]
        """
        # This is used to make discrete sampling without replacement memory
        # efficient.
        for sub_grid in self.param_grid:
            # XXX: could memoize information used here
            if not sub_grid:
                if ind == 0:
                    return {}
                else:
                    ind -= 1
                    continue

            # Reverse so most frequent cycling parameter comes first
            keys, values_lists = zip(*sorted(sub_grid.items())[::-1])
            sizes = [len(v_list) for v_list in values_lists]
            total = np.product(sizes)

            if ind >= total:
                # Try the next grid
                ind -= total
            else:
                out = {}
                for key, v_list, n in zip(keys, values_lists, sizes):
                    ind, offset = divmod(ind, n)
                    out[key] = v_list[offset]
                return out

        raise IndexError('ParameterGrid index out of range')


class GridSearch:
    def __init__(self,
        masuite_id: str,
        AgentClass: Agent,
        AlgClass,
        run_fn,
        param_grid: Union[Mapping, Iterable],
        score_fn,
        log_to_terminal: bool=False,
        seed: bool=True,
        overwrite: bool=False
    ) -> None:
        self.masuite_id = masuite_id
        self.AgentClass = AgentClass
        self.AlgClass = AlgClass
        self.run_fn = run_fn
        self.param_grid = ParamGrid(param_grid)
        self.score_fn = score_fn
        self.log_to_terminal = log_to_terminal
        self.seed = seed
        self.overwrite = overwrite
    
    
    def run_search(self):
        print(f"Running gridsearch over {len(self.param_grid)} parameter combinations")
        log_files = list()
        for i, params in enumerate(self.param_grid):
            print(f"{i+1}: Running {self.masuite_id} with params {params}")
            run_info = self.run_experiment(params)
            log_files.append(run_info["log_save_path"])
        self.log_files = log_files
        print(f"Gridsearch complete, experiments run: {len(self.param_grid)}")
    

    def run_experiment(self, params):
        run.args.__setattr__("log_params", True)
        run.args.__setattr__("overwrite", self.overwrite)
        run.args.__setattr__("seed", False)
        safe_masuite_id = self.masuite_id.replace("/", "")
        run.args.__setattr__("save_path", f"tmp/{safe_masuite_id}-gridsearch")
        # set the parameter values
        for key, value in params.items():
            run.args.__setattr__(key, value)
        run_info = self.run_fn(self.masuite_id, self.AgentClass, self.AlgClass, self.log_to_terminal)
        return run_info
        