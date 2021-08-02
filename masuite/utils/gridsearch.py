from typing import Dict, List

from masuite.agents.base import Agent
# from masuite.algos.base import Algorithm

class GridSearch:
    def __init__(self,
        masuite_id: str,
        AgentClass: Agent,
        AlgClass,
        run_fn: function,
        param_grid: Dict[str, List],
    ) -> None:
        self.masuite_id = masuite_id
        self.AgentClass = AgentClass
        self.AlgClass = AlgClass
        self.run_fn = run_fn
        self.param_grid = param_grid
        pass