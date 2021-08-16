from masuite.agents.pytorch.policy_gradient.agent import PGAgent
from masuite.algos.pytorch.simplepg import SimplePG
from masuite.algos.utils import run
from masuite.utils.gridsearch import GridSearch

param_grid = dict(
    lr=[1e-1, 1e-2, 1e-3, 1e-4],
    batch_size=[100, 500, 1000, 5000],
    num_epochs=[50, 100, 500],
)

searcher = GridSearch(
    masuite_id="smallsoccer/0",
    AgentClass=PGAgent,
    AlgClass=SimplePG,
    run_fn=run.run_discrete_pg_experiment,
    param_grid=param_grid,
    score_fn=None
)

searcher.run_search()