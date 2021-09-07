from masuite.agents.pytorch.policy_gradient.agent import PGAgent
from masuite.algos.pytorch.simplepg import SimplePG
from masuite.algos.pytorch.stackpg import StackPG
from masuite.algos.utils.run import run_discrete_pg_experiment
from masuite.utils.gridsearch import GridSearch
from masuite.experiments.smallsoccer.analysis import simple_score

MASUITE_ID = 'soccer/0'
PARAMS = dict(
  lr=[1e-1, 1e-2, 1e-3, 1e-4],
  num_epochs=[100],
  batch_size=[1000, 5000, 1000, 20000],
  hidden_sizes=[[32], [64], [128], [128, 64, 32], [256, 128, 64]]
)

simple_gs = GridSearch(
  masuite_id=MASUITE_ID,
  AgentClass=PGAgent,
  AlgClass=SimplePG,
  run_fn=run_discrete_pg_experiment,
  param_grid=PARAMS,
  score_fn=simple_score,
  save_path='tmp/soccer-gs',
  overwrite=True,
)

stack_gs = GridSearch(
  masuite_id=MASUITE_ID,
  AgentClass=PGAgent,
  AlgClass=StackPG,
  run_fn=run_discrete_pg_experiment,
  param_grid=PARAMS,
  score_fn=simple_score,
  overwrite=True
)

if __name__ == '__main__':
    simple_gs.run_search()
    stack_gs.run_search()