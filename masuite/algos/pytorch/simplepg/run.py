from masuite.agents.pytorch.policy_gradient.agent import PGAgent
from masuite.algos.pytorch import SimplePG
from masuite.algos.utils import run


if __name__ == '__main__':
    masuite_id = "cartpole_simplepg/0" # default experiment
    AgentClass = PGAgent
    AlgClass = SimplePG
    run.run_pg_experiment(masuite_id, AgentClass, AlgClass)