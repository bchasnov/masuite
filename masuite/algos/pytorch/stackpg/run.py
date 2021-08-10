from masuite.agents.pytorch.policy_gradient import PGAgent
from masuite.algos.pytorch import StackPG
from masuite.algos.utils import run

if __name__ == "__main__":
    masuite_id = "smallsoccer/0" # default experiment
    AgentClass = PGAgent
    AlgClass = StackPG
    run.run_discrete_pg_experiment(masuite_id, AgentClass, AlgClass)