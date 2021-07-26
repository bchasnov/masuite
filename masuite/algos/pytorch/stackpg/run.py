from torch.serialization import STORAGE_KEY_SEPARATOR
from masuite.algos.pytorch.simplepg.run import AgentClass, AlgClass
from masuite.agents.pytorch.policy_gradient import PGAgent
from masuite.algos.pytorch import StackPG
from masuite.algos.utils import run

if __name__ == "__main__":
    masuite_id = "soccer_stackpg/0" # default experiment
    AgentClass = PGAgent
    AlgClass = StackPG
    run.run_pg_experiment