"""Basic test coverage for agent training."""

from masuite.environments.cartpole import CartPoleEnv
from masuite.baselines.pytorch.policy_gradient import PolicyGradient
from bsuite import bsuite
from bsuite.baselines import experiment
from bsuite.baselines.pytorch import policy_gradient

def test_run(self, env_id: str, alg_id: str):
  env = masuite.load_env(env_id=env_id)

  agents = []
  agents.append(policy_gradient.default_agent(
      env.observation_spec(), env.action_spec())
  agents.append(policy_gradient.default_agent(
      env.observation_spec(), env.action_spec())

  alg = masuite.load_alg(alg_id=alg_id, agents=agents)

  experiment.run(
      agents=agents,
      environment=env,
      algorithm=alg,
      num_episodes=5)


if __name__ == '__main__':
  test_run()

