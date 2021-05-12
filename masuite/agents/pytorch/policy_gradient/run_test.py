"""Basic test coverage for agent training."""
from masuite import masuite
from masuite.agents import experiment
from masuite.agents.pytorch import policy_gradient
from masuite.algos.pytorch.simple_pg.simple_pg import SimplePG

# def test_run(masuite_id: str):
def test_run():
    masuite_id = 'cartpole_simplepg/0'
    env = masuite.load_from_id(masuite_id=masuite_id)
    agent = [policy_gradient.default_agent(env.env_dim, env.act_dim)]
    alg = SimplePG(
        agents=agent,
        obs_dim=env.env_dim,
        act_dim=env.act_dim,
        shared_state=env.shared_state,
        n_players=env.n_players,
        batch_size=1,
    )

    experiment.run(
        alg=alg,
        env=env,
        num_epochs=1,
        batch_size=1
    )


if __name__ == '__main__':
    test_run()
    # test_run('cartpole_simplepg/0')

