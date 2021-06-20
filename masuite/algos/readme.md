# Algos

Algorithms are responsible for tracking environment episodes and experiment epochs and taking a learning step when appropriate. Algorithms interact directly with agents. Each algorithm must be passed the following required parameters on creation:

* `agents list(agents)`: a list of `Agent` instances defining the agents responsible for choosing actions.

* `shared_state bool`: whether or not env observations are unique to each agent, or a single state is seen by each agent.

* `n_players int`: number of players interacting with the environment (this should equal the lenght of `agents`).

* `batch_size int`: number of steps in a single epoch.

Additionally, there will likely be other required algorithm-specific parameters. Consult the corresponding algorithm's docstring to identify these.

Algorithms have two required functions:

* `update(self, obs, acts, rews, done)`: updates the algorithm with the given information. If `done = True` the algorithm also ends the episode and determines if the epoch should be ended as well (based on `batch_size`). If the epoch is ended, information about that epoch is returned (in dictionary form). Otherwise, returns `None`.

* `get_agent_params(self, copy: bool=True)`: returns the current agent action policy parameters. This is used for checkpointing purposes.

## Loading an Algorithm

To intialize an algorithm import the class and create an instance of that class:

```python
from masuite.algos.pytorch.simmple_pg import SimplePG

# initialize the list of agents (agents) and an environment instance (env)

alg = SimplePG(
    agents=agents,
    shared_state=env.shared_state,
    n_players=n_players,
    batch_size=batch_size,
    **kwargs
)
```

## Interacting with an algorithm

Each step, the algorithm's update function is called. Update internally decides what other steps to take such as ending an episode or epoch:

```python
for episode in range(20):
    obs = env.reset()
    while True:
        # choose action(s)
        obs, rews, done, info = env.step(acts)
        batch_info = alg.update(obs, acts, rews, done)
        if batch_info is not None: # epoch is over
            # determine whether a checkpoint should be logged
            if checkpoint_due:
                curr_params = alg.get_agent_params()
                # log curr_params
            break
        if done:
            obs = env.reset()

        if done: break
```
