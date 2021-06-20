# Agents
Agents are responsible for learning an action policy and choosing actions given a set of _observations_. Each agent instance must be passed the following parameters on creation:
* `env_dim list(int)` - defines the shape of the observations the agent will be passed when choosing an action.
* `n_acts list(int)` - defines the possible actions the agent can choose.

In addition, an optional `lr` parameter can be passed specifying the _learning rate_ for the agent when updating its action policy.

Each agent must also have two public functions:
* `select_action(self, obs)` - chooses what actions to take based on the current action policy and the passed observations.
* `update(self, grad)` - makes a single action policy update based on `grad` and the learning rate.

Agents are usually influenced by an algorithm when updating the action policy.

## Creating an Agent

To initialize an agent import the class and create an instance of the class:

```python
from masuite.agents.pytorch.policy_gradient.agent import PGAgent

agent = PGAgent(env_dim, n_acts, lr) # lr is optional
```

## Interacting with an agent

An agent receives the current environment observations and uses them to choose an action. The agent's action policy is updated when `update()` is called. Note that the update function is usually called internally by an algorithm when appropriate.

```python
for episode in range(20):
    obs = env.reset()
    while True:
        act = agent.select_action(obs)
        obs, rews, done, info = env.step(act)

        if done: break
    # if not running an algorithm
    grads = determine_grad_update_somehow()
    agent.update(grads)
```
