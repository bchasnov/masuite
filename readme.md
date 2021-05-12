# Multi-agent suite for games optimization (`masuite`)
## Introduction
The multi-agent suite is a collection of experiments that investigate the capabilities of learning in continuous games (also called differentiable or smooth games).

## Technical overview
Experiments are defined in the `experiments` folder. Each folder inside corresponds to one experiment and contains:
* A file that defines an environment, with a level of configurability (python file with the same name as the experiment).
* A sequence of keyword arguments for this environment, defined in `SETTINGS` variable in the experiment's `sweep.py` file
* A file `analysis.py` that defines the plotting and analysis tools (WIP).
Logging is done inside an environment, which allows for data to be output in the correct format regardless of the agent or algorithm structure.

## Getting Started
The main way to use the package is through the command line or juptyer notebooks.

### Installation
We recommend installing a virtual environment.

```
python3 -m venv masuite
source masuite/bin/activate
pip install .
```

### Command line
To confirm the installation was successful run:
```
pytest --disable-warnings
```
Do not worry about the warnings that are encountered as they come from the gym dependency.


### Notebooks
For example notebook usage, see the jupyter notebooks in `masuite/notebooks`.

## Environments
The environments define the environment dynamics agents interact with. All environments inherit from `masuite.environments.base.Environment`. Each environment has at least the following parameters:
* `mapping_seed: int` - defines the seed for any randomness in the enviornment.
* `n_players: int` - defines the number of players that interact with the environment.
* `env_dim: list(int)` - defines the shape of the observations the environment makes public to learning agents.
* `act_dim: list(int)` - defines the shape of the actions to be inputted by a single learning agent.
* `shared_state: bool` - defines whether or not a single state is shared between each learning agent, or if each agent has its own state.

In addition, each environment has two public functions:
* `step(self, actions)` - executes a single timestep transition given actions (a list of all agents' chosen actions for that timestep). `step` returns the resulting state/observations, the calculated reward for each agent based on their actions and the resulting state, a boolean indicating whether the environment is "done" meaning it needs to be reset, and a dictionary containing any other info that could be interesting/useful.
* `reset(self)` - resets the environment to initial state and returns the initial state/observations.

### Loading an environment
Environments are specificed by an `env_id` string. The string consists of the environment name followed by an integer specifying the random `mapping_seed` in the following format:
```
import masuite

env = masuite.load_env('cartpole/0')
```

The above returns a raw environment instance.

## Agents
Agents are responsible for learning an action policy and choosing actions given a set of _observations_. Each agent instance must be passed the following parameters on creation:
* `env_dim list(int)` - defines the shape of the observations the agent will be passed when choosing an action.
* `act_dim list(int)` - defines the shape of the actions the agent will return.

In addition, an optional `lr` parameter can be passed specifying the _learning rate_ for the agent when updating its action policy.

Each agent must also have two public functions:
* `select_action(self, obs)` - chooses what actions to take based on the current action policy and the passed observations.
* `update(self, grad)` - makes a single action policy update based on `grad` and the learning rate.


## Experiments


### Loading an environemnt with logging
```
import masuite

env = masuite.load_and_record('cartpole/0', save_path='/path/to/results')
```

### Interacting with an environment
Example run loop for 2 agents (no state)
```
def sample(env, agent1, agent2):
  x = agent1.act()  
  y = agent2.act()
  fx, fy = env.step((x,y))
  return fx, fy
```

Example run loop for 2 agents in an environment (with state)
```
def sample(env, agent1, agent2)
  rets = []
  for _ in range(env.num_episodes):
    obs = env.reset()
    while not done:
      act1 = agent1.act(obs)
      act2 = agent2.act(obs)
      obs, rews, done, info = env.step((act1, act2))
      rets.append(rews)
  return rets 
```

## Algorithms
### Loading an algorithm
```
import masuite

alg = masuite.load_alg('simgrad')
```

### Loading an algorithm with logging

```
import masuite

alg = masuite.load_alg_and_record('stackgrad/reg', results_dir='/path/to/results')
```

### Training agents
For simultaneous gradient descent
```
def train(x, y):
  fx, fy = sample(x, y)
  gx = alg.grad(fx, x)  
  gy = alg.grad(fy, y)
  info = alg.step((gx, gy))
  return info
```

For stackelberg gradient descent
```
def train(x, y):
  fx, fy = sample(x, y)
  gy = alg.grad(fy, y)
  gx = alg.grad(fx, x, gy)
  info = alg.step((gx, gy))
  return info
```

## Baseline agents
We include implementations of several common agents in the `baselines` directory

## Planned features
### environments
Environments model cost/reward functions and state transitions.
* Matrix game (numpy, jax)
* Bimatrix game (numpy, jax)
* Quadratic game (numpy, jax, pytorch)
* Linear dynamics quadratic game (numpy, jax, pytorch)
* Multi-agent optimal control (cvxpy, julia)

### agents
Agents hold parameters and decision-making logic. They can have limited or full knowledge of the system. 
* Actor: policy gradient 
* Implicit actor: stackelberg gradient
* Cricic: stackelberg actor-critic 

### algorithms
Algorithms combine agents by transfering necessary information to one another (if they need to coordinate) 
* Simultaneous (using optimizers)
* Stackelberg (using optimizers)
* Symplectic gradient adjustment

### experiments
Experiments combine environemnts, agents and algorithms together.
* Run stackeblerg update in quadratic games.
* Run policy gradient in linear quadratic games.
* Run policy gradient in markov games.
* Run stackelberg policy gradient in linear quadratic games.
