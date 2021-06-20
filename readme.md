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
cd /path/to/masuite/dir/
python3 -m venv masuite
source masuite/bin/activate
```

If you are simply going to run masuite experiments, or use already implemented tools, install using:

```
pip install .
```

Alternatively, if you are planning on editing/developing masuite source code, it is recommended to install in development mode using:

```
python3 setup.py --develop
```


### Command line
To confirm the installation was successful run:
```
pytest --disable-warnings
```
Do not worry about the warnings that are encountered as they come from the gym dependency.
(Note: this is a WIP feature that only tests that the package is installed and some basic tests can be run. More verbose testing still needs to be written.)

### Notebooks
For example notebook usage, see the jupyter notebooks in `masuite/notebooks`.



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
