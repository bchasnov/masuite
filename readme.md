# Multi-agent suite for games optimization (`masuite`)
## Introduction
The multi-agent suite is a collection of experiments that investigate the capabilities of learning in continuous games (also called differentiable or smooth games).

## Technical overview
Experiments are defined in the `experiments` folder. Each folder inside corresponds to one experiment and contains:
* A file that defines an environment, with a level of configurability.
* A sequence of keyword arguments for this environment, defined in `CONFIG` variable in the experiment's `config.py` file
* A file `analysis.py` that defines the plotting and analysis tools.
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
To confirm the installation was successful, run:
```
python tests
```

### Notebooks
For example usage, see
the jupyter notebooks in `masuite/notebooks`.
You can open it 

### Installation


## Environments

The environments are defined by a set of functions `(f1, f2, ..., fn)` where `fi` is the `i`th agent's cost/reward. 
The costs/rewards of each agent are dependent on the shared state, its own action and the actions of others.

The case where agents are coupled through costs and dynamics are the emphasized.
Those are cases where the gradient of `fi` with respect the `j` agent's action (`j!=i`) is non-zero in general.
We can show this concept with an example.

**Example:** Here is a simple quadratic game defined by the pair of costs `(fx, fy)`
```
fx(x,y) = (1/2) Ax^2 + Bxy
fy(x,y) = Cyx + (1/2) Dy^2
```
We use `𝛅z` as the gradient operator with respect to `z`.
Then
```
𝛅x(fx)(x,y) = Ax + By
𝛅y(fy)(x,y) = Cx + Dy
```
The fact that `B` and `C` are non-zero is important when considering the gradient of agents in games. These terms do not show up 
in single agent learning problems.

### Loading an environment
Environments are specificed by a `masuite_id` string. 
```
import masuite

env = masuite.load_env('lqgame/zs/0')
```

### Loading an environemnt with logging
```
import masuite

env = masuite.load_and_record('lqgame/zs/0', save_path='/path/to/results')
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
