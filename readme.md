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
We recommend installing a virtual environment. If the below code does not work, try "python" for "python3 -m venv masuite"

```
cd /path/to/masuite/
python3 -m venv masuite
source masuite/bin/activate
```

If you are simply going to run masuite experiments, or use already implemented tools, install using:

```
pip install .
```

Alternatively, if you are planning on editing/developing masuite source code, it is recommended to install in development mode using:

```
python3 setup.py develop
```


### Command line
To confirm the installation was successful run:
```
pytest --disable-warnings
```
Do not worry about the warnings that are encountered as they come from the gym dependency.

(Note: this is a WIP feature that only tests that the package is installed and some basic tests can be run. More verbose testing still needs to be written).

## Agents
Agents are responsible for learning an action policy and choosing actions given a set of _observations_. Each agent instance must be passed the following parameters on creation:
* `env_dim list(int)` - defines the shape of the observations the agent will be passed when choosing an action.
* `act_dim list(int)` - defines the shape of the actions the agent will return.

In addition, an optional `lr` parameter can be passed specifying the _learning rate_ for the agent when updating its action policy.

Each agent must also have two public functions:
* `select_action(self, obs)` - chooses what actions to take based on the current action policy and the passed observations.
* `update(self, grad)` - makes a single action policy update based on `grad` and the learning rate.


## Experiments

Masuite has three main modules that when combined create an experiment. The three modules are:

1. Environments: The environment agents interact with.
2. Agents - Responsible for choosing actions to pass to the environment based on some action policy.
3. Algos - Responsible for choosing updates for agents' action policies and passing these updates to the agent(s).

Additionally, experiements are generally initialized with logging to track progress and checkpoints.

Experiements are usually run from the command line. When this is the case, the used algorithm's `run.py` file is called with various (optional) command args. The main argument you should be aware of is the `masuite-id`. This specifies the name of the experiment to run as well as which configuration to run. The general form is `experiment_name/config_num`. The configurations (and their corresponding `config_num`) can be found in `experiments/experiment_name/sweep.py`.

To run the single-player simple pg cartpole experiment run:

```
python3 masuite/algos/pytorch/simplepg/run.py --masuite-id=cartpole_simplepg/0
```
Additional optional command line args can be found in `run.py`. If 'python3' doesn't work try 'python'

To run the two-player simple pg smallsoccer experiment run:

```
python3 masuite/experiments/smallsoccer_gridsearch.py
```

## Tournaments

After training agents and extracting their checkpoint information, you can compare the quality of agents in a tournament game.

To run a tournament, first activate Masuite, then type the code below into the terminal.
```
python3 masuite/analysis/tournament_demo.py masuite/analysis/soccer_checkpoints out.pkl
```
* "Python3" is to open the python file
* "masuite/analysis/tournament_demo.py" is the file that has the tournament information in it.
* "masuite/analysis/soccer_checkpoints" is the folder where the checkpoints of the agents are in
* "out.pkl" is the output pickle file name that can be opened in python and get the info from the tournament.

### Loading an environemnt

```python
import masuite

env = masuite.load_from_id('env_id')
```

### Interacting with an environment

Example run loop for 2 agents (no state)

```python
def sample(env, agent1, agent2):
  x = agent1.act()  
  y = agent2.act()
  fx, fy = env.step((x,y))
  return fx, fy
```

Example run loop for 2 agents in an environment (with state)

```python
def sample(env, agent1, agent2)
  rets = []
  for _ in range(env.num_episodes):
    obs = env.reset() # resets environment to initial state
    while not done:
      act1 = agent1.act(obs)
      act2 = agent2.act(obs)
      obs, rews, done, info = env.step((act1, act2))
      rets.append(rews)
  return rets 
```

## Algorithms

### Loading an algorithm
```python
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

## General Experiment Runflow
### Initialization
1. Create `Environment` instance
2. Create `Logger` instance
3. Create `Agent` instance(s)
4. Create `Algo` instance
5. Pass to `experiment.run()`

### Training Loop
```
for epoch in range(num_epochs):
    obs <- env.reset()
    while True:
        acts <- agent.select_action()
        obs, rews, done <- env.step(acts)
        batch_info <- alg.update(obs, acts, rews, done)
        if batch is over:
            break
```
