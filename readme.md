# masuite
Multi-agent suite with a focus on continuous games. Other names for these are
* differentiable games
* smooth games

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


## Planned features:
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
