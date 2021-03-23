# Environments

There are the supported multi-agent environments:

- [x] Linear quadratic game

  Linear Quadratic (LQ) game dynamics. Child class of [OpenAI's gym.Env](https://github.com/openai/gym/blob/c8a659369d98706b3c98b84b80a34a832bbdc6c0/gym/core.py#L8). The state update is defined as:
  
  ![equation](https://latex.codecogs.com/gif.latex?s_%7Bt&plus;1%7D%20%3D%20A%20%5Ccdot%20s_t%20&plus;%20%28%5Csum_%7Bi%3D1%7D%5E%7Bn%5C_players%7D%20B_i%20%5Ccdot%20a_i%29%20&plus;%20%5Comega)
  
  Player 1's reward in a two-player game is defined as:
  
  ![equation](https://latex.codecogs.com/gif.latex?r_1%20%3D%20-x%5ET%20%5Ccdot%20Q_1%20%5Ccdot%20x%20-%20a_1%5ET%20%5Ccdot%20R_%7B11%7D%20%5Ccdot%20a_1%20&plus;%20a_2%5ET%20%5Ccdot%20R_%7B12%7D%20%5Ccdot%20a_2)
  
- [ ] Finite matrix game
- [ ] Markov Soccer
- [ ] Cartpole 
- [ ] Prisoner's dilemma
- [ ] Bimatrix games
- [ ] ...
