from masuite.environments.base import Environment
import numpy as np

class GridEnv(Environment):
    def __init__(self, mapping_seed, n_players=2):
        self.ball_possesion = np.random.randint(1, n_players+1)

        player1 = GridPlayer()
        player2 = GridPlayer()

    def step(self, acts):
        act1, act2 = acts

        state1, state2 = None, None # TODO: states of both players
        next_state1, _, rew1, info1 = player1.step()
        next_state2, _, rew2, info2 = player2.step()


        # If player 1 goes first
        if next_state1 == state2:
            # player 1 hits player 2
            if ball_possession == 0:
                # player 1 has the ball and ....
                pass # TODO: fill out the logic
        
        return None


class GridPlayer():
    def __init__(self, init_state=None):
        self.moves = ((-1,0), #left
            (0,-1), #down
            (1,0), #right
            (0,1)) #up
        
        self.state = (3, 3)
        self.dim = (5,5)
        

    def step(self, act):
        move = self.moves[act]
        next_state = self.state

        next_state[0] += move[0]
        next_state[1] += move[1]
 
        next_state[0] = max(min(next_state[0], self.dim[0]-1, 0)
        next_state[1] = max(min(next_state[1], self.dim[1]-1, 0)

        rew = 0
        done = False
        info = 0

        return next_state, rew, done, info
