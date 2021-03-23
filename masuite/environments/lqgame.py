#!/usr/bin/python3
import torch
import numpy as np
import gym
from gym import spaces

def _value_error(param_name):
    """
    Raises a value error for missing LQGame class parameters specified by
    param_name

    @param param_name string: name of parameter that is missing
    """
    raise ValueError(f'{param_name} must be included in params dict')


class LinearQuadraticGame(gym.Env):
    """
    Linear Quadratic Game for n-players
    
    Every player has same dimension of actions.

    Parameters:
    -@param params: **params: a combination of some (or none) of the following:
     Required:
        - states: initial state of the game
        - A: state transition matrix
        - Bs: control input matrices (1 for each player)
        - Qs: reward transition matrix
        - Rs: reward weights for each player. Must have the shape
            |n_players|x|n_players|
            observations
     Optional:
        - n_players: number of players in the game
        - n_states: number of states in the game
        - n_actions: number of actions each player takes each step
        - noise_state: standard deviation of noise added to state.
        - action_space: Space object corresponding to valid actions
        - observation_space: Space observation corresponding to valid
            and size as high. Defaults to -1*high if not specified.

    Additional class parameters:
        - rewards: current cumulative rewards for each player. Initialized to 0s
    """

    def __init__(self, **params):
        #TODO: decide if these 4 params should be required, or the function setting defaults is ok
        self.n_players = params['n_players'] if 'n_players' in params else 1
        self.n_states = params['n_states'] if 'n_states' in params else 1
        self.n_actions = params['n_actions'] if 'n_actions' in params else 1
        self.clip_states = params['clip_states'] if 'clip_states' in params else 1000

        if 'states' in params:
            self.states = params['states']
            # store the intial state for use in reset()
            self.initial_states = params['states']
        else:
            _value_error('states')

        self.rewards = torch.zeros(self.n_players, 1)
        self.noise_state = params['noise_state'] if 'noise_state' in params else 1e-2

        if 'A' in params:
            self.A = params['A']
        else:
           _value_error('A matrix') 


        if 'Bs' in params:
            self.Bs = params['Bs']
        else:
            _value_error('B matrices')

        if 'Qs' in params:
            self.Qs = params['Qs']
        else:
            _value_error('Q matrices')

        if 'Rs' in params:
            self.Rs = params['Rs']
        else:
            _value_error('R matrices')

        if 'action_space' in params:
            err_msg = 'action_space must be a subclass of spaces.Space'
            assert issubclass(type(params['action_space']), spaces.Space), err_msg
            self.action_space = params['action_space']
        else:
            vals = np.ones(self.n_players) * float('inf')
            self.action_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(self.n_players, self.n_actions, 1))
        
        if 'observation_space' in params:
            err_msg = 'observation_space must be a subclass of spaces.Space'
            assert isinstance(params['observation_space'], spaces.Space)
            self.observation_space = params['observation_space']
        else:
            vals = np.ones(len(self.states.flatten())) * float('inf')
            self.observation_space = spaces.Box(-vals, vals)


    def __str__(self):
        text = f'===== current states =====\n{self.states.numpy()}\n'
        text += f'===== current rewards =====\n{self.rewards.numpy()}\n'
        return text
        

    def step(self, actions, reward_offset=0):
        """Advance the state of the game by one step based on actions

        @param actions: torch.tensor of size |self.n_players|x|self.n_actions|x|1|

        @return (torch.FloatTensor, torch.FloatTensor): updated states, each
            player's reward for this step, whether the game is "done" 
            (currently always False), and "info" (see gym documentation for
            details, currently an empty dict)
        """
        # vpg sends
        if type(actions) == np.ndarray:
            actions = torch.FloatTensor(actions)

        err_msg = 'actions must be in the game\'s action space'
        if actions.shape != torch.Size([self.n_players, self.n_actions]): print(actions)
        assert actions.shape == torch.Size([self.n_players, self.n_actions]), err_msg
        '''update states, see writeup.md for equation'''
        self.states = (self.A@self.states).flatten()
        for i in range(self.n_players):
            # Noise is added in the policy
            noise = torch.empty(len(self.states.flatten())).normal_(mean=0, std=self.noise_state)
            self.states += self.Bs[i]@actions[i] + noise
                
        if np.any(np.abs(self.states.detach().numpy()) > self.clip_states):
            print("Warning: the state is saturating. The system might be unstable.")
            self.states = torch.clamp(self.states, min=-self.clip_states, max=self.clip_states)
        '''update rewards, see writeup.md for equation'''
        # updating player i's reward
        # r1 = -x.T@Q1@x-u1.T@R11@u1+u2.T@R12@u2
        for i in range(self.n_players):
            if self.Qs is not None:
                self.rewards[i] = -(self.states.t()@self.Qs[i]@self.states)
            else:
                self.rewards[i] = -(self.states.t()@self.states)
            # with respect to player j's actions
            for j in range(self.n_players):
                r0 = actions[j].t()@actions[j]
                if self.Rs is not None:
                    #TODO: better way to check this
                    if (actions[j].t()@self.Rs[i][j]).size() == torch.Size([]):
                        prod = torch.Tensor([actions[j].t()@self.Rs[i][j]])
                        action_rew_update = -(prod@actions[j])
                    else:
                        action_rew_update = -(actions[j].t()@self.Rs[i][j]@actions[j]).flatten()
                else:
                    assert "Not Supported"
                self.rewards[i] += action_rew_update

        return self.states.flatten(), self.rewards, False, {}
    
    
    def reset(self):
        """
        Reset the states of the game and player rewards
        """
        self.states = self.initial_states
        self.rewards = torch.zeros(self.n_players, 1)
        return self.states
