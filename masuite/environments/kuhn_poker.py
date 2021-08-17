from masuite.environments.base import DiscreteEnvironment
from open_spiel.python.games.kuhn_poker import KuhnPokerGame

class KuhnPoker(DiscreteEnvironment):
    n_acts = 2
    def __init__(self, mapping_seed: int):
        self.wrapped_env = KuhnPokerGame()

    def step(self, acts):
        if self.state._next_player == 0:
            self.state._apply_action(acts[0])
            self.state._apply_action(acts[1])
        else:
            self.state._apply_action(acts[1])
            self.state._apply_action(acts[0])
        
        rewards = self.state.returns()
        done = self.state._game_over

        return rewards, done, {}

    def reset(self):
        self.state = self.wrapped_env.new_initial_state()
        
        return (self.state.cards, self.state.bets, self.state.pot), self.state._game_over