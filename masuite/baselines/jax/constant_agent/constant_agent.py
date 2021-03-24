class ConstantAgent():
  def __init__(self, action_spec, optimizer, rng):
    self._state
    self._forward
    self._rng
    
  def act(self):
    return self._state
  
  def update(self, actions):
    self._state = self._sgd_step(self._state, actions)
    
    
  
