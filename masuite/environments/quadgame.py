
class QuadraticGameTwoPlayer():
  def __init__(self, num_actions):
    na1, na2 = num_actions
    self.A = np.eye(na1)
    self.B = np.zeros((na1,na2))
    self.C = np.zeros((na2,na1))
    self.D = np.eye(na2)
  
  def step(actions):
    x, y = actions
    fx = (1/2)*x.dot(self.A.dot(x)) + x.dot(self.B.dot(y))
    fy = y.dot(self.C.dot(x)) + (1/2)*y.dot(self.D.dot(y))
    return (fx, fy), {}
  
