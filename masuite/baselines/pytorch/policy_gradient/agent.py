class PolicyGradient(lr):
  pass


def default_agent(obs_spec, act_spec, seed=0):
  return PolicyGradient()
