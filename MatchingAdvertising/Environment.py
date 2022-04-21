import numpy as np


# The environment is defined by
#   A number of arms
#   A probability distribution for each arm and reward function
# The environment interacts with the learner
# by returning a stochastic reward depending on the pulled arm

class Environment:
    def __init__(self, n_arms, probabilities):
        self.n_arms = n_arms
        self.probabilities = probabilities

    def round(self, pulled_arm):
        # Binomial distribution with a single trial N=1 is a Bernoulli distribution
        reward = np.random.binomial(1, self.probabilities[pulled_arm])
        return reward
