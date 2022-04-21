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
        #reward = np.random.gaussian(1, self.probabilities[pulled_arm])  # Bernoulli
        reward = np.random.uniform(0,1)
        #WE HAVE A PROBLEM, WHY WE JUST GET INTEGER VALUES, shouldnt they be between 0 and 1? or just 0/1?
        return reward

