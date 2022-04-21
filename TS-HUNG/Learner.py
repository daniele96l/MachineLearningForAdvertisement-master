import numpy as np


# A learner object is defined by
#   - the number of arms that he can pull
#   - the current round
#   - the list of the collected rewards
#
# The learner interacts with the environment by selecting the arm to pull
# at each round and observing the reward given by the environment

class Learner:

    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.t = 0
        self.rewards_per_arm = x = [[] for i in range(n_arms)]
        self.collected_rewards = np.array([])

    def update_observations(self, pulled_arm, reward):
        self.rewards_per_arm[pulled_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)
