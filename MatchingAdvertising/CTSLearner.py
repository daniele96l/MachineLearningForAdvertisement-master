import numpy as np


# Combinatorial Thompson Sampling Learner
class CTSLearner:
    def __init__(self, n_ads, n_slots, t):
        # Beta parameters for each edge-arm. Array looks like
        # [
        # [[1,1],[1,1],[1,1],[1,1]],    ## Ad 1 - [Slot1, Slot2, Slot3, Slot4]
        # [[1,1],[1,1],[1,1],[1,1]],    ## Ad 2 - [Slot1, Slot2, Slot3, Slot4]
        # [[1,1],[1,1],[1,1],[1,1]],    ## Ad 3 - [Slot1, Slot2, Slot3, Slot4]
        # [[1,1],[1,1],[1,1],[1,1]]     ## Ad 4 - [Slot1, Slot2, Slot3, Slot4]
        # ...
        # ]
        self.beta_parameters = np.array([[np.ones(shape=2) for j in range(n_slots)] for i in range(n_ads)])
        # initialize parameters of learner
        self.t = 0
        # Rewards for each edge-arm
        # [
        # [[],[],[],[]],    ## Ad 1 - [Slot1, Slot2, Slot3, Slot4]
        # [[],[],[],[]],    ## Ad 2 - [Slot1, Slot2, Slot3, Slot4]
        # [[],[],[],[]],    ## Ad 3 - [Slot1, Slot2, Slot3, Slot4]
        # [[],[],[],[]]     ## Ad 4 - [Slot1, Slot2, Slot3, Slot4]
        # ...
        # ]
        self.rewards_per_arm = [[[] for j in range(n_slots)] for i in range(n_ads)]
        self.collected_rewards = [[] for j in range(t)]

    def update_observations(self, arm, reward):
        self.rewards_per_arm[arm[0]][arm[1]].append(reward)

    def pull_arm(self):
        # pull arm
        return

    def update(self, superarm, reward, t):
        self.t += 1
        self.collected_rewards[t].append(reward)
        for arm_i, arm in enumerate(superarm, start=0):
            self.update_observations(arm, reward[arm_i])
            self.beta_parameters[arm[0], arm[1], 0] += reward[arm_i]
            self.beta_parameters[arm[0], arm[1], 1] += 1 - reward[arm_i]
