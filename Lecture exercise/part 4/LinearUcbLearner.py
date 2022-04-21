import numpy as np


class LinearUcbLearner():
    def __init__(self, arms_features):
        self.arms = arms_features
        self.dim = arms_features.shape[1]
        self.collected_rewards = []
        self.pulled_arms = []
        self.c = 2.0  # exploration factor
        self.M = np.identity(self.dim)
        self.b = np.atleast_2d(np.zeros(self.dim)).T
        self.theta = np.dot(np.linalg.inv(self.M), self.b)

    def compute_ucbs(self):
        self.theta = np.dot(np.linalg.inv(self.M), self.b)
        ucbs = []
        for arm in self.arms:
            arm = np.atleast_2d(arm).T
            ucb = np.dot(self.theta.T, arm) + self.c * np.sqrt(np.dot(arm.T, np.dot(np.linalg.inv(self.M), arm)))
            ucbs.append(ucb[0][0])
        return ucbs

    def pull_arm(self):
        ucbs = self.compute_ucbs()
        return np.argmax(ucbs)

    def update_estimation(self, arm_idx, reward):
        arm = np.atleast_2d(self.arms[arm_idx]).T
        self.M += np.dot(arm, arm.T)
        self.b += reward * arm

    def update(self, arm_idx, reward):
        self.pulled_arms.append(arm_idx)
        self.collected_rewards.append(reward)
        self.update_estimation(arm_idx, reward)
