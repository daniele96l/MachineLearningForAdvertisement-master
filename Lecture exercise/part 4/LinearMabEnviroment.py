import numpy as np


class LinearMabEnvironment():
    def __init__(self, n_arms, dim):
        self.theta = np.random.dirichlet(np.ones(dim), size=1)
        self.arms_features = np.random.binomial(1, 0.5, size=(n_arms, dim))
        self.p = np.zeros(n_arms)
        for i in range(0, n_arms):
            self.p[i] = np.dot(self.theta, self.arms_features[i])

    def round(self, pulled_arm):
        return 1 if np.random.random() < self.p[pulled_arm] else 0

    def opt(self):
        return np.max(self.p)
