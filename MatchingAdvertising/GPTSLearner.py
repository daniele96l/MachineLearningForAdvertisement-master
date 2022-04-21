from Learner import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C,Matern as M


class GPTSLearner(Learner):
    def __init__(self, n_arms,n_ads,n_bids, n_budget,t, D_budget, bids):
        super().__init__(n_arms)
        self.arms = np.array([[np.ones(shape=1) for j in range(n_ads)] for i in range(n_ads)])
        self.means = np.zeros(self.n_arms)
        self.sigmas = np.ones(self.n_arms) * 10
        self.pulled_arms = []
        self.bids = bids
        self.D_budget = D_budget
        alpha = 10
        #kernel =C(1.0, (1e-3, 1e3)) * RBF([2111111111111, 1], (1e-2, 1e2))
        #kernel = C(1.0, (1e-3, 1e3)) * RBF([5,5], (1e-2, 1e2))
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha = alpha **2, n_restarts_optimizer=10)
        self.input = np.array([[self.D_budget[i], self.bids[j]] for i in range(n_bids) for j in range(n_ads) ])
        #self.input = np.array([[2500, 25],[2500, 50],[2500, 75],[2500, 100],[5000, 25],[5000, 50],[5000, 75],[5000, 100],[7500, 25],[7500, 50],[7500, 75],[7500, 100],[10000, 25],[10000, 50],[10000, 75],[10000, 100]])
        self.rewards_per_arm = [[[] for j in range(n_bids)] for i in range(n_budget)]
        self.collected_rewards = [[] for j in range(t)]
        self.collected_rewardsy = np.array([])


    def update(self,arm, reward,t):
        self.t += 1
        self.update_observation(arm, reward,t)
        self.update_model(t)

    def update_observation(self, arm_idx, reward,t):
        #self.rewards_per_arm[arm_.append(reward)]
        self.collected_rewardsy = np.append(self.collected_rewardsy, reward)
        #self.collected_rewards[t].append(reward)
        value = np.array([self.D_budget[arm_idx[0]],self.bids[arm_idx[1]]])
        #print(value)
        self.pulled_arms.append(value)
        # [3,1]
    def update_model(self,t):
        x =np.atleast_2d(self.pulled_arms)
        y = self.collected_rewardsy #N click
        self.gp.fit(x, y)
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.input), return_std=True)
        self.sigmas = np.maximum(self.sigmas, 1e-2)

    def estimate_n(self):
        #print(np.random.normal(self.means,self.sigmas))
        return np.random.normal(self.means,self.sigmas)

    def update_reward(self,reward,t):
        self.collected_rewards[t].append(reward)




