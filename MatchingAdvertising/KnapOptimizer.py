import numpy as np


class KnapOptimizer():
    def __init__(self, n_bids, n_budget, n_subcampaign, bids):
        self.bids = bids
        self.n_bids = n_bids
        self.n_budget = n_budget
        self.n_subcampaign = n_subcampaign
        self.step1n = []
        self.step2n = self.step1n.append([[[] for j in range(self.n_bids)] for i in range(self.n_budget)])

        for i in range(n_subcampaign):
            self.step1n.append([[[] for j in range(self.n_bids)] for i in range(self.n_budget)])
            self.step1n[i] = np.random.randint(120, size=(self.n_bids, self.n_budget))

        self.finalm = [[[] for j in range(self.n_bids)] for i in range(self.n_budget)]
        self.result = 0
        self.first = True
        # superarm
        # [budget,bid] subcampaign 1
        # [budget,bid] subcampaign 2
        # [budget,bid] subcampaign 3
        # [budget,bid] subcampaign 4

    def Optimize(self, n):
        self.step1n = n
        maxc = np.zeros(shape=(4, 4))
        #Here we get the max for each row of budget for each subcampaign
        for i in range(self.n_subcampaign):
            maxc[i] = np.amax(self.step1n[i], axis=1)
        self.step2n = maxc #array with the maximum value of each row of the initial matrix
        superarm = self.step1(self.step2n)

        return superarm

    def step1(self, step2bs):

        self.finalm[0] = step2bs[0]  #Just get the first row
        #We apply the knapsack step2 of slide
        for i in range(1, self.n_budget):
            for j in range(1, self.n_budget + 1):
                inv = self.finalm[i - 1][0:j]
                inv = inv[::-1]
                res = np.add(inv, step2bs[i][0:j])
                self.finalm[i][j - 1] = np.amax(res)

        self.result = np.argmax(self.finalm, axis=1) #Index of maximum value -> result

        superarm = np.zeros(shape = (4,2), dtype = int)
        bids  = []

        #Here we get the couple bid budget from the final result of previous step
        for i in range(self.n_subcampaign):
            val = self.step2n[i][self.result[i]]
            index = np.where(self.step1n[i][self.result[i]] == val)
            bids.append(float(self.bids[index[0]]))
            superarm[i] = self.result[i], index[0]

        return superarm
