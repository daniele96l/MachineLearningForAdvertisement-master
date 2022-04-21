from Environment import *


class AdAuctionEnvironment(Environment):
    def __init__(self, advertisers, publisher, users, real_q, real_q_klass):
        self.advertisers = advertisers
        self.publisher = publisher
        self.users = users
        self.real_q = real_q

        self.real_q_klass = real_q_klass

    def simulate_user_behaviour(self, user, edges):
        reward = np.zeros(len(edges))
        for edge in edges:
            i = edge[0]  # number of advertiser
            j = edge[1]  # number of slot
            q_ij = self.real_q_klass[user.klass][i][j]  # real probability of click

            reward[j] = np.random.binomial(1, q_ij)

        return reward

    def simulate_user_behaviour_as_aggregate(self, user, edges):
        reward = np.zeros(len(edges))
        for edge in edges:
            i = edge[0]  # number of advertiser
            j = edge[1]  # number of slot
            q_ij = self.real_q[i][j]  # real probability of click

            reward[j] = np.random.binomial(1, q_ij)
            #print(reward[j] ,j, "reward")
        return reward

    def simulate_user_behaviour_auction(self, user, q, advertisers):
        reward = np.zeros(len(advertisers))
        for i in range(len(advertisers)):
            reward[i] = np.random.binomial(1, q[i])

        return reward

        return reward

    def simulate_user_behaviour_bidding(self, q_ij):
        reward = np.zeros(len(q_ij))
        for i in range(len(q_ij)):
            reward[i] = np.random.binomial(1, q_ij[i])
        return reward

    def simulate_user_behaviour_opt(self, q_ij):
        reward = np.zeros(len(q_ij))
        for i in range(len(q_ij)):
            reward += np.random.binomial(1, q_ij[i])
        return reward

