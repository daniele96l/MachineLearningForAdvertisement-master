from typing import List, Union

import matplotlib.pyplot as plt
from numpy.core.multiarray import ndarray

from Environment import *
from Learner import *
from CTSLearner import *
from Hungarian import *


n_arms = 4 # WHAT THE HELL DO I HAVE TO DO WITH PROBABILITIES?
p = np.array([0.15, 0.1, 0.1,0.35])  # probabilities for each arm (probability of obtaining 1 as a sample (Bernoulli lies in {0,1})à
opt = p[3]  # This is the optimal arm (0.35 is the greatest) --> My guess

T = 1  # Time Horizon
X = 1  #number of arms of the superarm
n_experiments = 1    # number of experiments
arraysRewards = []
ts_rewards_per_experiment = []
gr_rewards_per_experiment = []
new_array = np.zeros(shape= 16)

def update(oldarray, newarray):
    for x in range(len(new_array)):
        if  new_array[x] == 0:
            new_array[x] = oldarray[x]
        new_array[x] += oldarray[x] #not sure about this

    return 0


for e in range(n_experiments): #number of experiments
    oldvalues = np.zeros(shape= 16)
    idx, rewards = getArms_updateMatrix(np.random.randint(1, size=(16, 1)), oldvalues = oldvalues)  # RUN THE HUNGARIAN FOR THE FIRST TIME
    # n_arms = len(idx)
    print("experiment number: ",e)
    print("array of indexes", idx)

    env = Environment(n_arms=n_arms, probabilities=p)
    cts_learner = CTSLearner(n_arms)

    for t in range(T):   # T = time orizon
        for x in range(len(idx)): #here we pull every arm of the superarm

            if(idx[x] != 0): #the arms with index 0 are not in the superarm
               # pulled_arm =  idx[x]*x   #THIS IS THE INDEX OF THE ARM NEEDED TO PULL
                pulled_arm = cts_learner.pull_arm()  #IT WAS THIS WAY
                reward = env.round(pulled_arm) #assign the reward of the pulled arm, THE ENVIROMENT IS GIVING ME THE REWARD
                new_array[x] = reward
                update(rewards,new_array)  #cumulative reward
                cts_learner.update(pulled_arm, reward) #update the values in the ts_learner


            print(new_array, "New array")
            idx,rewards = getArms_updateMatrix(new_array, rewards)  # RUN THE HUNGARIAN AND GET THE SUPERARMS with the new arms

    ts_rewards_per_experiment.append(cts_learner.collected_rewards)



# Regret = T*opt - sum_t(rewards_t)

plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")


# Calculate the instantaneous regret for each t for each experiment
# Note: regret_t = optimal_arm_value - pulled_arm_value
# Note: a positive regret is "bad", a negative regret is "good"
regrets = opt - ts_rewards_per_experiment

# Calculate the average regret for each iteration t
# Note that we are conducting n_experiments so we need the average over all the experiments for each iteration t
# Note that axis=0 means that you are averaging over each iteration t (over the column)
avg_regrets = np.mean(regrets, axis=0)

# Avg_Regret = T*opt - sum_t(value_t)
# Note that we have already calculated the average regret (opt-value)
# So we only need to cumulatively sum the array containing the avg_regret for each itearation t
# np.cumsum(array) returns an array with item at position i equal to the sum of the items at previous positions (pos i included)
avg_regret = np.cumsum(avg_regrets)
# Plot
plt.plot(avg_regret, 'r')

# The same is done for Greedy_Learner
plt.plot(np.cumsum(np.mean(opt-gr_rewards_per_experiment, axis=0)), 'g')

plt.legend(["TS"])
plt.show()

