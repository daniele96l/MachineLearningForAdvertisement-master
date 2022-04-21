from KnapOptimizer import *

from Publisher import *
from Advertiser import *
from VCG_auction import *
from AdAuctionEnvironment import *
from GPTSLearner import *
from User import *
from CTSLearner import *
from BiddingEnvironment import *
from hungarian_algorithm import hungarian_algorithm, convert_matrix
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm



def calculate_opt(real_q, n_slots, n_ads):
    opt = hungarian_algorithm(convert_matrix(real_q))
    #print("CALCULATE OPT", opt)
    m = opt[1]
    opt_q = np.array([])
    for j in range(n_slots):
        for i in range(n_ads):
            if m[i][j] == 1:
                opt_q = np.append(opt_q, real_q[i][j])
    return opt_q

def calculate_opt_advreal(real_q):
    opt_q = np.max(real_q[0])
    return opt_q

def make_smoother(data):
    smoothed = []
    for d_i, d in enumerate(data):
        if d_i >= 25:
            smoothed.append(np.mean(data[d_i - 25:d_i]))
        else:
            if d_i >= 5:
                smoothed.append(np.mean(data[d_i - 5:d_i]))
            else:
                smoothed.append(d)
    return smoothed

def generate_klasses_proportion(n_klasses):
    p = np.random.randint(100, size=n_klasses) + 20
    p = p / p.sum()
    return p


def generate_users(klasses_proportion, n_users):
    users = []
    klasses_features = [
        [1, 1],
        [0, 1],
        [1, 0]
    ]
    klasses = np.random.choice([0, 1, 2], n_users, p=klasses_proportion)
    for klass in klasses:
        f1 = klasses_features[klass][0]
        f2 = klasses_features[klass][1]
        user = User(feature1=f1, feature2=f2, klass=klass)
        users.append(user)
    np.random.shuffle(users)
    return users

#This function is use to update the budget after each click of the user
def update_budget(reward, advertisers, idx_subcampaign):
    for a in range(N_ADS):
        if(reward[a] == 1):
            #print("QUANTO per sub", idx_subcampaign , "ed adv ", a, "prezzo ", paying[idx_subcampaign][a])
            #print(advertisers[a].budget , advertisers[a].d_budget )
            if(advertisers[a].budget > paying[idx_subcampaign][a]):
                advertisers[a].budget -= paying[idx_subcampaign][a]  # TOTAL BUDGET is updated
            else:
                reward[a] == 0
                #advertisers[a].budget = 0

            if(advertisers[a].d_budget[idx_subcampaign] >paying[idx_subcampaign][a]):
                advertisers[a].d_budget[idx_subcampaign] -= paying[idx_subcampaign][a]  # daily b
            else:
                reward[a] == 0
                advertisers[a].d_budget[idx_subcampaign] = 0
                no_money_d[a][idx_subcampaign] = True
                #no_money_d[a][idx_subcampaign] = True
                #advertisers[a].d_budget[idx_subcampaign] = 0
            #USED only FOR PLOTTING, we saved the remaining budget over time
            if(a == 0):
                b1.append(advertisers[a].budget)
            if (a == 1):
                b2.append(advertisers[a].budget)
            if (a == 2):
                b3.append(advertisers[a].budget)
            if (a == 3):
                b4.append(advertisers[a].budget)
    return reward
#Function that check the daily budget for every advertiser using a matrix of boolean value
def check_dbudget(advertisers,idx_sub):
    for a in range(N_ADS):
        if (advertisers[a].d_budget[idx_sub] <= 0):
            no_money_d[idx_sub][a] = True

#Function that check the total budget of every advertiser using a matrix of boolean value
def check_budget(advertisers):
    for a in range(N_ADS):
        if (advertisers[a].budget <= 0):
            no_money_b[a] = True
            for i in range(N_SUBCAMPAIGN):
                no_money_d[i][a] = True

#This function is used to get the value of q(click probability) resulting from the auction
def get_q(res_auction,arm):
    for i in range(N_ADS):
        idx = res_auction[arm][0].index(i)
        q_adv[arm][i] = res_auction[arm][1][idx]
    #print("res auction ",res_auction[arm][0])
    idx = res_auction[arm][0].index(0)
    #print("position auction")
    vincitori[arm][idx] += 1    #USED only to see the position that our advertiser get after every auction
    return q_adv[arm]

#Function used to print the result(choice of the advertiser and knapsack) and remaining budget after finish the experiments
def print_result():
    print("-----------RESULT----------\n")
    row_labels = [dbud, dbud*2, dbud*3, dbud*4]
    col_labels = [bids[0], bids[1],bids[2], bids[3]]
    for i in range(N_SUBCAMPAIGN):
        print("SUBCAMPAIGN ", i+1)
        df = pd.DataFrame(choice[i], columns=col_labels, index=row_labels)
        df.astype(int)
        print(df)
        print("Number of times in position 1, 2, 3, 4 after auction: ", vincitori[i])
        print("--------------------------------")

    print("Remaining Budget\nAdvertiser 1: ", advertisers[0].budget," Advertiser 2: ", advertisers[1].budget," Advertiser 3: ", advertisers[2].budget," Advertiser 4: ", advertisers[3].budget)

#------------PARAMETER SETTING------------#

T = 200
number_of_experiments = 30

N_BIDS = 4                  #Number of linspaced Bids
N_BUDGET = 4                #Number of Daily budget choices
N_SUBCAMPAIGN = 4
N_ARMS = N_BIDS * N_BUDGET
N_ADS = 4
N_SLOTS = 4
N_USERS = 30                #Number of users Visiting the site each day
N_KLASSES = 3               #Classes of user (used only to get the starting q)
N_AUCTION = 3              #Number of auction per day
tot_b = 80000         #Total Budget
bids = np.linspace(start = 0.25, stop = 1, num = N_BIDS)
bid_competitor = bids[int(N_BIDS/2)-1]
dbud = ((tot_b/T) / 4) / N_SUBCAMPAIGN        #Used only to divide daily budget
d_budget= [dbud *3, dbud *3, dbud *3, dbud *3]          #Daily budget competitor (static)
our_d_budget = [dbud, dbud * 2, dbud * 3, dbud * 4]     #Our possible choice of daily budget
SLOTS_QUALITY = np.linspace(start = 1, stop = 0.25, num = N_SLOTS) #Quality of the slots(used in VCG Auction)

#------------PARAMETER SETTING------------#




#------------Summary Variable------------#

paying = np.zeros(shape=(4,4))
no_money_d = np.zeros((4, 4), dtype=bool)
no_money_b = np.zeros((4), dtype=bool)
has_finished = np.zeros((4,T), dtype=int)

b1 = []     #Budget of Advertisers
b2 = []
b3 = []
b4 = []
vincitori = np.zeros(shape=(4,4))
choice = np.zeros(shape=(4,4,4),dtype=int)

#------------Summary Variable------------#


publisher1 = Publisher(n_slots=4)
publishers = [publisher1]

k_p = generate_klasses_proportion(N_KLASSES)
real_q_klass = []
for klass in range(N_KLASSES):
    real_q_klass.append(np.random.uniform(size=(N_ADS, N_SLOTS)))

real_q_aggregate = np.sum(list(map(lambda x: x[1] * k_p[x[0]], enumerate(real_q_klass))), axis=0)
opt_q_klass = list(map(lambda x: calculate_opt(x, n_slots=N_SLOTS, n_ads=N_ADS), real_q_klass))

real_q_aggregate = real_q_klass[0]
cts_rewards_per_experiment_aggregate = [[], [], [], []]
opt_q_aggregate = calculate_opt(real_q_aggregate, n_slots=N_SLOTS, n_ads=N_ADS)



k_p = generate_klasses_proportion(N_KLASSES)


for publisher in publishers:
    advertisers = []
    for i in range(N_ADS):
        advertiser = Advertiser(bid=bid_competitor, publisher=publisher, budget=tot_b, d_budget= np.random.uniform(d_budget[np.random.randint(0,3)], size=4 ))
        advertisers.append(advertiser)


    for e in range(number_of_experiments):

        ##### For every experiment
        ##      we initiliaze budget, istance of learners, instance of knapsack optimizer

        for a in range(len(advertisers)):
            advertisers[a].budget = tot_b #at every experiment set a new budget
        print("Experiment ", e+1, "/", number_of_experiments)

        learner_by_subcampaign = []
        for subcampaign in range(N_SUBCAMPAIGN):
            advlearner = GPTSLearner(n_arms =N_ARMS,n_ads = N_ADS,n_bids = N_BIDS, n_budget= N_BUDGET, t = T, bids = bids, D_budget = our_d_budget)
            learner_by_subcampaign.append(advlearner)

        knap = KnapOptimizer(n_bids=N_BIDS, n_budget=N_BUDGET, n_subcampaign=4,bids = bids)

        for t in tqdm(range(T)):
            #### HERE WE ITERATE FOR EVERY DAY
            # 1. Initialize every day daily budget, Environment, Daily reward
            no_money_d = np.zeros((4, 4), dtype=bool)
            for a in range(1, len(advertisers)):
                advertisers[a].d_budget = np.random.uniform(d_budget[np.random.randint(0,3)], size=4)  # at every day i set a new d_budget
                advertisers[a].bid = bids[int(N_BIDS/2)-1] + np.random.normal(0.1, 0.5, 1)/20

            users = generate_users(k_p,N_USERS)
            Adenvironment = AdAuctionEnvironment(advertisers, publisher, users, real_q=real_q_aggregate, real_q_klass=real_q_klass)

            sample_n = []
            res_auction = []
            reward_gaussian = [0, 0, 0, 0]
            q_adv = np.zeros(shape=(4,4))

            # 2. Take sample of estimation of n (estimated number of click)
            for i in range(N_SUBCAMPAIGN):
                sample_n.append(np.reshape(learner_by_subcampaign[i].estimate_n(), (4,4)))

            # 3. Run the Knapsack Optimizer and get the Optimal Daily Budget/Bid Couple
            #    and set the daily budget to our advertiser (for every subcampaign)
            superarm = knap.Optimize(sample_n)
            for i in range(N_SUBCAMPAIGN):
                advertisers[0].d_budget[i] = our_d_budget[superarm[i][0]]
                choice[i][superarm[0][0]][superarm[0][1]] += 1

            # 4. Iterate Over N_Auction
            #    For each subcampaign create an run a simulation of an auction
            for auc in range(N_AUCTION):
                for arm in range(N_SUBCAMPAIGN):
                    auction = VCG_auction(real_q_aggregate, superarm[arm], N_SLOTS, advertisers,minbid=bids[0])
                    res_auction.append(auction.choosing_the_slot(real_q_aggregate, SLOTS_QUALITY,arm))
                    q_adv[arm] = get_q(res_auction,arm)
                    check_dbudget(advertisers, arm)

                    #Here we Have the pay per click for each subcampaign
                    paying[arm] = [x for _,x in sorted(zip(res_auction[arm][0],res_auction[arm][2]))]

                check_budget(advertisers)

                # 5. For every User (Over N_Subcampaign)
                #    Check budget and update the budget
                #    get the reward after a simulation of user behaviour
                for user in users:
                    for j in range(N_SUBCAMPAIGN):

                        if (not (no_money_d[0][j])):
                            has_finished[j][t] += 1

                        reward = Adenvironment.simulate_user_behaviour_auction(user, q_adv[j], advertisers) #SIMULART |||||||||||||
                        reward = update_budget(reward, advertisers,j)
                        #check_dbudget(advertisers,j)
                        #check_budget(advertisers)
                        reward_gaussian[j] += reward[0]

            #6. At the end of the Day update the Gaussian process Regressor with the number of click
            for i in range(N_SUBCAMPAIGN):
                learner_by_subcampaign[i].update(superarm[i], reward_gaussian[i], t)


        # Accumulate the reward
        for i in range(N_SUBCAMPAIGN):
            cts_rewards_per_experiment_aggregate[i].append(learner_by_subcampaign[i].collected_rewardsy)


    #After finish the experiment we can print the result and see the choice of learner and the remaining budget
    print_result()
    for i in range(N_SUBCAMPAIGN):
        cts_rewards_per_experiment_aggregate[i] = np.array(cts_rewards_per_experiment_aggregate[i])
    opt_q_aggregate = calculate_opt_advreal(real_q_aggregate)
    cumsum_aggregate = []
    reward_aggregate = []
    for i in range(N_SUBCAMPAIGN):
        cumsum_aggregate.append(np.cumsum(np.mean(opt_q_aggregate - (cts_rewards_per_experiment_aggregate[i]/(N_USERS*N_AUCTION)), axis=0),axis=0))
        reward_aggregate.append(np.mean(cts_rewards_per_experiment_aggregate[i],axis=0))

    smooth_reward = make_smoother(np.mean(reward_aggregate, axis=0))


    plt.figure(1)
    plt.title("Budget")
    plt.xlabel("t")
    plt.ylabel("Budget")
    plt.plot(b1, 'm')
    plt.plot(b2, 'r')
    plt.plot(b3, 'b')
    plt.plot(b4, 'y')
    plt.show()

    plt.figure(2)
    plt.title("Regret for each subcampaign")
    plt.plot(cumsum_aggregate[0], "m")
    plt.plot(cumsum_aggregate[1], "g")
    plt.plot(cumsum_aggregate[2], "b")
    plt.plot(cumsum_aggregate[3], "y")
    plt.legend(["Sub1", "Sub2", "Sub3", "Sub4"])
    plt.show()

    plt.figure(3)
    plt.title("Regret")
    plt.xlabel("t")
    plt.ylabel("Regret")
    plt.plot(np.mean(cumsum_aggregate, axis= 0), 'm')
    plt.show()

    plt.figure(4)
    plt.title("Reward")
    plt.xlabel("t")
    plt.ylabel("Reward")
    #plt.plot(np.mean(reward_aggregate,axis=0), 'm')
    plt.plot(smooth_reward, 'g')
    plt.show()
