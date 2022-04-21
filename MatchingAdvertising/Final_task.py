from Publisher import *
from Advertiser import *
from AdAuctionEnvironment import *
from VCG_auction import *
from KnapOptimizer import *
from GPTSLearner import *
from User import *
from CTSLearner import *
from hungarian_algorithm import hungarian_algorithm, convert_matrix
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


from Publisher import *
from Advertiser import *
from AdAuctionEnvironment import *
from User import *
from CTSLearner import *
from hungarian_algorithm import hungarian_algorithm, convert_matrix
import numpy as np
import matplotlib.pyplot as plt

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

def extract_hungarian_result(initial_matrix, hungarian_matrix):
    m_shape = np.shape(hungarian_matrix)
    result = np.array([])
    for j in range(m_shape[0]):
        for i in range(m_shape[1]):
            if hungarian_matrix[i][j] == 1:
                result = np.append(result, initial_matrix[i][j])
    return result


def calculate_opt(real_q, n_slots, n_ads):
    opt = hungarian_algorithm(convert_matrix(real_q))
    opt_q = extract_hungarian_result(real_q, opt[1])
    return opt_q

def calculate_opt_advreal(real_q):
    opt_q = np.max(real_q[0])
    return opt_q

def generate_klasses_proportion(n_klasses):
    p = np.random.randint(100, size=n_klasses) + 20
    p = p / p.sum()
    return p


def generate_users(klasses_proportion, n_users):
    features_per_klass = [[[0, 0]], [[0, 1], [1, 0]], [[1, 1]]]
    users = []
    klasses = np.random.choice([0, 1, 2], n_users, p=klasses_proportion)
    for klass in klasses:
        f_num = np.random.randint(len(features_per_klass[klass]))
        feature1 = features_per_klass[klass][f_num][0]
        feature2 = features_per_klass[klass][f_num][1]
        new_user = User(feature1=feature1, feature2=feature2, klass=klass)
        users.append(new_user)

    np.random.shuffle(users)
    return users


# calculates sum and count matricies for context
def calculate_sc_for_context(context, user_data):
    s_matrix = np.zeros(shape=(4, 4))
    c_matrix = np.zeros(shape=(4, 4))
    for day in user_data:
        for data in day:
            user = data[0]
            reward = data[1]
            superarm = data[2]
            if user.klass in context:
                for i, r_i in enumerate(reward):
                    arm = superarm[i]
                    s_matrix[arm[0], arm[1]] += r_i
                    c_matrix[arm[0], arm[1]] += 1
    # to preserve of nan values
    for i in range(c_matrix.shape[0]):
        for j in range(c_matrix.shape[1]):
            if c_matrix[i, j] == 0:
                c_matrix[i, j] += 1

    return s_matrix, c_matrix


# calculates lower bound for context given data
def calculate_lb_for_context(context, user_data):
    s_matrix, c_matrix = calculate_sc_for_context(context, user_data)
    mean = s_matrix / c_matrix
    lb = mean - np.sqrt(-np.log(0.90) / (2 * c_matrix))
    return lb


# calculates expected reward for partition given context and user classes proportion
def calculate_er_for_partition(partition, user_data, k_p):
    er = np.zeros(shape=4)
    for cont in partition:
        # p_cont - probability that context cont occures
        p_cont = 0
        for k in cont:
            p_cont = p_cont + k_p[k]

        lower_bound = calculate_lb_for_context(cont, user_data)
        er_raw = hungarian_algorithm(convert_matrix(lower_bound))
        expected_reward = extract_hungarian_result(lower_bound, er_raw[1])
        er += p_cont * expected_reward
    return er


def choose_best_partition(user_data, partitions, k_p, prev_p_index):
    # for every possible partition of space of the features (we have 1 feature - user class: {c1,c2,c3})
    # evaluate whether partitioning is better than no doing that
    ers = []
    for p_i, p in enumerate(partitions):
        er = calculate_er_for_partition(p, user_data, k_p)
        ers.append(np.sum(er))
    partition_index = np.argmax(ers)
    if prev_p_index > partition_index:
        partition_index = prev_p_index
    partition = partitions[partition_index]
    return partition, partition_index


def samples_from_learner(cts_learner, n_ads, n_slots):
    samples = np.zeros(shape=(n_ads, n_slots))
    for i in range(N_ADS):
        for j in range(N_SLOTS):
            a = cts_learner.beta_parameters[i][j][0]
            b = cts_learner.beta_parameters[i][j][1]
            samples[i][j] = np.random.beta(a=a, b=b)
    return samples


def get_context_for_user(user_klass, partition):
    context = []
    for c in partition:
        if user_klass in c:
            context = c
    return context


def get_context_index(context, contexts):
    context_index = -1

    for i, c in enumerate(contexts):
        if c == context:
            context_index = i

    return context_index

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




################################################

# T - Time horizon - number of days
T = 200

number_of_experiments = 30

# number of advertisers for each publisher
DAYS_SPLIT = 25
N_BIDS = 4                  #Number of linspaced Bids
N_BUDGET = 4                #Number of Daily budget choices
N_SUBCAMPAIGN = 4
N_ARMS = N_BIDS * N_BUDGET
N_ADS = 4
N_SLOTS = 4
N_USERS = 30               #Number of users Visiting the site each day
N_KLASSES = 3               #Classes of user (used only to get the starting q)
N_AUCTION = 3               #Number of auction per day
tot_b = 80000             #Total Budget
bids = np.linspace(start = 0.25, stop = 1, num = N_BIDS)
bid_competitor = bids[int(N_BIDS/2)-1]
dbud = ((tot_b/T) / 4) / N_SUBCAMPAIGN        #Used only to divide daily budget
d_budget= [dbud *3, dbud *3, dbud *3, dbud *3]          #Daily budget competitor (static)
our_d_budget = [dbud, dbud * 2, dbud * 3, dbud * 4]     #Our possible choice of daily budget
SLOTS_QUALITY = np.linspace(start = 1, stop = 0.25, num = N_SLOTS) #Quality of the slots(used in VCG Auction)




#------------Summary Variable------------#

paying = np.zeros(shape=(4,4))
no_money_d = np.zeros((4, 4), dtype=bool)
no_money_b = np.zeros((4), dtype=bool)
has_finished = np.zeros((4,T), dtype=int)

b1 = []     #Cumulative Budget of Advertisers
b2 = []
b3 = []
b4 = []
vincitori = np.zeros(shape=(4,4))
choice = np.zeros(shape=(4,4,4),dtype=int)



publisher1 = Publisher(n_slots=4)

publishers = [publisher1]

# initialize probabilities q_ij total and for each class

k_p = generate_klasses_proportion(N_KLASSES)
assert k_p.sum() == 1.0
print("User klasses proportion:")
print(k_p)

real_q_klass = []
for klass in range(N_KLASSES):
    real_q_klass.append(np.random.uniform(size=(N_ADS, N_SLOTS)))

real_q_aggregate = np.sum(list(map(lambda x: x[1] * k_p[x[0]], enumerate(real_q_klass))), axis=0)

rewards_vcg = [[], [], [], []]
real_q_adv = real_q_klass[0]
cts_rewards_per_experiment_aggregate = []
cts_rewards_per_experiment_disaggregate = []

# All possible partitions for 3 user classes
partitions = [
    [[0, 1, 2]],  # 0
    [[0, 1], [2]],  # 1
    [[0], [1, 2]],  # 2
    [[0, 2], [1]],  # 3
    [[0], [1], [2]]  # 4
]

contexts = [
    [0, 1, 2],  # 0
    [0, 1],  # 1
    [0, 2],  # 2
    [1, 2],  # 3
    [0],  # 4
    [1],
    [2]
]

cts_rewards_per_ex_partition = [[] for i in range(len(contexts))]
user_data = []

# Learn q_ij

for publisher in publishers:
    advertisers = []
    for i in range(N_ADS):
        advertiser = Advertiser(bid=bid_competitor, publisher=publisher, budget=tot_b, d_budget= np.random.uniform(d_budget[np.random.randint(0,3)], size=4 ))
        advertisers.append(advertiser)

    for e in range(number_of_experiments):
        print("Experiment ", e+1, "/", number_of_experiments)

        for a in range(len(advertisers)):
            advertisers[a].budget = tot_b #at every experiment set a new budget

        learner_by_subcampaign = []
        for subcampaign in range(N_SUBCAMPAIGN):
            advlearner = GPTSLearner(n_arms =N_ARMS, n_ads = N_ADS, n_bids = N_BIDS, n_budget= N_BUDGET, t = T, bids = bids, D_budget = our_d_budget)
            learner_by_subcampaign.append(advlearner)

        knap = KnapOptimizer(n_bids=N_BIDS, n_budget=N_BUDGET, n_subcampaign=4,bids = bids)

        cts_learner_aggregate = CTSLearner(n_ads=N_ADS, n_slots=publisher.n_slots, t=T)

        learners_by_context = []
        for part in range(len(contexts)):
            learner_by_context = CTSLearner(n_ads=N_ADS, n_slots=publisher.n_slots, t=T)
            learners_by_context.append(learner_by_context)

        user_data.append([])
        cts_rewards_per_experiment_disaggregate.append([])
        period = 0

        # Default partition is partition that aggregates all 3 user classes
        partition = partitions[0]
        partition_index = 0
        #print(partition)
        for t in tqdm(range(T)):
            # generate contexts (partition)ù

            no_money_d = np.zeros((4, 4), dtype=bool)
            for a in range(1, len(advertisers)):
                advertisers[a].d_budget = np.random.uniform(d_budget[np.random.randint(0, 3)],
                                                            size=4)  # at every day i set a new d_budget
                advertisers[a].bid = bids[int(N_BIDS / 2) - 1] + np.random.normal(0.1, 0.5, 1) / 20

            sample_n = []
            res_auction = []
            reward_gaussian = [0, 0, 0, 0]
            q_adv = np.zeros(shape=(4, 4))

            # 2. Take sample of estimation of n (estimated number of click)
            for i in range(N_SUBCAMPAIGN):
                sample_n.append(np.reshape(learner_by_subcampaign[i].estimate_n(), (4, 4)))

            # 3. Run the Knapsack Optimizer and get the Optimal Daily Budget/Bid Couple
            #    and set the daily budget to our advertiser (for every subcampaign)
            superarm_adv = knap.Optimize(sample_n)
            for i in range(N_SUBCAMPAIGN):
                advertisers[0].d_budget[i] = our_d_budget[superarm_adv[i][0]]
                choice[i][superarm_adv[0][0]][superarm_adv[0][1]] += 1

            if int(t / DAYS_SPLIT) > period:
                period += 1
                # choose best partition for new period by collected data
                partition, partition_index = choose_best_partition(user_data[e], partitions, k_p,
                                                                   prev_p_index=partition_index)
                #print(partition)

            user_data[e].append([])
            cts_rewards_per_experiment_disaggregate[e].append([])

            users = generate_users(k_p, N_USERS)

            environment = AdAuctionEnvironment(advertisers, publisher, users, real_q=real_q_aggregate,
                                               real_q_klass=real_q_klass)
            for auc in range(N_AUCTION):
                for arm in range(N_SUBCAMPAIGN):
                    auction = VCG_auction(real_q_adv, superarm_adv[arm], N_SLOTS, advertisers,minbid=bids[0])
                    res_auction.append(auction.choosing_the_slot(real_q_adv, SLOTS_QUALITY,arm))
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

                        reward_adv = environment.simulate_user_behaviour_auction(user, q_adv[j], advertisers) #SIMULART |||||||||||||
                        reward_adv = update_budget(reward_adv, advertisers,j)
                        check_dbudget(advertisers,j)
                        check_budget(advertisers)
                        reward_gaussian[j] += reward_adv[0]

            for i in range(N_SUBCAMPAIGN):
                learner_by_subcampaign[i].update(superarm_adv[i], reward_gaussian[i], t)


            for user in users:
                # ############ aggregate Learner
                # 1. FOR EVERY ARM MAKE A SAMPLE  q_ij - i.e. PULL EACH ARM
                samples_aggregate = samples_from_learner(cts_learner_aggregate, N_ADS, N_SLOTS)
                superarm_aggregate = publisher.allocate_ads(samples_aggregate)
                # 2. PLAY SUPERARM -  i.e. make a ROUND
                reward_aggregate = environment.simulate_user_behaviour_as_aggregate(user, superarm_aggregate)
                # 3. UPDATE BETA DISTRIBUTIONS
                cts_learner_aggregate.update(superarm_aggregate, reward_aggregate, t=t)

                # ######## learner for context
                # 1. FOR EVERY ARM MAKE A SAMPLE  q_ij - i.e. PULL EACH ARM
                context = get_context_for_user(user.klass, partition)
                context_index = get_context_index(context, contexts)
                context_learner = learners_by_context[context_index]

                partition_samples = samples_from_learner(context_learner, N_ADS, N_SLOTS)
                superarm = publisher.allocate_ads(partition_samples)
                # 2. PLAY SUPERARM -  i.e. make a ROUND
                reward = environment.simulate_user_behaviour(user, superarm)
                # 3. UPDATE BETA DISTRIBUTIONS
                context_learner.update(superarm, reward, t=t)

                user_data[e][t].append([user, reward, superarm])


        # print(partition)
        # collect results for publisher
        cts_rewards_per_experiment_aggregate.append(cts_learner_aggregate.collected_rewards)


        for i in range(N_SUBCAMPAIGN):
            rewards_vcg[i].append(learner_by_subcampaign[i].collected_rewardsy)


        for context_index in range(len(contexts)):
            collected_rewards = learners_by_context[context_index].collected_rewards
            cts_rewards_per_ex_partition[context_index].append(collected_rewards)
        for t in range(T):
            c = []
            for context_index in range(len(contexts)):
                c.append(cts_rewards_per_ex_partition[context_index][e][t])

            cts_rewards_per_experiment_disaggregate[e][t] = np.sum(np.array(c), axis=0)

    print_result()
    #print("TOTAL STEP: ", (N_USERS * N_AUCTION), "/", has_finished[0])


    for i in range(N_SUBCAMPAIGN):
        rewards_vcg[i] = np.array(rewards_vcg[i])
    opt_q_aggregate_adv = calculate_opt_advreal(real_q_adv)
    #opt_q_aggregate = calculate_opt(real_q_aggregate, n_slots=N_SLOTS, n_ads=N_ADS)
    cumsum_aggregate_adv = []
    reward_aggregate_adv = []
    for i in range(N_SUBCAMPAIGN):
        cumsum_aggregate_adv.append(np.cumsum(np.mean(opt_q_aggregate_adv - (rewards_vcg[i]/(N_USERS*N_AUCTION)), axis=0),axis=0))
        reward_aggregate_adv.append(np.mean(rewards_vcg[i],axis=0))

    smooth_reward = []
    reward_aggregate2 = np.mean(reward_aggregate_adv, axis=0)
    for r_i, r in enumerate(reward_aggregate2):
        if r_i >= 25:
            smooth_reward.append(np.mean(reward_aggregate2[r_i - 25:r_i]))
        else:
            if r_i >= 5:
                smooth_reward.append(np.mean(reward_aggregate2[r_i - 5:r_i]))
            else:
                smooth_reward.append(r)

    # Plot curve
    # Prepare data for aggregated model
    cts_rewards_per_experiment_aggregate = np.array(cts_rewards_per_experiment_aggregate)
    opt_q_aggregate = calculate_opt(real_q_aggregate, n_slots=N_SLOTS, n_ads=N_ADS)*N_USERS

    # Join disaggregated rewards for each experiment and day
    cts_rewards_per_experiment_disaggregate = np.array(cts_rewards_per_experiment_disaggregate)
    opt_q_klass = list(map(lambda x: calculate_opt(x, n_slots=N_SLOTS, n_ads=N_ADS), real_q_klass))
    opt_q_disaggregate = np.sum(list(map(lambda x: x[1] * k_p[x[0]]*N_USERS, enumerate(opt_q_klass))), axis=0)


    mean_reward_aggregate = np.mean(cts_rewards_per_experiment_aggregate, axis=0)
    mean_reward_disaggregate = np.mean(cts_rewards_per_experiment_disaggregate, axis=0)

    reward_disaggregate = list(map(lambda x: np.sum(x), mean_reward_disaggregate))
    reward_aggregate = list(map(lambda x: np.sum(x), mean_reward_aggregate))


    smooth_reward_a = make_smoother(reward_aggregate)
    smooth_reward_d = make_smoother(reward_disaggregate)



    plt.figure(1)
    plt.title("Regret Publisher (Point 3)")
    plt.xlabel("t")
    plt.ylabel("Regret")
    plt.plot(np.cumsum(np.sum(opt_q_aggregate) - reward_aggregate), 'orange')
    plt.legend(["Aggregated regret"])
    plt.show()

    plt.figure(2)
    plt.title("Reward Publisher(Point 3)")
    plt.xlabel("t")
    plt.ylabel("Reward")
    plt.plot(smooth_reward_a, 'green')
    plt.legend(["Aggregated reward"])
    plt.show()


    plt.figure(3)
    plt.title("Regret Publisher (Point 4)")
    plt.xlabel("t")
    plt.ylabel("Regret")
    plt.plot(np.cumsum(np.sum(opt_q_disaggregate) - reward_disaggregate), 'orange')
    plt.legend(["Disaggregated regret"])
    plt.show()

    plt.figure(4)
    plt.title("Reward Publisher (Point 4)")
    plt.xlabel("t")
    plt.ylabel("Reward")
    plt.plot(smooth_reward_d, 'green')
    plt.legend(["Context reward"])
    plt.show()





    plt.figure(5)
    plt.title("Regret Advertiser (Point 6)")
    plt.xlabel("t")
    plt.ylabel("Regret")
    plt.plot(list(map(lambda x: np.sum(x), np.mean(cumsum_aggregate_adv, axis= 0))), 'blue')
    plt.legend(["Advertiser regret"])
    plt.show()


    # Plot reward

    plt.figure(6)
    plt.title("Reward Advertiser (point 6)")
    plt.xlabel("t")
    plt.ylabel("Reward")
    plt.plot(smooth_reward, 'green')
    plt.legend(["Advertiser reward"])
    plt.show()


    plt.show()