# file matching_advertising.py

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


################################################

# T - Time horizon - number of days
T = 200

number_of_experiments = 80

# number of advertisers for each publisher

N_ADS = 4
N_SLOTS = 4
N_USERS = 30  # number of users for each day
N_KLASSES = 3
PERIOD_TIME = 25

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
        advertiser = Advertiser(bid=1, publisher=publisher, budget= 1, d_budget=1)
        advertisers.append(advertiser)

    for e in range(number_of_experiments):
        print(np.round((e + 1) / number_of_experiments * 10000) / 100, "%")
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
        print(partition)
        for t in range(T):
            # generate contexts (partition)
            if int(t / PERIOD_TIME) > period:
                period += 1
                # choose best partition for new period by collected data
                partition, partition_index = choose_best_partition(user_data[e], partitions, k_p,
                                                                   prev_p_index=partition_index)
                print(partition)

            user_data[e].append([])
            cts_rewards_per_experiment_disaggregate[e].append([])

            users = generate_users(k_p, N_USERS)

            environment = AdAuctionEnvironment(advertisers, publisher, users, real_q=real_q_aggregate,
                                               real_q_klass=real_q_klass)

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

        for context_index in range(len(contexts)):
            collected_rewards = learners_by_context[context_index].collected_rewards
            cts_rewards_per_ex_partition[context_index].append(collected_rewards)
        for t in range(T):
            c = []
            for context_index in range(len(contexts)):
                c.append(cts_rewards_per_ex_partition[context_index][e][t])

            cts_rewards_per_experiment_disaggregate[e][t] = np.sum(np.array(c), axis=0)

    # Plot curve
    # Prepare data for aggregated model
    cts_rewards_per_experiment_aggregate = np.array(cts_rewards_per_experiment_aggregate)
    opt_q_aggregate = calculate_opt(real_q_aggregate, n_slots=N_SLOTS, n_ads=N_ADS)*N_USERS
    cumsum_aggregate = np.cumsum(np.mean(opt_q_aggregate - cts_rewards_per_experiment_aggregate, axis=0), axis=0)
    regret_aggregate = list(map(lambda x: np.sum(x), cumsum_aggregate))

    # Join disaggregated rewards for each experiment and day
    cts_rewards_per_experiment_disaggregate = np.array(cts_rewards_per_experiment_disaggregate)
    opt_q_klass = list(map(lambda x: calculate_opt(x, n_slots=N_SLOTS, n_ads=N_ADS), real_q_klass))
    opt_q_disaggregate = np.sum(list(map(lambda x: x[1] * k_p[x[0]] * N_USERS, enumerate(opt_q_klass))), axis=0)
    cumsum_disaggregate = np.cumsum(np.mean(opt_q_disaggregate - cts_rewards_per_experiment_disaggregate, axis=0),
                                    axis=0)
    regret_disagregate = list(map(lambda x: np.sum(x), cumsum_disaggregate))
    ######### WRONG CURVES
    # plt.figure(1)
    # plt.xlabel("t")
    # plt.ylabel("Regret")
    # colors = ['r', 'g', 'b']
    # plt.plot(regret_aggregate, 'm')
    # plt.legend(["Aggregated"])
    # plt.show()
    #
    #
    # plt.figure(1)
    # plt.xlabel("t")
    # plt.ylabel("Regret")
    # colors = ['r', 'g', 'b']
    # plt.plot(regret_disagregate, 'orange')
    # plt.legend(["Disaggregated"])
    # plt.show()
    #######################

    # Plot reward
    mean_reward_aggregate = np.mean(cts_rewards_per_experiment_aggregate, axis=0)
    mean_reward_disaggregate = np.mean(cts_rewards_per_experiment_disaggregate, axis=0)

    reward_disaggregate = list(map(lambda x: np.sum(x), mean_reward_disaggregate))
    reward_aggregate = list(map(lambda x: np.sum(x), mean_reward_aggregate))

    plt.figure(2)
    plt.xlabel("t")
    plt.ylabel("Reward")
    plt.plot(reward_aggregate, 'm')
    plt.plot(reward_disaggregate, 'orange')
    plt.legend(["Aggregated", "Disaggregated"])
    plt.show()

    plt.figure(3)
    plt.xlabel("t")
    plt.ylabel("Regret")
    plt.plot(np.cumsum(np.sum(opt_q_disaggregate) - reward_disaggregate), 'orange')
    plt.legend(["Disaggregated"])
    plt.show()

    plt.figure(3)
    plt.xlabel("t")
    plt.ylabel("Regret")
    plt.plot(np.cumsum(np.sum(opt_q_aggregate) - reward_aggregate), 'm')
    plt.legend(["Aggregated"])
    plt.show()

    smooth_reward_a = make_smoother(reward_aggregate)
    smooth_reward_d = make_smoother(reward_disaggregate)
    plt.figure(4)
    plt.xlabel("t")
    plt.ylabel("Reward")
    plt.plot(smooth_reward_a, 'm')
    plt.plot(smooth_reward_d, 'orange')
    plt.legend(["Aggregated", "Disaggregated"])
    plt.show()
    # separated plots
    plt.figure(5)
    plt.xlabel("t")
    plt.ylabel("Reward")
    plt.plot(smooth_reward_a, 'm')
    plt.legend(["Aggregated"])
    plt.show()

    plt.figure(6)
    plt.xlabel("t")
    plt.ylabel("Reward")
    plt.plot(smooth_reward_d, 'orange')
    plt.legend(["Disaggregated"])
    plt.show()