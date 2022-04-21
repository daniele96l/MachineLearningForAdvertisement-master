import numpy as np
from copy import  copy


def simulate_episode(init_prob_matrix, n_step_max):
    prob_matrix = init_prob_matrix.copy()
    n_nodes = prob_matrix.shape[0]
    initial_active_nodes = np.random.binomial(1,0.1, size = (n_nodes))
    history = np.array([initial_active_nodes])
    active_nodes = initial_active_nodes
    newly_active_nodes = active_nodes
    t = 0
    while(t < n_step_max and np.sum(newly_active_nodes) > 0):
        p = (prob_matrix.T* active_nodes).T #select only the nodes related with the active nodes
        activated_edges = p> np.random.rand(p.shape[0], p.shape[1]) #compute the value of the activated edges by sampling a value form a random distribution from
        # 0 to 1 ando compating it with each edges, if the value of the probbaility is bigger than the value drawn by the distribution than the enge will be activated
        prob_matrix = prob_matrix*((p!=0)== activated_edges) #we remove from the probability matrix all the values of the probabilities related to the previus activate nodes
        newly_active_nodes = (np.sum(activated_edges,axis=0)> 0) * (1-active_nodes) # we compute ht avlaue of newly activate nodes
        active_nodes = np.array(active_nodes + newly_active_nodes) #we update the value of active nodes
        history = np.concatenate((history, [newly_active_nodes]), axis=0) #append to the history variable the newly activate nodes
        t+= 1
    return history

def estimate_probabilities(dataset, node_index, n_nodes):  #now we estimate the probabilities
    estimated_prob = np.ones(n_nodes)*1.0/(n_nodes-1)  #estimation of the probabiliteis initial value, all the edges begin with equal prob
    credits = np.zeros(n_nodes) #initialize the credit of each nodes
    occurr_v_active = np.zeros(n_nodes) #occurrencies of each node in all episodes
    n_episodes = len(dataset) #number of episodes
    for episode in dataset: #we iterato on each episode of the dataset
        idx_w_active = np.argwhere(episode[:, node_index] == 1).reshape(-1)  #we identify in which row the target has been acivated
        if len(idx_w_active) > 0 and idx_w_active > 0: #ccheck which where the nodes active in the precius step
            active_nodes_in_prev_step = episode[idx_w_active-1,:].reshape(-1)
            credits += active_nodes_in_prev_step/np.sum(active_nodes_in_prev_step) #we assign the credits
        for v in range(0, n_nodes): #we have to check the accurrencies or each node in each step
            if(v!= node_index): #
                idx_v_active = np.argwhere(episode[:,v] == 1).reshape(-1) #in which line of our episode the node has been active
                if len(idx_v_active) > 0 and (idx_v_active<idx_w_active or len(idx_w_active)== 0): #check if the node has been active at least one in this espisode
                    occurr_v_active[v] += 1 #in this case we increase the value
    estimated_prob = credits/occurr_v_active 
    estimated_prob = np.nan_to_num(estimated_prob)
    return  estimated_prob



n_nodes = 5
n_episodes = 1000
pro_matrix = np.random.uniform(0.0,0.05,(n_nodes, n_nodes))
node_index = 4
dataset = []


for e in range(0, n_episodes):
    dataset.append(simulate_episode(init_prob_matrix= pro_matrix, n_step_max= 10 ))

estimated_prob = estimate_probabilities(dataset = dataset, node_index = node_index, n_nodes = n_nodes)

print("true  P matrix", pro_matrix[:,4])
print("Estimated p matrix", estimated_prob)


