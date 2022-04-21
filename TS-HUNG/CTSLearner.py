from Learner import *



# Combinatorial Thompson Sampling Learner
class CTSLearner(Learner):

    def __init__(self, n_arms):
        super(CTSLearner, self).__init__(n_arms)
        self.beta_parameters = np.ones((n_arms, 2))



    #this funcion becomes useless since we know from the hungarian which arm to pull
    def pull_arm(self): #here we have to select the arm to pull
                                       #two parameters alpha and beta of the distribution
        idx = np.argmax(np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1]))
        #np.argmax select the index of the maximum value
        print("idex:", idx)

        return idx #index

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.beta_parameters[pulled_arm, 0] += reward
        self.beta_parameters[pulled_arm, 1] += (1.0 - reward)
