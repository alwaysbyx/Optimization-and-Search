import numpy as np
from numpy.core.numeric import Infinity
import copy

inf = float('inf')
class policy_iteration:
    def __init__(self, rows, columns, obstacle, credit,discount=0.99, error = 0.01):
        self.utility = np.zeros((rows,columns))
        self.rows = rows
        self.cols = columns
        self.ob = obstacle
        self.discount = discount
        self.error = error
        self.credit = credit
        self.k = 4
    
    def init_rewards(self):
        self.reward_matrix = np.array([[-0.04 for _ in range(self.cols)] for _ in range(self.rows)])
        for (i,j) in self.ob:
            self.reward_matrix[i][j] = -5
        for (i,j) in self.credit:
            self.reward_matrix[i][j] = 2
    
    def init_policy(self):
        policy = np.array([[np.random.choice(4) for _ in range(self.cols)] for _ in range(self.rows)]) #0up,1down,2left,3right
        return policy
    
    def get_utility(self, utility, policy):
        '''
        update utility function based a simplified version of Bellman  Equation (assume a policy)
        '''
        u = copy.deepcopy(utility)
        self.init_rewards()
        for _ in range(self.k):
            for i in range(self.rows):
                for j in range(self.cols):
                    if policy[i][j] == 0:
                        nextstate = (i-1,j)
                    elif policy[i][j] == 1:
                        nextstate = (i+1,j)
                    elif policy[i][j] == 2:
                        nextstate = (i,j-1)
                    elif policy[i][j] == 3:
                        nextstate = (i,j+1)
                    if 0 <= nextstate[0] < self.rows and 0 <= nextstate[1] < self.cols:
                        u[i][j] = self.reward_matrix[i][j] + self.discount * utility[nextstate[0]][nextstate[1]]
        return u

    def get_policy(self):
        '''
        main function: returns a policy
        '''
        policy = self.init_policy()
        self.get_utility(policy)
        unchanged = False
        print(policy)
        while not unchanged:
            unchanged = True
            for i in range(self.rows):
                for j in range(self.cols):
                    state = (i,j)
                    action = self.find_best_action(state, self.utility)
                    if action != policy[i][j]:
                        policy[i][j] = action
                        unchanged = False
    
    def get_next_policy(self, utility, policy):
        unchanged = True
        for i in range(self.rows):
            for j in range(self.cols):
                state = (i,j)
                action = self.find_best_action(state, utility)
                if action != policy[i][j]:
                    policy[i][j] = action
                    unchanged = False
        return policy, unchanged

    def find_best_action(self,state,utility):
        i, j  = state[0], state[1]
        states = [(i-1,j),(i+1,j),(i,j-1),(i,j+1)]
        utilities = []
        for (i,j) in states:
            if 0 <= i < self.rows and 0 <= j < self.cols:
                utilities.append(utility[i][j])
            else:
                utilities.append(-inf)
        max_u = max(utilities)
        action = utilities.index(max_u)
        return action

if __name__ == '__main__':
    pi = policy_iteration(6,5,[(1,1),(2,3)],[(1,2)])
    pi.get_policy()