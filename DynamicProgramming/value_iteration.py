import numpy as np
from numpy.core.numeric import Infinity
import copy

inf = float('inf')
class value_iteration:
    def __init__(self, rows, columns, obstacle, credit,discount=0.99, error = 0.01):
        self.utility = np.zeros((rows,columns))
        self.rows = rows
        self.cols = columns
        self.ob = obstacle
        self.discount = discount
        self.error = error
        self.credit = credit
    
    def init_rewards(self):
        self.reward_matrix = np.array([[-0.04 for _ in range(self.cols)] for _ in range(self.rows)])
        for (i,j) in self.ob:
            self.reward_matrix[i][j] = -5
        for (i,j) in self.credit:
            self.reward_matrix[i][j] = 2
    
    def get_utility(self):
        self.init_rewards()
        count = 0
        while count < 2:
            epi = 0
            count += 1
            for i in range(self.rows):
                for j in range(self.cols):
                    state = (i,j)
                    u_ = copy.deepcopy(self.utility[i][j])
                    self.utility[i][j] = self.reward_matrix[i][j] + self.discount * self.find_max_action_utinity(state,self.utility)
                    epi = max(abs(self.utility[i][j] - u_),epi)
            if epi < self.error*(1-self.discount)/self.discount:
                return None
            print(self.utility)
            print(count)
    
    def get_next_utility(self, utility):
        self.init_rewards()
        for i in range(self.rows):
            for j in range(self.cols):
                state = (i,j)
                utility[i][j] = self.reward_matrix[i][j] + self.discount * self.find_max_action_utinity(state,utility)
        return utility

    def find_max_action_utinity(self,state,utility):
        i, j  = state[0], state[1]
        max_utility = -inf
        states = [(i+1,j),(i-1,j),(i,j+1),(i,j-1)]
        for (i,j) in states:
            if 0 <= i < self.rows and 0 <= j < self.cols:
                max_utility = max(max_utility, utility[i][j])
        return max_utility


if __name__ == '__main__':
    vi = value_iteration(6,5,[(1,1),(2,3)],[(1,2)])
    vi.get_utility()
