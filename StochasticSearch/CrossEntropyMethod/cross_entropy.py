import math
import matplotlib.pyplot as plt
from numpy.lib.npyio import _savez_compressed_dispatcher
from numpy.random import *
import  numpy.random as random
from scipy.spatial.distance import pdist,squareform
import numpy as np
"""
x:  list N size: position x 
y:  list N size: position y 
N:  A int, the amount of samples generated at each iteration
maxite: A int, iterations
alpha: A float, using to update parameters for smoothness
rho: A float which corresponds to the quantile of an empirical distribution.
"""
class cross_entropy:
    def __init__(self, x, y, N, maxite,alpha,rho,show=True):
        self.x = x
        self.y = y
        self.n = len(x)
        self.N = N
        self.node = [[x[i],y[i]] for i in range(self.n)]
        self.distance = squareform(pdist(self.node))
        self.maxite = maxite
        self.alpha = alpha
        self.rho = rho
        self.show_animation = show

    def search(self):
        scores = []
        self.init_parameters()
        for ite in range(self.maxite):
            samples = self.get_samples()
            samples = self.choose_samples(samples)
            self.update_parameters(samples)
            score = samples[0][0]
            scores.append(score)
            print(f'iteration={ite}, score={score}')
            if self.show_animation:
                plt.cla()
                #oute = np.array(samples[0][1])
                #plt.plot(np.array(self.x)[route.astype(int)],np.array(self.y)[route.astype(int)],'o-', lw=2, color='orange')
                #plt.plot(self.x,self.y,'o',color='red')
                plt.title('cross-entropy methods')
                plt.plot(scores)
                a1 = plt.annotate(f'step:{ite}\n f:{round(score,2)}', xy=(0.85, 0.9), xycoords='axes fraction',color='black')
                plt.axis('equal')
                plt.pause(0.001)
                if ite != self.maxite-1:
                    a1.remove()
        if self.show_animation:
            plt.pause(0)

    
    def init_parameters(self):
        '''init parameter trainsition_matrix and get samples according to this'''
        p_ij = 1 / (self.n - 1)
        trans_mat = np.full_like(self.distance, p_ij, float)
        np.fill_diagonal(trans_mat, 0)
        self.trans_mat = trans_mat
        self.trans_mat_old = trans_mat

    def get_sample(self):
        '''generate one sample according to parameters'''
        trans_mat = self.trans_mat.copy()
        x = [0]
        for i in range(self.n-1):
            trans_mat[:,x[-1]] = 0
            row_sum = trans_mat.sum(axis=1)
            trans_mat = (trans_mat.T / row_sum).T
            choice = random.choice(np.arange(self.n),p=trans_mat[x[-1],:])
            x.append(choice)
        x.append(0)
        return x

    def get_samples(self):
        '''generate n samples according to parameters'''
        return [self.get_sample() for _ in range(self.N)]

    def choose_samples(self, samples):
        '''choose the elite set from n samples generated'''
        num = int(self.rho * self.N)
        scores = []
        for sample in samples:
            scores.append(self.get_function_value(sample))
        scoredsamples = list(zip(scores,samples))
        scoredsamples = sorted(scoredsamples, key=lambda x: x[0])
        return scoredsamples[:num+1]
    
    def update_parameters(self,samples):
        '''using the new elite set to update parameters'''
        self.trans_mat_old = self.trans_mat.copy()
        trans_mat = np.zeros_like(self.trans_mat)
        for sample in samples:
            for idx in range(self.n-1):
                i = sample[1][idx]
                j = sample[1][idx+1]
                trans_mat[i,j] += 1
        trans_mat = trans_mat / len(samples)
        self.trans_mat = self.alpha * trans_mat + (1-self.alpha) * self.trans_mat_old
    
    def get_function_value(self,x0):
        '''
        here the function value (what we are going to minimize) is the total length of path
        '''
        f = 0
        for i in range(self.n):
            f += self.distance[x0[i],x0[i+1]]
        return f


def point(x0,y0,r):
    theta = random.random() * 2 * math.pi
    return [x0 + math.cos(theta)*r, y0 + math.sin(theta)*r]
    
def main():
    random.seed(10)
    n = 30
    xy = [point(0,0,2) for _ in range(n)]
    x = [point[0] for point in xy]
    y = [point[1] for point in xy]
    sl = cross_entropy(x, y, N=200, maxite=300, alpha=0.9, rho=0.05)
    sl.search()

if __name__ == '__main__':
    main()