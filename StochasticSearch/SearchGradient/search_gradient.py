import math
from operator import is_
import matplotlib.pyplot as plt
from numpy.lib.npyio import _savez_compressed_dispatcher
from numpy.random import *
import  numpy.random as random
import numpy as np
from numpy.linalg import inv
import torch

class search_gradient:
    def __init__(self, N, maxite, objection_params, lr, show=True):
        self.N = N
        self.maxite = maxite
        self.lr = lr
        self.objection_params = objection_params
        self.show_animation = show

    def search(self):
        scores = []
        self.init_parameters()
        for ite in range(self.maxite):
            if ite==30:
                self.lr = 0.1
            samples = self.get_samples()
            x = [item[0] for item in samples]
            y = [item[1] for item in samples]
            scoredsamples = self.get_scores(samples)
            self.update_parameters(scoredsamples)
            best_score = scoredsamples[0][0]
            scores.append(best_score)
            print(f'iteration={ite}, score={best_score}')
            if self.show_animation:
                ax = self.init_plot()
                #plt.cla()
                ax.plot(x,y,'o', color='black')
                ax.plot(self.best_sample[0],self.best_sample[1],'o',color='red')
                plt.title('search_gradient')
                a1 = plt.annotate(f'step:{ite}\n f:{round(best_score,2)}', xy=(0.85, 0.9), xycoords='axes fraction',color='black')
                plt.pause(0.001)
                if ite != self.maxite-1:
                    a1.remove()
        if self.show_animation:
            plt.pause(0)
    
    def init_parameters(self):
        '''init parameter trainsition_matrix and get samples according to this'''
        self.mu = np.array([[-2.],[-2.]])
        self.sigma = np.array([[0.1,0.],[0,0.1]])

    def get_samples(self):
        '''generate n samples according to parameters'''
        samples = []
        while len(samples)!=self.N:
            sample = multivariate_normal(self.mu.squeeze(),self.sigma)
            if sample[0] > -8 and sample[0] < 8 and sample[1] > -8 and sample[1] < 8:
                samples.append(sample)
        return samples

    def get_scores(self, samples):
        '''choose the elite set from n samples generated'''
        scores = []
        for sample in samples:
            scores.append(self.get_function_value(sample))
        scoredsamples = list(zip(scores,samples))
        scoredsamples = sorted(scoredsamples, key=lambda x: x[0])
        self.best_sample = scoredsamples[0][1]
        self.samples = [item[1] for item in scoredsamples]
        return scoredsamples
    
    def update_parameters(self,samples):
        '''using the new elite set to update parameters'''
        Dmu = 0.
        Dsigma = 0.
        is_ = inv(self.sigma)
        for sample in samples:
            score = sample[0]
            xy = sample[1]
            #print('score',score)
            #print('xy',xy)
            xy = xy.reshape(2,1)
            dmu = np.dot(is_,(xy-self.mu))
            Dmu += dmu*score
            dsigma = -1/2*is_ + 1/2*np.dot(np.dot(dmu,(xy-self.mu).T),is_)
            Dsigma += dsigma*score
        Dmu /= len(samples)
        Dsigma /= len(samples)
        self.mu -= self.lr * Dmu
        self.sigma -= self.lr * Dsigma
        print(self.mu,self.sigma)

    def get_function_value(self,sample,plot=False):
        '''
        here the function value (what we are going to minimize) is f([x,y])
        '''
        X = sample[0]
        Y = sample[1]
        XY = np.concatenate((np.expand_dims(X, -1), np.expand_dims(Y, -1)), -1)
        XY = torch.from_numpy(XY)
        params = self.objection_params
        X, Y = torch.split(XY, [1,1], dim=-1)
        Z = 0.
        for i in range(len(params)//3):
            Z -= params[3*i]*torch.exp(-(X-params[3*i+1])**2 - (Y-params[3*i+2])**2)
        Z += 0.15*X*torch.cos(Y)
        if not plot:
            Z = Z.detach().numpy()[0]
        else:
            Z = Z.squeeze(-1).detach().numpy()
        return Z

    def init_plot(self):
        N = 300
        x = np.linspace(-8.0, 8.0, N)
        y = np.linspace(-8.0, 8.0, N)
        X, Y = np.meshgrid(x, y)
        Z = self.get_function_value([X,Y],True)
        fig, ax = plt.subplots(figsize=(12,8))
        CS = ax.contourf(X, Y, Z, cmap='Blues')
        return ax
    
def main():
    random.seed(10)
    params = [1., 1., 3., 0.5, 1., 3., 1., -2., 2.]
    sg = search_gradient(N=50, maxite=300, objection_params=params, lr=0.5)
    sg.search()

if __name__ == '__main__':
    main()