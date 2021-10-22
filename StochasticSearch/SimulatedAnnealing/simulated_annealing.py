import math
import matplotlib.pyplot as plt
from numpy.random import *
import  numpy.random as random
from scipy.spatial.distance import pdist
import numpy as np
"""
t0: temperature
alpha: cooling coefficient here we use 'exponential annealing'
n:  the number of nodes on the map
x:  position x array with (N*1) size
y:  position y array with (N*1) size
"""

class simulated_annealing:
    def __init__(self, x, y, t0, alpha, show):
        self.t0 = t0 
        self.alpha = alpha
        self.x = x
        self.y = y
        self.n = len(x)
        self.node = [[x[i],y[i]] for i in range(self.n)]
        self.distance = pdist(self.node)
        self.outIte = 300
        self.inIte = 20
        self.p = np.array([0.3, 0.3, 0.4]) #swap/reversion/insertion
        self.show_animation = show
    
    def search(self):
        r0 = permutation(self.n)
        f0 = self.get_function_value(r0)
        bestR = r0
        bestF = f0
        bestFOutlier = [bestF]
        T = self.t0
        for outite in range(self.outIte):
            for inite in range(self.inIte):
                r1 = self.get_neighbor(r0)
                #print(f'r0={r0},r1={r1}\n')
                f1 = self.get_function_value(r1)
                if f1 <= f0:
                    r0 = r1
                    f0 = f1
                else:
                    delta = (f1-f0) / f0
                    p = np.exp(-delta/T)
                    if random.random() <= p:
                        r0 = r1
                        f0 = f1
                if f0 < bestF:
                    bestR = r0
                    bestF = f0
            bestFOutlier.append(bestF)
            print(f'ite={outite}, f_value = {bestF}\n')
            T = self.alpha * T
            if self.show_animation:
                plt.cla()
                route = r0.copy()
                route = np.append(route,r0[0])
                #print(route)
                #print(np.array(self.x)[route.astype(int)])
                plt.plot(np.array(self.x)[route.astype(int)],np.array(self.y)[route.astype(int)],'o-', lw=2, color='orange')
                plt.plot(self.x,self.y,'o',color='red')
                a1 = plt.annotate(f'step:{outite}\n f:{round(bestF,2)}', xy=(0.85, 0.9), xycoords='axes fraction',color='black')
                a2 = plt.annotate(f'T={self.t0}', xy=(0.5, 1.05), xycoords='axes fraction',color='black')
                plt.axis('equal')
                plt.pause(0.001)
                if outite != self.outIte-1:
                    a1.remove()
                    a2.remove()
        if self.show_animation:
            plt.pause(0)
        

    def get_function_value(self,x0):
        f = 0
        x = x0.copy()
        x = np.append(x,x[0])
        for i in range(self.n):
            u = x[i]
            v = x[i+1]
            f += (self.x[u]-self.x[v])**2 + (self.y[u]-self.y[v])**2
        return f
    
    def get_neighbor(self,x):
        index = choice([1,2,3],p=self.p.ravel())
        if index==1:
            #print('swap')
            newx = self.swap(x)
        elif index==2:
            #print('reversion')
            newx = self.reversion(x)
        else:
            #print('insertion')
            newx = self.insertion(x)
        return newx

    def reversion(self,x):
        newx = x.copy()
        tmp = permutation(self.n)
        s1 = min(tmp[:2])
        s2 = max(tmp[:2])
        newx[s1:s2] = newx[s1:s2][::-1]
        return newx
    
    def swap(self,x):
        newx = x.copy()
        tmp = permutation(self.n)
        s1 = min(tmp[:2])
        s2 = max(tmp[:2])
        tmp = newx[s1]
        newx[s1] = newx[s2]
        newx[s2] = tmp
        return newx
    
    def insertion(self,x):
        newx = x.copy()
        tmp = permutation(self.n)
        s1 = tmp[0]
        s2 = tmp[1]
        to_insert = newx[s1]
        newx = np.delete(newx,s1)
        if s1 < s2:
            newx = np.insert(newx,s2,to_insert)
        else:
            newx = np.insert(newx,s2+1,to_insert)
        return newx

def point(x0,y0,r):
    theta = random.random() * 2 * math.pi
    return [x0 + math.cos(theta)*r, y0 + math.sin(theta)*r]
    
def main():
    random.seed(10)
    n = 30
    xy = [point(0,0,2) for _ in range(n)]
    x = [point[0] for point in xy]
    y = [point[1] for point in xy]
    #plt.scatter(x,y)
    #plt.axis("equal")
    #plt.show()
    sl = simulated_annealing(x, y, t0=0.5, alpha=0.99, show=True)
    sl.search()

if __name__ == '__main__':
    main()