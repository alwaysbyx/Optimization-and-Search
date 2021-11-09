import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import  numpy.random as random
import matplotlib.animation as animation
from numpy.linalg import *

show_animation = True


def compute_gradient(x,Q,b):
    g1 = np.dot(Q,x) + b
    return g1

def compute_target(x,Q,b):
    target = np.dot(np.dot(x.T,Q),x) + np.dot(b.T,x)
    return target[0][0]

def conjugate_descent(n,Q,b):
    '''
    input: 
        n: dimension
        Q: Q matrix Rn*n
        b: b vector Rn
    output:
        X: all searched x
        target: optimal x
    '''
    rho = 0.1
    x = random.randn(n,1)
    g = compute_gradient(x, Q, b)
    d = -g
    target = compute_target(x, Q, b)
    Target = [target]
    fig = plt.figure()
    step = 0
    X = [x]
    while np.dot(g.T,g) > rho:
        step += 1
        a = np.dot(g.T,g) / np.dot(np.dot(d.T,Q),d)
        x = x + a * d
        pre_g = g
        g = compute_gradient(x, Q, b)
        beta = np.dot(g.T,g) / np.dot(pre_g.T,pre_g)
        d = -g + beta * d
        Target.append(compute_target(x, Q, b))
        X.append(x)
        if show_animation:  
            im = plt.plot(Target,color='gray')
            anno = plt.annotate('step:%d'%step, xy=(0.85, 0.9), xycoords='axes fraction',color='black')
            plt.axis("equal")
            plt.pause(0.001)
            anno.remove()
    if show_animation:
        anno = plt.annotate('step:%d'%step, xy=(0.85, 0.9), xycoords='axes fraction',color='black')
        plt.pause(0)
    return X, Target
    
def main():
    random.seed(10)
    n = 10
    Qsqrt = random.randn(n,n)
    b = random.randn(n,1)
    Q = np.dot(Qsqrt.T,Qsqrt)
    conjugate_descent(n,Q,b)

if __name__ == '__main__':
    main()