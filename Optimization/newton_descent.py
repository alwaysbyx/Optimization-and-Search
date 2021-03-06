import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import  numpy.random as random
import matplotlib.animation as animation
from numpy.linalg import *

show_animation = True

def compute_gradient(x,Q,b):
    g1 = np.dot(Q,x) + b
    g2 = Q
    return np.dot(inv(g2), g1)

def compute_target(x,Q,b):
    target = np.dot(np.dot(x.T,Q),x) + np.dot(b.T,x)
    return target[0][0]

def newton_descent(n,Q,b):
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
    X = [x]
    g = compute_gradient(x, Q, b)
    target = compute_target(x, Q, b)
    Target = [target]
    fig = plt.figure()
    step = 0
    while np.dot(g.T,g) > rho:
        step += 1
        x = x - g
        X.append(x)
        g = compute_gradient(x, Q, b)
        Target.append(compute_target(x, Q, b))
        if show_animation:  
            #plt.cla()
            im = plt.plot(Target,color='gray')
            anno = plt.annotate('step:%d'%step, xy=(0.85, 0.9), xycoords='axes fraction',color='black')
            #plt.axis("equal")
            plt.xlim(-0.1,2)
            plt.pause(0.1)
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
    newton_descent(n,Q,b)

if __name__ == '__main__':
    main()