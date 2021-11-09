import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import  numpy.random as random
import matplotlib.animation as animation

show_animation = True


def compute_gradient(x,Q,b):
    return np.dot(Q,x) + b

def compute_target(x,Q,b):
    target = np.dot(np.dot(x.T,Q),x) + np.dot(b.T,x)
    return target[0][0]

def gradient_descent(n,Q,b):
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
    lr = 0.01
    g = compute_gradient(x, Q, b)
    target = compute_target(x, Q, b)
    Target = [target]
    fig = plt.figure()
    step = 0
    while np.dot(g.T,g) > rho:
        step += 1
        x = x - lr * g
        X.append(x)
        g = compute_gradient(x, Q, b)
        Target.append(compute_target(x, Q, b))
        if show_animation:  
            #plt.cla()
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
    gradient_descent(n,Q,b)

if __name__ == '__main__':
    main()