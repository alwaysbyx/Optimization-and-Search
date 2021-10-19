import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import  numpy.random as random
import matplotlib.animation as animation

show_animation = True
save = True
if save:
    ims = []
def compute_gradient(x,Q,b):
    return np.dot(Q,x) + b

def compute_target(x,Q,b):
    target = np.dot(np.dot(x.T,Q),x) + np.dot(b.T,x)
    return target[0][0]

def gradient_descent(n,Q,b):
    rho = 1
    x = random.randn(n,1)
    lr = 0.01
    g = compute_gradient(x, Q, b)
    target = compute_target(x, Q, b)
    Target = [target]
    fig = plt.figure()
    while np.dot(g.T,g) > rho:
        x = x - lr * g
        g = compute_gradient(x, Q, b)
        Target.append(compute_target(x, Q, b))
        if show_animation:  
            #plt.cla()
            im = plt.plot(Target,color='gray')
            plt.axis("equal")
            plt.pause(0.001)
            if save:
                ims.append(im)
    if save:
        print(len(ims))
        ani = animation.ArtistAnimation(fig, ims, interval=50)
        ani.save('images/gradient_descent_1.gif',writer="pillow")


def main():
    random.seed(10)
    n = 10
    Qsqrt = random.randn(n,n)
    b = random.randn(n,1)
    Q = np.dot(Qsqrt.T,Qsqrt)
    gradient_descent(n,Q,b)

if __name__ == '__main__':
    main()