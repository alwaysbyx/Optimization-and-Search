import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import  numpy.random as random
import matplotlib.animation as animation
import conjugate_descent
import gradient_descent
import newton_descent

def f(x1,x2,Q,b):
    return Q[0][0]*x1*x1 + Q[1][1]*x2*x2 + 2*Q[0][1]*x1*x2 + b[0][0]*x1 + b[1][0]*x2

def main():
    random.seed(1)
    n = 2
    Qsqrt = random.randn(n,n)
    b = random.randn(n,1)
    Q = np.dot(Qsqrt.T,Qsqrt)
    x1 = np.linspace(-10,10,200)
    x2 = np.linspace(-10,10,200)
    X1,X2 = np.meshgrid(x1,x2)

    X, Target  = newton_descent.newton_descent(n, Q, b)
    px = [x[0][0] for x in X]
    py = [x[1][0] for x in X]
    print(len(X))
    print(X)

    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'o-', lw=2)
    text = ax.text(.5, 1.05, '', transform = ax.transAxes, va='center')

    def init():
        ax.contourf(X1,X2,f(X1,X2,Q,b),20,alpha=0.75,cmap=plt.cm.hot)
        C=ax.contour(X1,X2,f(X1,X2,Q,b),20,colors='black',linewidths=0.5)
        ax.clabel(C,inline=True,fontsize=10)
    
    def update(i):
        line.set_data(px[:i],py[:i])
        text.set_text('step:%d'%(i))
        return line

    ani = animation.FuncAnimation(
        fig, update, interval = 200, frames=len(px)+1, init_func = init
    )
    plt.show()
    ani.save('images/newton_descent_2.gif',writer="pillow")



if __name__ == '__main__':
    main()