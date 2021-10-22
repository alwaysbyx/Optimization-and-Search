# Optimization-and-Search
Implementation and visualization of optimization algorithms.  
please add __MAthJax Plugin for Github__ to your browser.


# 1. Numerical Optimization    

Here we are trying to solve simple quadratic problem.  
$$\arg \text{min } \frac{1}{2}x^TQx + b^Tx$$
$$Q \geq 0, x \in R^n$$    

The animation(left) is tested on N=10, (right) for n=2;


## Gradient Descent   
using first-order gradient and learning rate

<div align=center>
<img width="48%" src="images/gradient_descent_1.gif"/>
<img width="48%" src="images/gradient_descent_2.gif"/>
</div>

## Conjugate Descent    
$x^{k+1} := x^k + a_kd^k$  
using line search to compute the step size $\alpha$  
$$a_k = \frac{\nabla f(x^k)^Td^k}{(d^k)^TQd^k}$$  
find conjugate direction  
$$d^{k+1} = -\nabla f(x^{k+1}) + \beta_kd^k$$    
$$ \beta_k = \frac{\nabla f(x^{k+1})^T\nabla f(x^{k+1})}{\nabla f(x^k)^T\nabla f(x^k)} \text{ (FR)}$$ 

<div align=center>
<img width="48%" src="images/conjugate_descent_1.gif"/>
<img width="48%" src="images/conjugate_descent_2.gif"/>
</div>

## Newton Method   
using second-order gradient  
$x^{k+1} := x^k + d^k$  
$d^k = -[\nabla^2 f(x^k)]^{-1}\nabla f(x^k)$
<div align=center>
<img width="48%" src="images/newton_descent_1.gif"/>
<img width="48%" src="images/newton_descent_2.gif"/>
</div>

# 2. Stochastic Search    
Here we try to use stochastic search to solve TCP problems.  
## Simulated Annealing    
<div align=center>
<img width="48%" src="images/SA.gif"/>
<img width="48%" src="images/SA2.gif"/>
</div>


## Cross-Entropy Methods


## Search Gradient


# 3. Classical Search    

## A* search   



