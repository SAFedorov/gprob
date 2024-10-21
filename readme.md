# gprob
gprob is a python package that implements a probabilistic programming language for Gaussian random variables with exact conditioning. It is built around the idea that arrays of Gaussian random variables can be handled in the same way as numerical numpy arrays.

To give a flavor of it, the first example shows a few manipulations with scalar variables,
```python
>>> import gprob as gp
>>> x = gp.normal()
>>> y = gp.normal()
>>> z = x + 0.2 * y + 3
>>> z
Normal(mean=3, var=1.04)
>>> gp.cov(z, y)  # covariance
array(0.2)
>>> z | {y - 0.5 * x: 1}  # conditioning
Normal(mean=2.76, var=0.968)
```

The second example constructs the distribution of trajectories of a Brownian particle observed in the beginning at x=0 and midway through its motion at x=1,
```python
>>> nstep = 5 * 10**3
>>> dx = gp.normal(0, 1/nstep, size=(nstep,))
>>> x = gp.cumsum(dx, 0)  # unconditional particle positions
>>> xc = x | {x[nstep//2]: 1}  # positions conditioned on x[nstep//2] == 1
>>> samples = xc.sample(10**2)  # sampling 100 trajectories
```
Visualizing the sampled trajectories,
```python
>>> import matplotlib.pyplot as plt
>>> plt.plot(samples.T, alpha=0.1, color='gray')
>>> plt.show()
```
![brownian readme](./assets/brownian_readme.png)

## Requirements
* python >= 3.9
* numpy >= 1.25
* scipy

## Installation
The package can be installed from PyPI,
```
pip install gprob
```

or from this repository (to get the latest version),

```
pip install git+https://github.com/SAFedorov/gprob
```

## Getting started
Have a look at the notebooks in the [examples](examples) folder, starting from the tutorials on
1. [Random variables](examples/1-random-variables.ipynb)
2. [Array operations](examples/2-array-operations.ipynb)
3. [Sparse arrays](examples/3-sparse-arrays.ipynb)
4. [Likelihood fitting](examples/4-likelihood-fitting-fisher.ipynb)

roughly in this order.

## How it works
There is a [supplementary note](https://safedorov.github.io/gprob-note/) that presents some of the underying theory, focusing on the theory of inference.

## On the number of variables
General multivariate Gaussian distributions of *n* variables require memory quadratic in *n* for their storage and computation time cubic in *n* for their exact conditioning, meaning that the necessary resources increase fast with the random variable size. 
My laptop can typically handle arrays of random variables whose sizes count in thousands.

If the Gaussian variables are such that their joint distribution is a direct product, they can be packed into sparse arrays. For those, memory and computational requirements grow linearly with the number of independent distributions, and the total number of variables can be larger. 

## Acknowledgements
gprob was inspired by (but works differently from) [GaussianInfer](https://github.com/damast93/GaussianInfer). See the corresponding paper,

D. Stein and S. Staton, "Compositional Semantics for Probabilistic Programs with Exact Conditioning," 2021 36th Annual ACM/IEEE Symposium on Logic in Computer Science (LICS), Rome, Italy, 2021, pp. 1-13, doi: 10.1109/LICS52264.2021.9470552 .

gprob uses the subscript parser from [opt-einsum](https://github.com/dgasmith/opt_einsum). Some linearization tricks and choices of tooling follow [autograd](https://github.com/HIPS/autograd).

