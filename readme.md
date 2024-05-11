# Plain Gaussain
Plain Gaussain is a python package that implements a simple probabilistic programming language for Gaussian random variables with exact conditioning and the support of array variables.

## Requirements
The core fnctionality requires:
* python >= 3.6
* [numpy](https://numpy.org/)
* [scipy](https://scipy.org/)

Using parametric distributions also requires: 
* [jax](https://jax.readthedocs.io/)

## Installation

This repo contains a python package that can be installed as usual, e.g.:

1) Download the project folder 
2) Navigate the terminal to the project folder and execute `pip install .` , in which case the command will copy the files to the standard location of python packages, or `pip install -e .` , in which the command will reference the files in the folder. 


## References
1. D. Stein and S. Staton, "Compositional Semantics for Probabilistic Programs with Exact Conditioning," 2021 36th Annual ACM/IEEE Symposium on Logic in Computer Science (LICS), Rome, Italy, 2021, pp. 1-13, doi: 10.1109/LICS52264.2021.9470552.