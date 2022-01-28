# Expectation Maximization for Student-t Mixture Models

*Keywords: Student-t distribution, mixture model, EM, PyTorch*

## Intro

This is a minimal Python package for fitting Student-t Mixture Models (STMM) to multi-dimensional datasets using the Expectation-Maximization (EM) algorithm. It is *completely* vectorized using PyTorch and is hence very, very fast compared to a naive loop-based implementation. 

Limitations: 

- It does not support learning the degree-of-freedom parameter(s), but I plan to include it in the near future. 
- It has not been tested on datasets with dimensions more than 2.
- It uses naive initialization strategies for parameters; there must be smarter ones out there.

Overall, this package is for (1) pedagogy and (2) proving the possibility of vectorizing EM for STMM. Please be cautious if you are using it for other purposes like research.

In code, I tried to follow the notation found in the original paper:

```bibtex
@article{peel2000robust,
  title={Robust mixture modelling using the t distribution},
  author={Peel, David and McLachlan, Geoffrey J},
  journal={Statistics and computing},
  volume={10},
  number={4},
  pages={339--348},
  year={2000},
  publisher={Springer}
}
```

## Example

Here's an example of fitting a 4 component STMM on a 2-dimensional dataset containing 100,000 examples:

```python
from emmixstudent import MixtureOfStudents

model = MixtureOfStudents(num_components=3, num_dimensions=2)
lower_bounds = model.fit(data)  # contains the lower bound to observed data log likelihood per timestep

# do some visualization here
```
