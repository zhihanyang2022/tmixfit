# em-mix-student

This is a minimal Python package for fitting Student-t Mixture Models (STMM) to multi-dimensional datasets using the Expectation-Maximization (EM) algorithm. It is *completely* vectorized and is hence very fast compared to a naive loop-based implementation. Currently, it does not support learning the degree-of-freedom parameter, but I plan to include it in the near future. 

Here's an example of fitting a 4 component STMM on a 2-dimensional dataset containing 100,000 examples:

```python
from emmixstudent import MixtureOfStudents

model = MixtureOfStudents(num_components=3, num_dimensions=2)
lower_bounds = model.fit(data)  # contains the lower bound to observed data log likelihood per timestep

# do some visualization here
```

If you use this package, please cite:

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
