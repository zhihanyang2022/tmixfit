# em-mix-student

A minimal Python package for fitting mixture of Student distributions (i.e., t distributions) to multi-dimensional datasets using the Expectation-Maximization (EM) algorithm.

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
