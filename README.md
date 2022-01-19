# em-mix-student

*This package does not support closed-source distributions, as it's under GNU-GPLv3 license.*

A minimal Python package for fitting mixture of Student distributions (i.e., t distributions) to multi-dimensional datasets using the Expectation-Maximization (EM) algorithm.

```python
from emmixstudent import MixtureOfStudents

model = MixtureOfStudents(num_components=3, num_dimensions=2)
lower_bounds = model.fit(data)  # contains the lower bound to observed data log likelihood per timestep

# do some visualization here
```
