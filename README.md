# em-mix-student

A minimal Python package for fitting mixture of Student distributions (i.e., t distributions) to multi-dimensional datasets using the Expectation-Maximization (EM) algorithm.

```python
from emmixstudent import MixtureOfStudents

model = MixtureOfStudents(num_components=3, num_dimensions=2)
model.fit(data)
```
