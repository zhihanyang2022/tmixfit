"""
test.py

A simple script to test whether the loop-based and vectorized implementations are identical numerically.
Useful for debugging the vectorized implementation.

Simply call "python test.py" to use this script.

First working on: Feb 4, 2022
"""

import numpy as np
import torch
from tmixfit import STMMLoop, STMMVectorized
from copy import deepcopy
from tqdm import tqdm

np.random.seed(42)  # for creating consistent datasets across runs
torch.manual_seed(42)  # for creating consistent initial parameters across runs

# Create a small synthetic dataset

cov_multiplier = 2
n_per_component = 5

dataset_1 = np.random.multivariate_normal(mean=np.array([1, 1]), cov=np.array([
    [0.1, 0.05],
    [0.05, 0.1]
]) * cov_multiplier, size=n_per_component)  # top right

dataset_2 = np.random.multivariate_normal(mean=np.array([-1, -1]), cov=np.array([
    [0.1, 0.05],
    [0.05, 0.1]
]) * cov_multiplier, size=n_per_component)  # bottom left

dataset_3 = np.random.multivariate_normal(mean=np.array([1, -1]), cov=np.array([
    [0.1, -0.05],
    [-0.05, 0.1]
]) * cov_multiplier, size=n_per_component)  # bottom right

dataset_4 = np.random.multivariate_normal(mean=np.array([-1, 1]), cov=np.array([
    [0.1, -0.05],
    [-0.05, 0.1]
]) * cov_multiplier, size=n_per_component)  # top left

dataset = np.vstack([dataset_1, dataset_2, dataset_3, dataset_4])
dataset_torch = torch.from_numpy(dataset)

assert np.allclose(dataset, dataset_torch.numpy())

# Sychronize parameters

# The reason why I'm using deepcopy here is because the numpy() method actually (surprisingly!) returns a reference to
# the original tensor. If the numpy version is modified, the original PyTorch tensor is modified.

model_vectorized = STMMVectorized(
    p=2, g=4, v=3, tune_v=True
)
model_loop = STMMLoop(
    p=2, g=4, v=3,
    pi_init=deepcopy(model_vectorized.pi.numpy()),
    mus_init=deepcopy(model_vectorized.mus.numpy()),
    Sigmas_init=deepcopy(model_vectorized.Sigmas.numpy()),
    tune_v=True
)

# Sanity check

assert np.allclose(model_loop.pi, model_vectorized.pi.numpy())
assert np.allclose(model_loop.mus, model_vectorized.mus.numpy())
assert np.allclose(model_loop.Sigmas, model_vectorized.Sigmas.numpy())

loglik_loop = model_loop.loglik(dataset)
loglik_torch = model_vectorized.loglik(dataset_torch)
assert np.allclose(loglik_loop, loglik_torch), (loglik_loop, loglik_torch)

print("Sanity tests passed!")

# Train one iteration at a time and compare parameter values

for iteration in tqdm(range(10)):

    model_loop.fit_one_iter(dataset)
    model_vectorized.fit_one_iter(dataset_torch)

    assert np.allclose(model_loop.tau_matrix, model_vectorized.tau_matrix.numpy())
    assert np.allclose(model_loop.u_matrix, model_vectorized.u_matrix.numpy())

    loglik_loop = model_loop.loglik(dataset)
    loglik_torch = model_vectorized.loglik(dataset_torch)

    assert np.allclose(loglik_loop, loglik_torch)
    assert np.allclose(model_loop.pi, model_vectorized.pi.numpy())
    assert np.allclose(model_loop.mus, model_vectorized.mus.numpy())
    assert np.allclose(model_loop.Sigmas, model_vectorized.Sigmas.numpy())

print("All tests passed!")
