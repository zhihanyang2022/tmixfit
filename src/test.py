import numpy as np
import torch
from tmixfit import STMMLoop, STMMVectorized


np.random.seed(42)

# Create a small synthetic dataset

cov_multiplier = 2
n_per_component = 20

dataset_1 = np.random.multivariate_normal(mean=np.array([1, 1]), cov=np.array([
    [0.1, 0.05],
    [0.05, 0.1]
]) * cov_multiplier, size=n_per_component)  # top right

dataset_2 = np.random.multivariate_normal(mean=np.array([-1,-1]), cov=np.array([
    [0.1, 0.05],
    [0.05, 0.1]
]) * cov_multiplier, size=n_per_component)  # bottom left

dataset_3 = np.random.multivariate_normal(mean=np.array([1,-1]), cov=np.array([
    [0.1, -0.05],
    [-0.05, 0.1]
]) * cov_multiplier, size=n_per_component)  # bottom right

dataset_4 = np.random.multivariate_normal(mean=np.array([-1,1]), cov=np.array([
    [0.1, -0.05],
    [-0.05, 0.1]
]) * cov_multiplier, size=n_per_component)  # top left

dataset = np.vstack([dataset_1, dataset_2, dataset_3, dataset_4])
dataset_torch = torch.from_numpy(dataset)

# Sychronize parameters

model_loop = STMMLoop(p=2, g=4, v=2)
model_vectorized = STMMVectorized(
    p=2, g=4, v=2,
    pi_init=torch.from_numpy(model_loop.pi),
    mus_init=torch.from_numpy(model_loop.mus),
    Sigmas_init=torch.from_numpy(model_loop.Sigmas)
)

# Sanity check

assert np.allclose(model_loop.pi, model_vectorized.pi.numpy())
assert np.allclose(model_loop.mus, model_vectorized.mus.numpy())
assert np.allclose(model_loop.Sigmas, model_vectorized.Sigmas.numpy())

loglik_loop = model_loop.loglik(dataset)
loglik_torch = model_vectorized.loglik(dataset_torch)
assert np.allclose(loglik_loop, loglik_torch), (loglik_loop, loglik_torch)

# Train one iteration at a time and compare parameter values

for iter in range(20):

    model_loop.fit_one_iter(dataset)
    model_vectorized.fit_one_iter(dataset_torch)

    loglik_loop = model_loop.loglik(dataset)
    loglik_torch = model_vectorized.loglik(dataset_torch)
    assert np.allclose(loglik_loop, loglik_torch), (loglik_loop, loglik_torch)
    assert np.allclose(model_loop.pi, model_vectorized.pi.numpy())
    assert np.allclose(model_loop.mus, model_vectorized.mus.numpy())
    assert np.allclose(model_loop.Sigmas, model_vectorized.Sigmas.numpy())

print("All tests passed!")
