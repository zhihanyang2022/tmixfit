# test whether both moethods give same parametesr if initialized the same way
# test whether both methods converge to reasonable parameters

import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from tmixfit import STMMVectorized

cov_multiplier = 1
n_per_component = 2500

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
dataset = torch.from_numpy(dataset)

model = STMMVectorized(p=2, g=4, v=100)

start = time.time()

with torch.no_grad():
    lls = model.fit(dataset, num_iters=50)

end = time.time()

print(end - start)

# plt.figure(figsize=(5, 5))
# plt.plot(lls)
# plt.xlabel('Number of Iterations'); plt.ylabel('Log-Likelihood')
# plt.show()

xs = np.linspace(-2.5, 2.5, 200)
ys = np.linspace(-2.5, 2.5, 200)
xxs, yys = np.meshgrid(xs, ys)
xs, ys = xxs.flatten(), yys.flatten()
all_coords = np.hstack([xs.reshape(-1, 1), ys.reshape(-1, 1)])
densities = model.pdf(torch.from_numpy(all_coords)).numpy().reshape(200, 200)
plt.figure(figsize=(5, 5))
ax = plt.gca()
ax.contourf(xxs, yys, densities, cmap='coolwarm')
cset = ax.contour(xxs, yys, densities, colors='k')
# plt.scatter(dataset[:, 0], dataset[:, 1], alpha=1, color='white', edgecolor='black', s=50)
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.show()
