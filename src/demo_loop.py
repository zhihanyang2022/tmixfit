# test whether both moethods give same parametesr if initialized the same way
# test whether both methods converge to reasonable parameters

import time
import numpy as np
import matplotlib.pyplot as plt
from tmixfit import STMMLoop


np.random.seed(42)

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

v = 2
model = STMMLoop(p=2, g=4, v=v)

start = time.time()

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
densities = model.pdf(all_coords).reshape(200, 200)
plt.figure(figsize=(5, 5))
ax = plt.gca()
ax.contourf(xxs, yys, densities, cmap='coolwarm')
# cset = ax.contour(xxs, yys, densities, colors='k')
plt.scatter(dataset[:, 0], dataset[:, 1], alpha=1, color='white', edgecolor='black', s=25)
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.show()
# plt.savefig(f"../four_clusters_v={v}.png", dpi=300)
