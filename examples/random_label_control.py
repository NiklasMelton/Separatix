"""Random label control example."""

import numpy as np
from sklearn.datasets import make_blobs

from separatix import diagnose

X, y = make_blobs(n_samples=250, centers=3, n_features=8, random_state=0)
rng = np.random.default_rng(0)
print(diagnose(X, rng.permutation(y), random_state=0))
