"""Sparse example."""

import numpy as np
from scipy import sparse

from separatix import diagnose

rng = np.random.default_rng(0)
X = sparse.random(400, 800, density=0.02, random_state=0, data_rvs=rng.standard_normal)
y = np.array([0] * 200 + [1] * 200)
report = diagnose(X, y, return_report=True, random_state=0, max_dense_mb=4)
print(report.recommendation_text)
print(report.densification_events)
