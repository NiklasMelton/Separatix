"""Kernel-like circles example."""

from sklearn.datasets import make_circles

from separatix import diagnose

X, y = make_circles(n_samples=250, noise=0.08, factor=0.4, random_state=0)
print(diagnose(X, y, random_state=0))
