"""Nonlinear moons example."""

from sklearn.datasets import make_moons

from separatix import diagnose

X, y = make_moons(n_samples=250, noise=0.2, random_state=0)
print(diagnose(X, y, random_state=0))
