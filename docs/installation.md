# Installation

## Base package

Install the latest release from PyPI:

```bash
pip install separatix
```

The base installation includes NumPy, SciPy, and scikit-learn. It supports dense
and sparse inputs, classification, multilabel classification, and explicit
regression without any optional dependency.

## Optional features

Install only the extras needed by your workflow:

```bash
pip install "separatix[pandas]"       # pandas DataFrame and Series input
pip install "separatix[multilabel]"  # iterative multilabel stratification
pip install "separatix[tda]"         # persistent homology through ripser
pip install "separatix[mlp]"         # optional PyTorch MLP probes
```

Extras can be combined:

```bash
pip install "separatix[pandas,multilabel,tda]"
```

Optional features degrade gracefully when their dependencies are absent. For
example, multilabel diagnostics use deterministic heuristic stratification
without the `multilabel` extra, while unavailable persistent topology is
recorded as skipped.

## Development version

To install the current `develop` branch:

```bash
pip install "git+https://github.com/NiklasMelton/Separatix.git@develop"
```

## Supported Python versions

`separatix` supports Python 3.9 through 3.14.
