# Examples

The repository contains runnable examples for the supported target paths and
several qualitative synthetic geometries.

## Minimal classification report

```{literalinclude} ../examples/basic_breast_cancer.py
:language: python
```

## Sparse, text-like features

```{literalinclude} ../examples/sparse_text_like_embeddings.py
:language: python
```

## Multilabel data

The OpenML example downloads a public dataset at runtime, so it requires network
access when executed.

```{literalinclude} ../examples/openml_multilabel_yeast.py
:language: python
```

## More examples

- [Linear hyperplane](https://github.com/NiklasMelton/Separatix/blob/develop/examples/high_dimensional_linear_hyperplane.py)
- [Curvilinear hyperplane](https://github.com/NiklasMelton/Separatix/blob/develop/examples/high_dimensional_curvilinear_hyperplane.py)
- [Two moons](https://github.com/NiklasMelton/Separatix/blob/develop/examples/moons_vs_linear.py)
- [Kernel-like circles](https://github.com/NiklasMelton/Separatix/blob/develop/examples/circles_kernel_signal.py)
- [Multiclass wine data](https://github.com/NiklasMelton/Separatix/blob/develop/examples/multiclass_wine.py)
- [Random-label control](https://github.com/NiklasMelton/Separatix/blob/develop/examples/random_label_control.py)
- [Probe family gallery](https://github.com/NiklasMelton/Separatix/blob/develop/examples/probe_family_gallery.py)
- [Recommendation complexity ladder](https://github.com/NiklasMelton/Separatix/blob/develop/examples/recommendation_complexity_ladder.py)

The examples are diagnostics, not benchmark claims. Results can vary with
dataset construction, sample size, optional dependencies, and configured
budget.
