# Examples

The repository contains runnable examples for the supported target paths and
several qualitative synthetic geometries.

Each example's complete source appears below, so the documentation remains a
self-contained reference. The same files can be run from the repository's
`examples/` directory.

## Basic breast-cancer report

```{literalinclude} ../examples/basic_breast_cancer.py
:language: python
```

## Linear hyperplane visualization

```{literalinclude} ../examples/linear_hyperplane_visual.py
:language: python
```

## Curvilinear boundary visualization

```{literalinclude} ../examples/curvilinear_boundary_visual.py
:language: python
```

## High-dimensional linear hyperplane

```{literalinclude} ../examples/high_dimensional_linear_hyperplane.py
:language: python
```

## High-dimensional curvilinear hyperplane

```{literalinclude} ../examples/high_dimensional_curvilinear_hyperplane.py
:language: python
```

## Two moons

```{literalinclude} ../examples/moons_vs_linear.py
:language: python
```

## Kernel-like circles

```{literalinclude} ../examples/circles_kernel_signal.py
:language: python
```

## Multiclass wine data

```{literalinclude} ../examples/multiclass_wine.py
:language: python
```

## Random-label control

```{literalinclude} ../examples/random_label_control.py
:language: python
```

## Sparse text-like embeddings

```{literalinclude} ../examples/sparse_text_like_embeddings.py
:language: python
```

## OpenML multilabel yeast

The OpenML example downloads a public dataset at runtime, so it requires network
access when executed.

```{literalinclude} ../examples/openml_multilabel_yeast.py
:language: python
```

## Probe family gallery

This longer script regenerates the probe-family image used in the project
overview.

```{literalinclude} ../examples/probe_family_gallery.py
:language: python
```

## Recommendation complexity ladder

This longer script regenerates the synthetic recommendation-ladder image used
in the project overview.

```{literalinclude} ../examples/recommendation_complexity_ladder.py
:language: python
```

The examples are diagnostics, not benchmark claims. Results can vary with
dataset construction, sample size, optional dependencies, and configured
budget.
