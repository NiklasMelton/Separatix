# Configuration reference

The functional API and estimator API share the main diagnostic options. The
defaults are deliberately conservative and memory-aware.

## Target and validation controls

| Option | Default | Purpose |
| --- | --- | --- |
| `target_mode` | `"auto"` | Route to single-label, multilabel, or explicit regression diagnostics. |
| `groups` | `None` | Keep related rows together during sampling and held-out evaluation. |
| `multilabel_stratification` | `"auto"` | Choose iterative stratification when available or the deterministic heuristic fallback. |
| `random_state` | `None` | Control sampling, splits, and randomized probes. |

## Cost and topology controls

| Option | Default | Purpose |
| --- | --- | --- |
| `budget` | `"standard"` | Select `"fast"`, `"standard"`, or `"extended"` probe and stability effort. |
| `topology` | `"auto"` | Select `"off"`, `"auto"`, `"graph"`, or `"persistent"`. |
| `max_samples` | `None` | Apply a hard row cap while preserving supervised support when feasible. |
| `max_dense_mb` | `512` | Limit the memory estimate for dense-only operations. |
| `densify_policy` | `"warn_and_sample"` | Fail, sample before densifying, or skip dense-only diagnostics. |
| `warn_on_densify` | `True` | Emit runtime warnings in addition to report events. |

`ComplexityProfiler` also exposes `min_dense_samples` and `n_jobs` for advanced
control. The public {class}`~separatix.ProfilerConfig` object validates and
stores the complete configuration.

## Optional MLP controls

MLP probes are disabled unless `mlp_probes=True` and PyTorch is installed.

| Option | Default | Purpose |
| --- | --- | --- |
| `mlp_probes` | `False` | Enable conditional feed-forward probe evaluation. |
| `mlp_device` | `"cpu"` | Select `"cpu"`, `"auto"`, `"cuda"`, or `"mps"`. |
| `mlp_trigger_skill_threshold` | `0.75` | Run the optional probe only when simpler-family evidence meets the configured trigger. |
| `mlp_min_improvement` | `0.02` | Require a minimum held-out gain before an MLP override. |
| `mlp_max_parameters` | `None` | Cap the MLP parameter count. |

An MLP recommendation requires complete held-out evidence. Failed or infeasible
group splits never fall back to in-sample override evidence.

## Choosing a budget

- `fast` is useful for iteration and skips persistent topology in auto mode.
- `standard` is the recommended default for most audits.
- `extended` spends more work on probes and stability evidence.

Whichever budget is selected, the report records the effective configuration,
sampling decisions, skipped diagnostics, and runtime.
