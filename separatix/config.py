"""Configuration objects for separatix."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from separatix.constants import BUDGETS


@dataclass
class ProfilerConfig:
    """Configuration for the separatix diagnostic profiler."""

    budget: Literal["fast", "standard", "extended"] = "standard"
    topology: Literal["off", "auto", "graph", "persistent"] = "auto"
    densify_policy: Literal["fail", "warn_and_sample", "skip"] = "warn_and_sample"
    max_dense_mb: int = 512
    max_samples: int | None = None
    min_dense_samples: int = 200
    random_state: int | None = None
    warn_on_densify: bool = True
    n_jobs: int | None = None
    target_mode: Literal["auto", "singlelabel", "multilabel", "regression"] = "auto"
    multilabel_stratification: Literal["auto", "iterative", "heuristic"] = "auto"
    mlp_probes: bool = False
    mlp_device: Literal["cpu", "auto", "cuda", "mps"] = "cpu"
    mlp_trigger_skill_threshold: float = 0.75
    mlp_min_improvement: float = 0.02
    mlp_max_parameters: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.target_mode not in {"auto", "singlelabel", "multilabel", "regression"}:
            raise ValueError(f"Unsupported target mode: {self.target_mode!r}")
        if self.multilabel_stratification not in {"auto", "iterative", "heuristic"}:
            raise ValueError(
                "Unsupported multilabel stratification mode: "
                f"{self.multilabel_stratification!r}"
            )
        if self.budget not in BUDGETS:
            raise ValueError(f"Unsupported budget: {self.budget!r}")
        if self.topology not in {"off", "auto", "graph", "persistent"}:
            raise ValueError(f"Unsupported topology mode: {self.topology!r}")
        if self.densify_policy not in {"fail", "warn_and_sample", "skip"}:
            raise ValueError(f"Unsupported densify policy: {self.densify_policy!r}")
        if self.max_dense_mb <= 0:
            raise ValueError("max_dense_mb must be positive.")
        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError("max_samples must be positive when provided.")
        if self.min_dense_samples <= 0:
            raise ValueError("min_dense_samples must be positive.")
        if self.mlp_device not in {"cpu", "auto", "cuda", "mps"}:
            raise ValueError(f"Unsupported mlp device: {self.mlp_device!r}")
        if not 0.0 <= self.mlp_trigger_skill_threshold <= 1.0:
            raise ValueError("mlp_trigger_skill_threshold must lie in [0, 1].")
        if not 0.0 <= self.mlp_min_improvement <= 1.0:
            raise ValueError("mlp_min_improvement must lie in [0, 1].")
        if self.mlp_max_parameters is not None and self.mlp_max_parameters <= 0:
            raise ValueError("mlp_max_parameters must be positive when provided.")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable configuration dictionary."""
        return asdict(self)
