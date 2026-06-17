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
    target_mode: Literal["auto", "singlelabel", "multilabel"] = "auto"
    multilabel_stratification: Literal["auto", "iterative", "heuristic"] = "auto"

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.target_mode not in {"auto", "singlelabel", "multilabel"}:
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

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable configuration dictionary."""
        return asdict(self)
