from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict


@dataclass
class RunResult:
    algorithm: str
    instance: str
    seed: int
    params: dict
    s_max: float
    total_dist: float
    elapsed_sec: float
    feasible: bool
    relative_error: float | None
    convergence: list[float]
    success_rate: list[float]
    dead_end_ratio: list[float]
    instance_path: str = ""   # path to .vrp file; used by viz.py to reload coords
    n_customers: int = 0      # stored for scaling plot (hypothesis 3.2)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["s_max"] = self.s_max if self.s_max != float("inf") else None
        return d

    def to_csv_row(self) -> dict:
        """Flat row for CSV (excludes list fields)."""
        return {
            "algorithm": self.algorithm,
            "instance": self.instance,
            "instance_path": self.instance_path,
            "n_customers": self.n_customers,
            "seed": self.seed,
            "s_max": self.s_max if self.s_max != float("inf") else "",
            "total_dist": self.total_dist,
            "elapsed_sec": self.elapsed_sec,
            "feasible": self.feasible,
            "relative_error": self.relative_error if self.relative_error is not None else "",
            **{f"param_{k}": v for k, v in self.params.items()},
        }


def iterations_to_delta(convergence: list[float], delta: float = 0.20) -> int | None:
    """Return the first iteration index where improvement from start exceeds delta."""
    if not convergence:
        return None
    start = convergence[0]
    for i, val in enumerate(convergence):
        if val > 0 and (start - val) / val >= delta:
            return i
    return None


def stagnation_length(convergence: list[float]) -> int:
    """Return number of trailing iterations with no improvement."""
    if not convergence:
        return 0
    best = convergence[0]
    last_improvement = 0
    for i, val in enumerate(convergence):
        if val < best:
            best = val
            last_improvement = i
    return len(convergence) - 1 - last_improvement
