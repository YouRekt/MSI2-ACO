from __future__ import annotations
import numpy as np
from src.aco_base import ACOBase
from src.solution import Solution


class MMAS(ACOBase):
    def __init__(self, tau_min: float = 0.01, tau_max: float = 1.0, **kwargs):
        kwargs["tau_init"] = tau_max
        super().__init__(**kwargs)
        self.tau_min = tau_min
        self.tau_max = tau_max

    def _update_pheromones(
        self,
        solutions: list[Solution],
        best_ever: Solution | None,
    ) -> None:
        iteration_best = min(
            (s for s in solutions if s.feasible),
            key=lambda s: s.total_dist,
            default=None,
        )
        if iteration_best is not None:
            self._deposit_on_routes(iteration_best.routes, 1.0 / iteration_best.total_dist)

        np.clip(self.pheromone, self.tau_min, self.tau_max, out=self.pheromone)
        np.fill_diagonal(self.pheromone, 0.0)
