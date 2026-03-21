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
        feasible = [s for s in solutions if s.feasible]
        if feasible:
            iteration_best = min(feasible, key=lambda s: s.total_dist)
            deposit = 1.0 / iteration_best.total_dist
            for route in iteration_best.routes:
                nodes = [0] + route + [0]
                for i in range(len(nodes) - 1):
                    self.pheromone[nodes[i], nodes[i+1]] += deposit
                    self.pheromone[nodes[i+1], nodes[i]] += deposit

        # Always clamp — enforces [tau_min, tau_max] even after infeasible iterations
        np.clip(self.pheromone, self.tau_min, self.tau_max, out=self.pheromone)
        np.fill_diagonal(self.pheromone, 0.0)
