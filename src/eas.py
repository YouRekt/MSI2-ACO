from __future__ import annotations
from src.aco_base import ACOBase
from src.solution import Solution


class EAS(ACOBase):
    def __init__(self, e: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.e = e

    def _update_pheromones(
        self,
        solutions: list[Solution],
        best_ever: Solution | None,
    ) -> None:
        super()._update_pheromones(solutions, best_ever)

        if best_ever is None or not best_ever.feasible:
            return
        elite_deposit = self.e / best_ever.total_dist
        for route in best_ever.routes:
            nodes = [0] + route + [0]
            for i in range(len(nodes) - 1):
                self.pheromone[nodes[i], nodes[i+1]] += elite_deposit
                self.pheromone[nodes[i+1], nodes[i]] += elite_deposit
