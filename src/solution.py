from dataclasses import dataclass

import numpy as np


def compute_total_dist(routes: list[list[int]], dist: np.ndarray, depot: int = 0) -> float:
    total = 0.0
    for route in routes:
        total += dist[depot, route[0]]
        for i in range(len(route) - 1):
            total += dist[route[i], route[i + 1]]
        total += dist[route[-1], depot]
    return total


@dataclass
class Solution:
    routes: list[list[int]]  # customer indices; depot not included; implicitly depot→...→depot
    total_dist: float
    feasible: bool
