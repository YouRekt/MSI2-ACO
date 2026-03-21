import numpy as np
import pytest
from src.aco_base import ACOBase
from src.eas import EAS
from src.vrp import load_vrp


def test_eas_returns_solution(tiny_vrp_file):
    inst = load_vrp(tiny_vrp_file)
    solver = EAS(n_ants=5, n_iterations=3, alpha=1.0, beta=2.0, rho=0.5, tau_init=1.0, e=5)
    solution, convergence, _, _ = solver.solve(inst, n_vehicles=2)
    assert len(convergence) == 3
    assert solution.total_dist > 0


def test_eas_higher_pheromone_on_best_edges(tiny_vrp_file):
    import random
    inst = load_vrp(tiny_vrp_file)

    random.seed(1); np.random.seed(1)
    aco = ACOBase(n_ants=10, n_iterations=5, alpha=1.0, beta=2.0, rho=0.5, tau_init=1.0)
    aco.solve(inst, n_vehicles=2)

    random.seed(1); np.random.seed(1)
    eas = EAS(n_ants=10, n_iterations=5, alpha=1.0, beta=2.0, rho=0.5, tau_init=1.0, e=10)
    eas.solve(inst, n_vehicles=2)

    assert eas.pheromone.sum() > aco.pheromone.sum()
