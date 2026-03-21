import numpy as np
import pytest
from src.mmas import MMAS
from src.vrp import load_vrp


def test_mmas_returns_solution(tiny_vrp_file):
    inst = load_vrp(tiny_vrp_file)
    solver = MMAS(n_ants=5, n_iterations=3, alpha=1.0, beta=2.0, rho=0.2,
                  tau_min=0.01, tau_max=1.0)
    solution, convergence, _, _ = solver.solve(inst, n_vehicles=2)
    assert len(convergence) == 3
    assert solution.total_dist > 0


def test_mmas_pheromone_clamped(tiny_vrp_file):
    inst = load_vrp(tiny_vrp_file)
    tau_min, tau_max = 0.1, 2.0
    solver = MMAS(n_ants=10, n_iterations=10, alpha=1.0, beta=2.0, rho=0.2,
                  tau_min=tau_min, tau_max=tau_max)
    solver.solve(inst, n_vehicles=2)
    mask = ~np.eye(solver.pheromone.shape[0], dtype=bool)
    assert np.all(solver.pheromone[mask] >= tau_min - 1e-9)
    assert np.all(solver.pheromone[mask] <= tau_max + 1e-9)


def test_mmas_initializes_at_tau_max(tiny_vrp_file):
    inst = load_vrp(tiny_vrp_file)
    tau_max = 5.0
    solver = MMAS(n_ants=5, n_iterations=1, alpha=1.0, beta=2.0, rho=0.2,
                  tau_min=0.1, tau_max=tau_max)
    import numpy as np
    n = inst.n_customers + 1
    solver.pheromone = np.full((n, n), tau_max)
    np.fill_diagonal(solver.pheromone, 0.0)
    mask = ~np.eye(n, dtype=bool)
    assert np.all(solver.pheromone[mask] == tau_max)
