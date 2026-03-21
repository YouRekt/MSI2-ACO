import numpy as np
import pytest
from src.aco_base import ACOBase
from src.vrp import load_vrp


def test_aco_returns_solution(tiny_vrp_file):
    inst = load_vrp(tiny_vrp_file)
    solver = ACOBase(n_ants=5, n_iterations=3, alpha=1.0, beta=2.0, rho=0.5, tau_init=1.0)
    solution, convergence, success_rate, dead_end_ratio = solver.solve(inst, n_vehicles=2)
    assert solution.routes is not None
    assert solution.total_dist > 0


def test_aco_convergence_length(tiny_vrp_file):
    inst = load_vrp(tiny_vrp_file)
    solver = ACOBase(n_ants=5, n_iterations=10, alpha=1.0, beta=2.0, rho=0.5, tau_init=1.0)
    _, convergence, _, _ = solver.solve(inst, n_vehicles=2)
    assert len(convergence) == 10


def test_aco_success_rate_bounds(tiny_vrp_file):
    inst = load_vrp(tiny_vrp_file)
    solver = ACOBase(n_ants=5, n_iterations=5, alpha=1.0, beta=2.0, rho=0.5, tau_init=1.0)
    _, _, success_rate, dead_end_ratio = solver.solve(inst, n_vehicles=2)
    assert len(success_rate) == 5
    assert all(0.0 <= s <= 1.0 for s in success_rate)
    assert all(d >= 0.0 for d in dead_end_ratio)


def test_aco_pheromone_updated(tiny_vrp_file):
    inst = load_vrp(tiny_vrp_file)
    solver = ACOBase(n_ants=5, n_iterations=1, alpha=1.0, beta=2.0, rho=0.5, tau_init=1.0)
    solver.solve(inst, n_vehicles=2)
    assert not np.all(solver.pheromone == 1.0)


def test_aco_convergence_non_increasing(tiny_vrp_file):
    inst = load_vrp(tiny_vrp_file)
    solver = ACOBase(n_ants=10, n_iterations=20, alpha=1.0, beta=2.0, rho=0.5, tau_init=1.0)
    _, convergence, _, _ = solver.solve(inst, n_vehicles=2)
    for i in range(1, len(convergence)):
        assert convergence[i] <= convergence[i-1] + 1e-9


def test_aco_respects_seed(tiny_vrp_file):
    import random
    import numpy as np
    inst = load_vrp(tiny_vrp_file)

    random.seed(7); np.random.seed(7)
    solver1 = ACOBase(n_ants=5, n_iterations=5, alpha=1.0, beta=2.0, rho=0.5, tau_init=1.0)
    sol1, conv1, _, _ = solver1.solve(inst, n_vehicles=2)

    random.seed(7); np.random.seed(7)
    solver2 = ACOBase(n_ants=5, n_iterations=5, alpha=1.0, beta=2.0, rho=0.5, tau_init=1.0)
    sol2, conv2, _, _ = solver2.solve(inst, n_vehicles=2)

    assert sol1.total_dist == sol2.total_dist
    assert conv1 == conv2
