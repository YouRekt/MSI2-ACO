from src.greedy import solve_greedy
from src.vrp import load_vrp


def test_greedy_feasible(tiny_vrp_file):
    inst = load_vrp(tiny_vrp_file)
    sol = solve_greedy(inst, n_vehicles=2)
    assert sol.feasible
    assert len(sol.routes) == 2
    visited = sorted(c for route in sol.routes for c in route)
    assert visited == [1, 2, 3]


def test_greedy_total_dist_positive(tiny_vrp_file):
    inst = load_vrp(tiny_vrp_file)
    sol = solve_greedy(inst, n_vehicles=2)
    assert sol.total_dist > 0


def test_greedy_capacity_respected(tiny_vrp_file):
    inst = load_vrp(tiny_vrp_file)
    sol = solve_greedy(inst, n_vehicles=2)
    for route in sol.routes:
        total_demand = sum(inst.demands[c] for c in route)
        assert total_demand <= inst.capacity


def test_greedy_smax_respected(tiny_vrp_file):
    inst = load_vrp(tiny_vrp_file)
    sol = solve_greedy(inst, n_vehicles=3, s_max=100.0)
    assert sol.feasible
    for route in sol.routes:
        if not route:
            continue
        dist = inst.dist[0, route[0]]
        for i in range(len(route) - 1):
            dist += inst.dist[route[i], route[i+1]]
        dist += inst.dist[route[-1], 0]
        assert dist <= 100.0 + 1e-9


def test_greedy_infeasible_when_not_enough_vehicles(tiny_vrp_file):
    inst = load_vrp(tiny_vrp_file)
    sol = solve_greedy(inst, n_vehicles=1)
    assert not sol.feasible
