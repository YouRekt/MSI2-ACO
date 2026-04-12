import json

import numpy as np

from experiments.runner import expand_grid, load_config, _resolve_instance, resolve_s_max_factors


def test_expand_grid_no_lists():
    params = {"alpha": 1.0, "beta": 2.0}
    result = list(expand_grid(params))
    assert result == [{"alpha": 1.0, "beta": 2.0}]


def test_expand_grid_with_lists():
    params = {"alpha": [0.5, 1.0], "beta": [1.0, 2.0]}
    result = list(expand_grid(params))
    assert len(result) == 4
    assert {"alpha": 0.5, "beta": 1.0} in result
    assert {"alpha": 1.0, "beta": 2.0} in result


def test_expand_grid_mixed():
    params = {"alpha": [1.0, 2.0], "rho": 0.5}
    result = list(expand_grid(params))
    assert len(result) == 2
    assert all(r["rho"] == 0.5 for r in result)


def test_expand_grid_empty():
    result = list(expand_grid({}))
    assert result == [{}]


def test_load_config_s_max_null(tmp_path):
    cfg = {"instances": [], "n_vehicles": 2, "seeds": [42], "s_max": [None], "algorithms": {}}
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    config = load_config(str(p))
    assert config["s_max"] == [float("inf")]


def test_resolve_instance_string():
    path, n = _resolve_instance("data/foo.vrp", 5)
    assert path == "data/foo.vrp"
    assert n == 5


def test_resolve_instance_dict_explicit_vehicles():
    path, n = _resolve_instance({"path": "data/foo.vrp", "n_vehicles": 7}, 5)
    assert path == "data/foo.vrp"
    assert n == 7


def test_resolve_instance_dict_fallback_vehicles():
    path, n = _resolve_instance({"path": "data/foo.vrp"}, 5)
    assert path == "data/foo.vrp"
    assert n == 5


def test_resolve_s_max_factors_with_sol(tmp_path):
    sol_path = tmp_path / "test.sol"
    sol_path.write_text("Route #1: 1 2 3\nRoute #2: 4 5\nCost 100\n")

    dist = np.zeros((6, 6))
    # Route 1: depot->1->2->3->depot = 10+10+10+10 = 40
    dist[0, 1] = 10; dist[1, 0] = 10
    dist[1, 2] = 10; dist[2, 1] = 10
    dist[2, 3] = 10; dist[3, 2] = 10
    dist[3, 0] = 10; dist[0, 3] = 10
    # Route 2: depot->4->5->depot = 10+10+10 = 30
    dist[0, 4] = 10; dist[4, 0] = 10
    dist[4, 5] = 10; dist[5, 4] = 10
    dist[5, 0] = 10; dist[0, 5] = 10

    result = resolve_s_max_factors(str(sol_path), dist, 0, [None, 2.0, 1.5])
    assert result[0] == float("inf")
    assert abs(result[1] - 80.0) < 1e-6
    assert abs(result[2] - 60.0) < 1e-6


def test_resolve_s_max_factors_no_sol(tmp_path):
    result = resolve_s_max_factors(str(tmp_path / "missing.sol"), None, 0, [None, 2.0])
    assert result == [float("inf")]


def test_load_config_s_max_factors(tmp_path):
    cfg = {"instances": [], "n_vehicles": 2, "seeds": [42],
           "s_max_factors": [None, 2.0, 1.5], "algorithms": {}}
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    config = load_config(str(p))
    assert config["s_max_factors"] == [None, 2.0, 1.5]
