import json
import csv
from pathlib import Path
from experiments.runner import expand_grid, load_config


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
