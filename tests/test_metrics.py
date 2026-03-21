from src.metrics import RunResult, iterations_to_delta, stagnation_length


def test_run_result_to_dict():
    rr = RunResult(
        algorithm="aco", instance="tiny", seed=42, params={"alpha": 1.0},
        s_max=float("inf"), total_dist=100.0, elapsed_sec=0.5, feasible=True,
        relative_error=0.05, convergence=[120.0, 110.0, 100.0],
        success_rate=[1.0, 1.0, 1.0], dead_end_ratio=[0.0, 0.0, 0.0],
        instance_path="data/tiny.vrp", n_customers=3,
    )
    d = rr.to_dict()
    assert d["algorithm"] == "aco"
    assert d["total_dist"] == 100.0
    assert "convergence" in d


def test_csv_row_includes_n_customers():
    rr = RunResult(
        algorithm="greedy", instance="tiny", seed=0, params={},
        s_max=float("inf"), total_dist=100.0, elapsed_sec=0.1, feasible=True,
        relative_error=None, convergence=[], success_rate=[], dead_end_ratio=[],
        instance_path="data/tiny.vrp", n_customers=3,
    )
    row = rr.to_csv_row()
    assert row["n_customers"] == 3
    assert row["instance_path"] == "data/tiny.vrp"


def test_relative_error_none_when_no_best_known():
    rr = RunResult(
        algorithm="greedy", instance="tiny", seed=0, params={},
        s_max=float("inf"), total_dist=100.0, elapsed_sec=0.1, feasible=True,
        relative_error=None, convergence=[], success_rate=[], dead_end_ratio=[],
    )
    assert rr.relative_error is None


def test_iterations_to_delta():
    convergence = [125.0, 100.0, 100.0, 100.0]
    assert iterations_to_delta(convergence, delta=0.20) == 1


def test_iterations_to_delta_never_reached():
    convergence = [100.0, 99.0, 98.0]
    assert iterations_to_delta(convergence, delta=0.20) is None


def test_stagnation_length():
    convergence = [100.0, 90.0, 85.0, 85.0, 85.0]
    assert stagnation_length(convergence) == 2


def test_stagnation_length_always_improving():
    convergence = [100.0, 90.0, 80.0]
    assert stagnation_length(convergence) == 0
