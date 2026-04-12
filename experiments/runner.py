from __future__ import annotations
import csv
import hashlib
import itertools
import json
import pathlib
import random
import sys
import time
from typing import Any, Iterator

import numpy as np

# Ensure src is importable
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.vrp import load_vrp, load_sol, resolve_best_known
from src.metrics import RunResult


def _resolve_instance(entry: str | dict, default_n_vehicles: int) -> tuple[str, int]:
    if isinstance(entry, str):
        return entry, default_n_vehicles
    return entry["path"], entry.get("n_vehicles", default_n_vehicles)


def expand_grid(params: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Expand any list values into a full Cartesian product of configurations."""
    keys = list(params.keys())
    values = [v if isinstance(v, list) else [v] for v in params.values()]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = json.load(f)
    # Normalise s_max: null → inf
    if "s_max" in cfg:
        cfg["s_max"] = [float("inf") if v is None else v for v in cfg["s_max"]]
    return cfg


def resolve_s_max_factors(
    sol_path: str,
    dist: np.ndarray | None,
    depot: int,
    factors: list[float | None],
) -> list[float]:
    """Convert S_max factors to absolute values using longest optimal route."""
    result: list[float] = []
    longest_route = None

    if dist is not None:
        sol_file = pathlib.Path(sol_path)
        if sol_file.exists():
            routes, _ = load_sol(str(sol_file))
            if routes:
                route_dists = []
                for route in routes:
                    d = dist[depot, route[0]]
                    for i in range(len(route) - 1):
                        d += dist[route[i], route[i + 1]]
                    d += dist[route[-1], depot]
                    route_dists.append(d)
                longest_route = max(route_dists)

    for f in factors:
        if f is None:
            result.append(float("inf"))
        elif longest_route is not None:
            result.append(longest_route * f)

    if not result:
        result.append(float("inf"))

    return result


def run_experiment(config_path: str, results_dir: str = "results") -> None:
    config = load_config(config_path)
    results_path = pathlib.Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    csv_path = results_path / "results.csv"

    rows: list[dict] = []

    instances = config["instances"]
    n_vehicles = config["n_vehicles"]
    seeds = config["seeds"]
    algorithms = config["algorithms"]

    for inst_entry in instances:
        instance_path, inst_n_vehicles = _resolve_instance(inst_entry, n_vehicles)
        instance = load_vrp(instance_path)
        best_known = resolve_best_known(instance_path, instance)

        # Resolve S_max values
        if "s_max_factors" in config:
            sol_path = pathlib.Path(instance_path).with_suffix(".sol")
            s_max_values = resolve_s_max_factors(
                str(sol_path), instance.dist, instance.depot,
                config["s_max_factors"],
            )
        else:
            s_max_values = config.get("s_max", [float("inf")])

        for algo_name, algo_params in algorithms.items():
            for param_combo in expand_grid(algo_params):
                for s_max in s_max_values:
                    for seed in seeds:
                        random.seed(seed)
                        np.random.seed(seed)

                        start = time.time()
                        solution, convergence, success_rate, dead_end_ratio = _run_solver(
                            algo_name, instance, inst_n_vehicles, s_max, param_combo
                        )
                        elapsed = time.time() - start

                        rel_error = None
                        if best_known is not None and s_max == float("inf") and solution.feasible:
                            rel_error = (solution.total_dist - best_known) / best_known

                        rr = RunResult(
                            algorithm=algo_name,
                            instance=instance.name,
                            seed=seed,
                            params=param_combo,
                            s_max=s_max,
                            total_dist=solution.total_dist,
                            elapsed_sec=elapsed,
                            feasible=solution.feasible,
                            relative_error=rel_error,
                            convergence=convergence,
                            success_rate=success_rate,
                            dead_end_ratio=dead_end_ratio,
                            routes=solution.routes,
                            instance_path=instance_path,
                            n_customers=instance.n_customers,
                        )

                        rows.append(rr.to_csv_row())

                        param_hash = hashlib.md5(json.dumps(param_combo, sort_keys=True).encode()).hexdigest()[:8]
                        s_max_tag = "inf" if s_max == float("inf") else f"{s_max:.0f}"
                        json_name = f"{instance.name}_{algo_name}_{seed}_{s_max_tag}_{param_hash}.json"
                        json_path = results_path / json_name
                        with open(json_path, "w") as f:
                            json.dump(rr.to_dict(), f, indent=2)

                        print(f"  {algo_name} | {instance.name} | seed={seed} | "
                              f"s_max={s_max:.1f} | dist={solution.total_dist:.1f} | "
                              f"feasible={solution.feasible} | t={elapsed:.2f}s")

    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults saved to {csv_path}")


def _run_solver(algo_name, instance, n_vehicles, s_max, params):
    if algo_name == "greedy":
        from src.greedy import solve_greedy
        sol = solve_greedy(instance, n_vehicles, s_max)
        return sol, [], [], []

    elif algo_name == "aco":
        from src.aco_base import ACOBase
        solver = ACOBase(**params)
        return solver.solve(instance, n_vehicles, s_max)

    elif algo_name == "eas":
        from src.eas import EAS
        solver = EAS(**params)
        return solver.solve(instance, n_vehicles, s_max)

    elif algo_name == "mmas":
        from src.mmas import MMAS
        solver = MMAS(**params)
        return solver.solve(instance, n_vehicles, s_max)

    raise ValueError(f"Unknown algorithm: {algo_name}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python experiments/runner.py <config.json> [results_dir]")
        sys.exit(1)
    results_dir = sys.argv[2] if len(sys.argv) > 2 else "results"
    run_experiment(sys.argv[1], results_dir)
