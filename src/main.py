import argparse
import json
import random
import time

import numpy as np

from src.vrp import load_vrp
from src.metrics import RunResult


def main():
    parser = argparse.ArgumentParser(description="Ant Colony Optimization for DCVRP")
    parser.add_argument("instance_path", type=str, help="Path to the .vrp instance file")
    parser.add_argument("--algo", type=str, default="aco",
                        choices=["greedy", "aco", "eas", "mmas"],
                        help="Solver to use")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-vehicles", type=int, required=True, help="Number of vehicles")
    parser.add_argument("--smax", type=float, default=float("inf"))

    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--n-ants", type=int, default=20)
    parser.add_argument("--n-iterations", type=int, default=200)
    parser.add_argument("--tau-init", type=float, default=1.0)
    parser.add_argument("--k-candidates", type=int, default=20)

    parser.add_argument("--e", type=int, default=None,
                        help="EAS: number of elite ants (default: n_ants)")

    parser.add_argument("--tau-min", type=float, default=0.01)
    parser.add_argument("--tau-max", type=float, default=1.0)

    parser.add_argument("--output-json", type=str, default=None,
                        help="Write RunResult JSON to this path")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    instance = load_vrp(args.instance_path, k_candidates=args.k_candidates)

    start = time.time()

    if args.algo == "greedy":
        from src.greedy import solve_greedy
        solution = solve_greedy(instance, args.n_vehicles, args.smax)
        convergence, success_rate, dead_end_ratio = [], [], []

    elif args.algo == "aco":
        from src.aco_base import ACOBase
        solver = ACOBase(
            n_ants=args.n_ants, n_iterations=args.n_iterations,
            alpha=args.alpha, beta=args.beta, rho=args.rho,
            tau_init=args.tau_init,
        )
        solution, convergence, success_rate, dead_end_ratio = solver.solve(
            instance, args.n_vehicles, args.smax
        )

    elif args.algo == "eas":
        from src.eas import EAS
        e = args.e if args.e is not None else args.n_ants
        solver = EAS(
            n_ants=args.n_ants, n_iterations=args.n_iterations,
            alpha=args.alpha, beta=args.beta, rho=args.rho,
            tau_init=args.tau_init, e=e,
        )
        solution, convergence, success_rate, dead_end_ratio = solver.solve(
            instance, args.n_vehicles, args.smax
        )

    elif args.algo == "mmas":
        from src.mmas import MMAS
        solver = MMAS(
            n_ants=args.n_ants, n_iterations=args.n_iterations,
            alpha=args.alpha, beta=args.beta, rho=args.rho,
            tau_min=args.tau_min, tau_max=args.tau_max,
        )
        solution, convergence, success_rate, dead_end_ratio = solver.solve(
            instance, args.n_vehicles, args.smax
        )

    elapsed = time.time() - start

    relative_error = None
    if instance.best_known and args.smax == float("inf") and solution.feasible:
        relative_error = (solution.total_dist - instance.best_known) / instance.best_known

    print(f"Algorithm : {args.algo}")
    print(f"Instance  : {instance.name} ({instance.n_customers} customers)")
    print(f"Feasible  : {solution.feasible}")
    print(f"Total dist: {solution.total_dist:.2f}")
    if relative_error is not None:
        print(f"Rel. error: {relative_error:.2%}")
    print(f"Time (s)  : {elapsed:.3f}")
    print(f"Routes    : {len(solution.routes)}")
    for i, route in enumerate(solution.routes):
        print(f"  Route {i+1}: depot -> {' -> '.join(map(str, route))} -> depot")

    if args.output_json:
        params = {
            "alpha": args.alpha, "beta": args.beta, "rho": args.rho,
            "n_ants": args.n_ants, "n_iterations": args.n_iterations,
        }
        rr = RunResult(
            algorithm=args.algo, instance=instance.name, seed=args.seed,
            params=params, s_max=args.smax, total_dist=solution.total_dist,
            elapsed_sec=elapsed, feasible=solution.feasible,
            relative_error=relative_error, convergence=convergence,
            success_rate=success_rate, dead_end_ratio=dead_end_ratio,
            instance_path=args.instance_path,
            n_customers=instance.n_customers,
        )
        import pathlib
        pathlib.Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(rr.to_dict(), f, indent=2)


if __name__ == "__main__":
    main()
