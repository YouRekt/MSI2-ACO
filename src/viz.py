from __future__ import annotations
import argparse
import json
import pathlib
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd


def plot_routes(
    routes: list[list[int]],
    coords: np.ndarray,
    title: str,
    out_path: str | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = cm.tab10.colors
    for i, route in enumerate(routes):
        path = [0] + route + [0]
        xs = [coords[n, 0] for n in path]
        ys = [coords[n, 1] for n in path]
        ax.plot(xs, ys, "-o", color=colors[i % len(colors)], markersize=5, label=f"Route {i+1}")
    ax.plot(coords[0, 0], coords[0, 1], "ks", markersize=10, label="Depot")
    ax.set_title(title)
    ax.legend(fontsize=8)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        plt.close()
    return fig


def plot_convergence(
    convergence_by_algo: dict[str, list[float]],
    title: str,
    out_path: str | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    for algo, curve in convergence_by_algo.items():
        ax.plot(curve, label=algo)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best distance")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        plt.close()
    return fig


def plot_boxplots(df: pd.DataFrame, out_path: str | None = None) -> plt.Figure:
    instances = df["instance"].unique()
    fig, axes = plt.subplots(1, len(instances), figsize=(6 * len(instances), 6), squeeze=False)
    for ax, inst in zip(axes[0], instances):
        sub = df[df["instance"] == inst]
        sub.boxplot(column="total_dist", by="algorithm", ax=ax)
        ax.set_title(inst)
        ax.set_xlabel("Algorithm")
        ax.set_ylabel("Total distance")
    plt.suptitle("Solution quality distribution across seeds")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        plt.close()
    return fig


def plot_scaling(df: pd.DataFrame, out_path: str | None = None) -> plt.Figure | None:
    sub = df[df["relative_error"].notna() & df["feasible"]]
    if sub.empty:
        print("No feasible runs with known best solutions found for scaling plot.")
        return None
    if "n_customers" not in sub.columns:
        print("'n_customers' column missing from CSV — skipping scaling plot.")
        return None
    grouped = sub.groupby(["algorithm", "n_customers"])["relative_error"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    for algo in grouped["algorithm"].unique():
        g = grouped[grouped["algorithm"] == algo].sort_values("n_customers")
        ax.plot(g["n_customers"], g["relative_error"], "-o", label=algo)
    ax.set_xlabel("Number of customers")
    ax.set_ylabel("Mean relative error vs best known")
    ax.set_title("Scalability: solution quality vs instance size")
    ax.legend()
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        plt.close()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Generate experiment visualisations")
    parser.add_argument("csv_path", help="Path to results/results.csv")
    parser.add_argument("--results-dir", default="results", help="Root results directory")
    args = parser.parse_args()

    results_dir = pathlib.Path(args.results_dir)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv_path)
    df["feasible"] = df["feasible"].astype(bool)
    df["relative_error"] = pd.to_numeric(df["relative_error"], errors="coerce")

    plot_boxplots(df, str(figures_dir / "boxplots.png"))
    print("Saved boxplots.png")

    plot_scaling(df, str(figures_dir / "scaling.png"))
    print("Saved scaling.png")

    json_files = list(results_dir.glob("*.json"))
    convergence_by_instance: dict[str, dict[str, list[float]]] = defaultdict(dict)
    for jf in json_files:
        with open(jf) as f:
            run = json.load(f)
        if run.get("convergence"):
            inst = run["instance"]
            algo = run["algorithm"]
            if algo not in convergence_by_instance[inst]:
                convergence_by_instance[inst][algo] = run["convergence"]

    for inst, curves in convergence_by_instance.items():
        out = str(figures_dir / f"convergence_{inst}.png")
        plot_convergence(curves, f"Convergence — {inst}", out)
        print(f"Saved convergence_{inst}.png")

    best_runs: dict[tuple, dict] = {}
    for jf in json_files:
        with open(jf) as f:
            run = json.load(f)
        if not run.get("feasible") or not run.get("routes"):
            continue
        key = (run["instance"], run["algorithm"])
        if key not in best_runs or run["total_dist"] < best_runs[key]["total_dist"]:
            best_runs[key] = run

    for (inst, algo), run in best_runs.items():
        if "instance_path" not in run:
            continue
        from src.vrp import load_vrp
        vrp_inst = load_vrp(run["instance_path"])
        out = str(figures_dir / f"routes_{inst}_{algo}.png")
        plot_routes(run["routes"], vrp_inst.coords, f"Routes — {algo} — {inst}", out)
        print(f"Saved routes_{inst}_{algo}.png")


if __name__ == "__main__":
    main()
