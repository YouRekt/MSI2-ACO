from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class VRPInstance:
    name: str
    n_customers: int
    capacity: int
    depot: int
    coords: np.ndarray
    demands: np.ndarray
    dist: np.ndarray
    best_known: float | None
    candidates: np.ndarray


def load_vrp(path: str, k_candidates: int = 20) -> VRPInstance:
    text = Path(path).read_text()
    lines = [l.strip() for l in text.splitlines() if l.strip() and l.strip() != "EOF"]

    name = ""
    best_known = None
    capacity = 0
    dimension = 0
    coords: dict[int, tuple[float, float]] = {}
    demands: dict[int, float] = {}
    depot_id = 1

    section = None
    for line in lines:
        if ":" in line and not line[0].isdigit():
            key, _, val = line.partition(":")
            key, val = key.strip().upper(), val.strip()
            if key == "NAME":
                name = val
            elif key == "COMMENT":
                m = re.search(r"(?:Best known|Optimal value)[^:]*:\s*([\d.]+)", val, re.IGNORECASE)
                if m:
                    best_known = float(m.group(1))
            elif key == "DIMENSION":
                dimension = int(val)
            elif key == "CAPACITY":
                capacity = int(val)
            continue

        if line.upper() in ("NODE_COORD_SECTION", "DEMAND_SECTION", "DEPOT_SECTION"):
            section = line.upper()
            continue

        parts = line.split()
        if not parts:
            continue

        node_id = int(parts[0])
        if section == "NODE_COORD_SECTION":
            coords[node_id] = (float(parts[1]), float(parts[2]))
        elif section == "DEMAND_SECTION":
            demands[node_id] = float(parts[1])
        elif section == "DEPOT_SECTION":
            if node_id != -1:
                depot_id = node_id

    node_ids = sorted(coords.keys())
    depot_idx = node_ids.index(depot_id)
    ordered = [node_ids[depot_idx]] + [n for n in node_ids if n != node_ids[depot_idx]]

    if dimension > 0:
        assert len(ordered) == dimension, f"DIMENSION={dimension} but parsed {len(ordered)} nodes"

    n = len(ordered)
    coord_arr = np.array([coords[i] for i in ordered], dtype=float)
    demand_arr = np.array([demands[i] for i in ordered], dtype=float)

    diff = coord_arr[:, np.newaxis, :] - coord_arr[np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))

    k = min(k_candidates, n - 1)
    # argpartition is O(n) per row vs O(n log n) for full argsort
    partitioned = np.argpartition(dist, range(1, k + 1), axis=1)[:, 1:k + 1]
    row_idx = np.arange(n)[:, None]
    order = np.argsort(dist[row_idx, partitioned], axis=1)
    candidates = partitioned[row_idx, order]

    return VRPInstance(
        name=name,
        n_customers=n - 1,
        capacity=capacity,
        depot=0,
        coords=coord_arr,
        demands=demand_arr,
        dist=dist,
        best_known=best_known,
        candidates=candidates,
    )


def load_sol(path: str) -> tuple[list[list[int]], float | None]:
    """Parse a .sol file. Returns (routes, cost).

    Customer IDs in .sol files are 1-based and correspond directly
    to our reindexed customer indices (depot=0, customers=1..n).
    """
    routes: list[list[int]] = []
    cost: float | None = None
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.upper().startswith("ROUTE"):
            _, _, nodes_str = line.partition(":")
            routes.append([int(x) for x in nodes_str.split()])
        elif line.lower().startswith("cost"):
            cost = float(line.split()[-1])
    return routes, cost


def resolve_best_known(vrp_path: str, instance: VRPInstance) -> float | None:
    """Return best_known from instance, falling back to .sol file cost."""
    if instance.best_known is not None:
        return instance.best_known
    sol_path = Path(vrp_path).with_suffix(".sol")
    if sol_path.exists():
        _, cost = load_sol(str(sol_path))
        return cost
    return None
