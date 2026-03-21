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
    coords: np.ndarray      # shape (n+1, 2)
    demands: np.ndarray     # shape (n+1,)
    dist: np.ndarray        # shape (n+1, n+1), Euclidean distances
    best_known: float | None
    candidates: np.ndarray  # shape (n+1, k), indices of k nearest neighbours


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
                m = re.search(r"Best known[^:]*:\s*([\d.]+)", val, re.IGNORECASE)
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

    # Build 0-indexed arrays (depot becomes index 0)
    node_ids = sorted(coords.keys())
    depot_idx = node_ids.index(depot_id)
    ordered = [node_ids[depot_idx]] + [n for n in node_ids if n != node_ids[depot_idx]]

    n = len(ordered)
    coord_arr = np.array([coords[i] for i in ordered], dtype=float)
    demand_arr = np.array([demands[i] for i in ordered], dtype=float)

    diff = coord_arr[:, np.newaxis, :] - coord_arr[np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))

    k = min(k_candidates, n - 1)
    candidates = np.argsort(dist, axis=1)[:, 1:k + 1]

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
