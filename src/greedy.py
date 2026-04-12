from __future__ import annotations
from src.solution import Solution, compute_total_dist
from src.vrp import VRPInstance


def solve_greedy(
    instance: VRPInstance,
    n_vehicles: int,
    s_max: float = float("inf"),
) -> Solution:
    unvisited = set(range(1, instance.n_customers + 1))
    routes: list[list[int]] = []

    for _ in range(n_vehicles):
        if not unvisited:
            break
        route: list[int] = []
        current = instance.depot
        load = 0.0
        route_dist = 0.0

        while unvisited:
            best_node = -1
            best_dist = float("inf")
            return_leg = instance.dist[current, instance.depot] if route else 0
            for node in unvisited:
                new_route_dist = (
                    route_dist - return_leg
                    + instance.dist[current, node]
                    + instance.dist[node, instance.depot]
                )
                new_load = load + instance.demands[node]
                if new_load <= instance.capacity and new_route_dist <= s_max:
                    d = instance.dist[current, node]
                    if d < best_dist:
                        best_dist = d
                        best_node = node

            if best_node == -1:
                break

            route_dist = (
                route_dist - return_leg
                + instance.dist[current, best_node]
                + instance.dist[best_node, instance.depot]
            )
            load += instance.demands[best_node]
            current = best_node
            route.append(best_node)
            unvisited.remove(best_node)

        if route:
            routes.append(route)

    feasible = len(unvisited) == 0
    return Solution(
        routes=routes,
        total_dist=compute_total_dist(routes, instance.dist, instance.depot),
        feasible=feasible,
    )
