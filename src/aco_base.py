from __future__ import annotations
import numpy as np
from src.vrp import VRPInstance
from src.solution import Solution


class ACOBase:
    def __init__(
        self,
        n_ants: int = 20,
        n_iterations: int = 200,
        alpha: float = 1.0,
        beta: float = 2.0,
        rho: float = 0.5,
        tau_init: float = 1.0,
    ):
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.tau_init = tau_init
        self.pheromone: np.ndarray | None = None

    def solve(
        self,
        instance: VRPInstance,
        n_vehicles: int,
        s_max: float = float("inf"),
    ) -> tuple[Solution, list[float], list[float], list[float]]:
        n = instance.n_customers + 1
        self.pheromone = np.full((n, n), self.tau_init, dtype=float)
        np.fill_diagonal(self.pheromone, 0.0)

        with np.errstate(divide="ignore"):
            eta = np.where(instance.dist > 0, 1.0 / instance.dist, 0.0)

        best_ever: Solution | None = None
        convergence: list[float] = []
        success_rate: list[float] = []
        dead_end_ratio: list[float] = []

        for _ in range(self.n_iterations):
            solutions: list[Solution] = []
            iter_feasible = 0
            iter_dead_ends = 0.0

            for _ in range(self.n_ants):
                sol, dead_ends = self._construct_solution(instance, n_vehicles, s_max, eta)
                solutions.append(sol)
                iter_dead_ends += dead_ends
                if sol.feasible:
                    iter_feasible += 1
                    if best_ever is None or sol.total_dist < best_ever.total_dist:
                        best_ever = sol

            self.pheromone *= (1 - self.rho)
            np.clip(self.pheromone, 0, None, out=self.pheromone)

            self._update_pheromones(solutions, best_ever)

            convergence.append(best_ever.total_dist if best_ever else float("inf"))
            success_rate.append(iter_feasible / self.n_ants)
            dead_end_ratio.append(iter_dead_ends / self.n_ants)

        if best_ever is None:
            best_ever = Solution(routes=[], total_dist=float("inf"), feasible=False)

        return best_ever, convergence, success_rate, dead_end_ratio

    def _construct_solution(
        self,
        instance: VRPInstance,
        n_vehicles: int,
        s_max: float,
        eta: np.ndarray,
    ) -> tuple[Solution, int]:
        unvisited = list(range(1, instance.n_customers + 1))
        routes: list[list[int]] = []
        dead_ends = 0

        for _ in range(n_vehicles):
            if not unvisited:
                break
            route: list[int] = []
            current = instance.depot
            load = 0.0
            route_dist = 0.0

            while unvisited:
                candidates = [
                    j for j in instance.candidates[current]
                    if j in unvisited
                    and load + instance.demands[j] <= instance.capacity
                    and route_dist + instance.dist[current, j] + instance.dist[j, instance.depot] <= s_max
                ]
                if not candidates:
                    candidates = [
                        j for j in unvisited
                        if load + instance.demands[j] <= instance.capacity
                        and route_dist + instance.dist[current, j] + instance.dist[j, instance.depot] <= s_max
                    ]
                if not candidates:
                    dead_ends += 1
                    break

                tau = self.pheromone[current, candidates]
                h = eta[current, candidates]
                scores = (tau ** self.alpha) * (h ** self.beta)
                total = scores.sum()
                if total == 0:
                    probs = np.ones(len(candidates)) / len(candidates)
                else:
                    probs = scores / total

                chosen = candidates[np.random.choice(len(candidates), p=probs)]
                route_dist += instance.dist[current, chosen]
                load += instance.demands[chosen]
                route.append(chosen)
                unvisited.remove(chosen)
                current = chosen

            if route:
                routes.append(route)

        total = 0.0
        for route in routes:
            total += instance.dist[instance.depot, route[0]]
            for i in range(len(route) - 1):
                total += instance.dist[route[i], route[i+1]]
            total += instance.dist[route[-1], instance.depot]

        feasible = len(unvisited) == 0
        return Solution(routes=routes, total_dist=total, feasible=feasible), dead_ends

    def _update_pheromones(
        self,
        solutions: list[Solution],
        best_ever: Solution | None,
    ) -> None:
        """Basic ACO: all feasible ants deposit 1/total_dist on their edges."""
        for sol in solutions:
            if not sol.feasible:
                continue
            deposit = 1.0 / sol.total_dist
            for route in sol.routes:
                nodes = [0] + route + [0]
                for i in range(len(nodes) - 1):
                    self.pheromone[nodes[i], nodes[i+1]] += deposit
                    self.pheromone[nodes[i+1], nodes[i]] += deposit
