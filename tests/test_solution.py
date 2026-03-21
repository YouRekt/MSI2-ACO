# tests/test_solution.py
from src.solution import Solution


def test_solution_fields():
    sol = Solution(routes=[[1, 2], [3]], total_dist=42.0, feasible=True)
    assert sol.routes == [[1, 2], [3]]
    assert sol.total_dist == 42.0
    assert sol.feasible is True


def test_solution_infeasible():
    sol = Solution(routes=[], total_dist=float("inf"), feasible=False)
    assert not sol.feasible
