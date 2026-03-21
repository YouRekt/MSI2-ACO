from dataclasses import dataclass


@dataclass
class Solution:
    routes: list[list[int]]  # customer indices; depot not included; implicitly depot→...→depot
    total_dist: float
    feasible: bool
