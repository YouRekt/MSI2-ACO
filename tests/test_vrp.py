import numpy as np
import pytest
from src.vrp import load_vrp, VRPInstance


def test_load_basic_fields(tiny_vrp_file):
    inst = load_vrp(tiny_vrp_file)
    assert inst.name == "tiny"
    assert inst.capacity == 50
    assert inst.n_customers == 3
    assert inst.depot == 0


def test_load_coords(tiny_vrp_file):
    inst = load_vrp(tiny_vrp_file)
    assert inst.coords.shape == (4, 2)
    np.testing.assert_array_equal(inst.coords[0], [0, 0])
    np.testing.assert_array_equal(inst.coords[1], [10, 0])


def test_load_demands(tiny_vrp_file):
    inst = load_vrp(tiny_vrp_file)
    assert inst.demands[0] == 0
    assert inst.demands[1] == 20
    assert inst.demands[2] == 20
    assert inst.demands[3] == 20


def test_distance_matrix(tiny_vrp_file):
    inst = load_vrp(tiny_vrp_file)
    assert inst.dist.shape == (4, 4)
    assert inst.dist[0, 0] == 0.0
    assert abs(inst.dist[0, 1] - 10.0) < 1e-6
    assert abs(inst.dist[1, 2] - 10.0) < 1e-6
    assert abs(inst.dist[0, 2] - np.sqrt(200)) < 1e-6


def test_best_known_parsed(tiny_vrp_file):
    inst = load_vrp(tiny_vrp_file)
    assert inst.best_known == 44.0


def test_candidates_shape(tiny_vrp_file):
    inst = load_vrp(tiny_vrp_file, k_candidates=2)
    assert inst.candidates.shape == (4, 2)


def test_candidates_are_nearest(tiny_vrp_file):
    inst = load_vrp(tiny_vrp_file, k_candidates=2)
    assert set(inst.candidates[1]).issubset({0, 2, 3})
