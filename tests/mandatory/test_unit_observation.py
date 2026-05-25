"""
Unit tests — observation parser and normalisation (parse_sumo_observations).
Run: python tests/mandatory/test_unit_observation.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import unittest
import torch
from backend.ai.trafix_v2 import parse_sumo_observations

J = 5
D = 20


def _base_obs(junction_id=0, phase=0, duration=30.0, queue=50.0):
    return {
        "intersection_id": junction_id,
        "north_left": 3, "north_through": 10, "north_right": 2,
        "south_left": 1, "south_through": 8,  "south_right": 0,
        "east_left":  4, "east_through":  12, "east_right": 3,
        "west_left":  0, "west_through":  5,  "west_right": 1,
        "queue_length": queue,
        "current_phase": phase,
        "phase_duration": duration,
    }


def _make_obs_list(phases=None, durations=None, queues=None):
    phases    = phases    or [0] * J
    durations = durations or [30.0] * J
    queues    = queues    or [50.0] * J
    return [_base_obs(i, phases[i], durations[i], queues[i]) for i in range(J)]


class TestOutputShape(unittest.TestCase):

    def test_shape_five_junctions(self):
        obs = parse_sumo_observations(_make_obs_list())
        self.assertEqual(obs.shape, (J, D))

    def test_dtype_float32(self):
        obs = parse_sumo_observations(_make_obs_list())
        self.assertEqual(obs.dtype, torch.float32)

    def test_no_nan(self):
        obs = parse_sumo_observations(_make_obs_list())
        self.assertFalse(torch.isnan(obs).any(), "NaN found in parsed observations")


class TestNormalisation(unittest.TestCase):

    def test_queue_normalised(self):
        obs = parse_sumo_observations(_make_obs_list(queues=[200.0] * J))
        # Index 12 = queue / 200 → should be 1.0
        self.assertAlmostEqual(obs[0, 12].item(), 1.0, places=5)

    def test_zero_queue(self):
        obs = parse_sumo_observations(_make_obs_list(queues=[0.0] * J))
        self.assertAlmostEqual(obs[0, 12].item(), 0.0, places=5)

    def test_duration_normalised(self):
        obs = parse_sumo_observations(_make_obs_list(durations=[120.0] * J))
        # Index 19 = duration / 120 → should be 1.0
        self.assertAlmostEqual(obs[0, 19].item(), 1.0, places=5)

    def test_duration_capped_at_3(self):
        obs = parse_sumo_observations(_make_obs_list(durations=[999.0] * J))
        self.assertAlmostEqual(obs[0, 19].item(), 3.0, places=5)

    def test_lane_counts_non_negative(self):
        obs = parse_sumo_observations(_make_obs_list())
        self.assertTrue((obs[:, :12] >= 0).all(),
                        "Lane count features must be non-negative")


class TestPhaseOneHot(unittest.TestCase):

    def test_phase_zero_one_hot(self):
        obs = parse_sumo_observations(_make_obs_list(phases=[0] * J))
        # Indices 13-18 are phase one-hot; index 13 = phase 0
        self.assertAlmostEqual(obs[0, 13].item(), 1.0, places=5)
        for p in range(1, 6):
            self.assertAlmostEqual(obs[0, 13 + p].item(), 0.0, places=5,
                                   msg=f"Phase bit {p} should be 0 for phase 0")

    def test_phase_three_one_hot(self):
        obs = parse_sumo_observations(_make_obs_list(phases=[3] * J))
        self.assertAlmostEqual(obs[0, 16].item(), 1.0, places=5)  # 13+3=16
        for p in range(6):
            if p != 3:
                self.assertAlmostEqual(obs[0, 13 + p].item(), 0.0, places=5,
                                       msg=f"Phase bit {p} should be 0 for phase 3")

    def test_each_phase_produces_unique_encoding(self):
        encodings = []
        for phase in range(6):
            obs = parse_sumo_observations(_make_obs_list(phases=[phase] * J))
            encodings.append(obs[0, 13:19].tolist())
        # All encodings should be distinct
        self.assertEqual(len(set(map(tuple, encodings))), 6)

    def test_phase_one_hot_sums_to_one(self):
        for phase in range(6):
            obs = parse_sumo_observations(_make_obs_list(phases=[phase] * J))
            total = obs[0, 13:19].sum().item()
            self.assertAlmostEqual(total, 1.0, places=5,
                                   msg=f"One-hot sum != 1 for phase {phase}")


class TestOrdering(unittest.TestCase):

    def test_sorted_by_junction_id(self):
        """Input order should not matter — output is sorted by intersection_id."""
        ordered = _make_obs_list()
        shuffled = list(reversed(ordered))
        obs_ordered  = parse_sumo_observations(ordered)
        obs_shuffled = parse_sumo_observations(shuffled)
        self.assertTrue(torch.allclose(obs_ordered, obs_shuffled),
                        "Observation order differs when input is shuffled")


class TestEdgeCases(unittest.TestCase):

    def test_zero_vehicles(self):
        zeros = [{
            "intersection_id": i,
            "north_left": 0, "north_through": 0, "north_right": 0,
            "south_left": 0, "south_through": 0, "south_right": 0,
            "east_left":  0, "east_through":  0, "east_right": 0,
            "west_left":  0, "west_through":  0, "west_right": 0,
            "queue_length": 0.0, "current_phase": 0, "phase_duration": 0.0,
        } for i in range(J)]
        obs = parse_sumo_observations(zeros)
        # All lane counts and queue should be 0; duration 0; phase 0 bit = 1
        self.assertAlmostEqual(obs[:, :13].abs().max().item(), 0.0, places=5)
        self.assertAlmostEqual(obs[0, 13].item(), 1.0, places=5)

    def test_high_counts_do_not_produce_nan(self):
        high = [dict(_base_obs(i), north_through=1000, queue_length=1000.0)
                for i in range(J)]
        obs = parse_sumo_observations(high)
        self.assertFalse(torch.isnan(obs).any())


if __name__ == "__main__":
    unittest.main(verbosity=2)
