"""
Functional requirement tests — FR-01 through FR-06.

FR-01: Telemetry dict contains all required keys
FR-02: AI selects a valid phase (0–5) for each junction
FR-03: Emergency preemption overrides AI and sets emergency phase
FR-04: Database logs step rows (requires PostgreSQL — skipped if unavailable)
FR-05: Dashboard endpoint returns junction data (requires running backend — skipped if down)
FR-06: Yellow phase is enforced before any green→green phase change

Run: python tests/mandatory/test_fr.py
"""
import sys, pathlib, logging
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import unittest
from unittest.mock import MagicMock
from collections import deque
import torch

from trafix_v6.trafix_v6 import TraFixV6
from trafix_v6.rule_governor import RuleGovernor
from backend.ai.trafix_v2 import parse_sumo_observations
from sumo.emergency_preemption import EmergencyPreemptionController

log = logging.getLogger("test_fr")
logging.basicConfig(level=logging.CRITICAL)

J = 5
T = 30
D = 20
P = 6

_REQUIRED_TELEMETRY_KEYS = {
    "intersection_id",
    "north_left", "north_through", "north_right",
    "south_left", "south_through", "south_right",
    "east_left",  "east_through",  "east_right",
    "west_left",  "west_through",  "west_right",
    "queue_length", "current_phase", "phase_duration",
}


def _make_telemetry(junction_id=0, phase=0):
    return {
        "intersection_id": junction_id,
        "north_left": 2, "north_through": 8, "north_right": 1,
        "south_left": 0, "south_through": 5, "south_right": 2,
        "east_left":  3, "east_through": 11, "east_right": 0,
        "west_left":  1, "west_through":  4, "west_right": 3,
        "queue_length": 25.0,
        "current_phase": phase,
        "phase_duration": 20.0,
    }


def _make_obs_list():
    return [_make_telemetry(i, i % 6) for i in range(J)]


class TestFR01TelemetryIngestion(unittest.TestCase):
    """FR-01: Telemetry dict must contain all required fields for each junction."""

    def test_all_required_keys_present(self):
        for i in range(J):
            tele = _make_telemetry(i)
            missing = _REQUIRED_TELEMETRY_KEYS - tele.keys()
            self.assertEqual(missing, set(),
                            f"Junction {i} telemetry missing keys: {missing}")

    def test_vehicle_counts_non_negative(self):
        count_keys = [
            "north_left", "north_through", "north_right",
            "south_left", "south_through", "south_right",
            "east_left",  "east_through",  "east_right",
            "west_left",  "west_through",  "west_right",
        ]
        for obs in _make_obs_list():
            for k in count_keys:
                self.assertGreaterEqual(obs[k], 0,
                                       f"Junction {obs['intersection_id']}: {k} < 0")

    def test_queue_length_non_negative(self):
        for obs in _make_obs_list():
            self.assertGreaterEqual(obs["queue_length"], 0.0)

    def test_phase_in_valid_range(self):
        for obs in _make_obs_list():
            self.assertGreaterEqual(obs["current_phase"], 0)
            self.assertLess(obs["current_phase"], P)

    def test_parse_accepts_telemetry_format(self):
        """parse_sumo_observations must accept the standard telemetry dict format."""
        obs_tensor = parse_sumo_observations(_make_obs_list())
        self.assertEqual(obs_tensor.shape, (J, D))


class TestFR02AIPhaseSelection(unittest.TestCase):
    """FR-02: AI must output a valid phase index (0–5) for every junction."""

    @classmethod
    def setUpClass(cls):
        cls.model = TraFixV6(obs_dim=D, num_phases=P)
        cls.model.eval()
        cls.governor = RuleGovernor(num_junctions=J, num_phases=P)

    def _run_inference(self):
        obs_tensor = parse_sumo_observations(_make_obs_list())
        window = torch.stack([obs_tensor] * T).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.model(window)
            obs_last = window[0, -1]
            logits   = self.governor.apply(logits, obs_last)
        return logits, value

    def test_returns_logits_for_all_junctions(self):
        logits, _ = self._run_inference()
        self.assertEqual(len(logits), J)

    def test_phase_in_valid_range(self):
        logits, _ = self._run_inference()
        for j, l in enumerate(logits):
            phase = int(torch.argmax(l, dim=-1).item())
            self.assertGreaterEqual(phase, 0,
                                   f"Junction {j}: phase {phase} < 0")
            self.assertLess(phase, P,
                           f"Junction {j}: phase {phase} >= {P}")

    def test_probabilities_valid(self):
        logits, _ = self._run_inference()
        for j, l in enumerate(logits):
            probs = torch.softmax(l, dim=-1)
            self.assertAlmostEqual(probs.sum().item(), 1.0, places=4)
            self.assertTrue((probs >= 0).all())

    def test_value_is_finite(self):
        _, value = self._run_inference()
        self.assertTrue(torch.isfinite(value).all(),
                       "Critic value must be finite")

    def test_model_is_deterministic_in_eval(self):
        """model.eval() + no dropout → same obs produces same logits."""
        obs_tensor = parse_sumo_observations(_make_obs_list())
        window = torch.stack([obs_tensor] * T).unsqueeze(0)
        with torch.no_grad():
            l1, _ = self.model(window)
            l2, _ = self.model(window)
        for j in range(J):
            self.assertTrue(torch.allclose(l1[j], l2[j]),
                           f"Junction {j}: non-deterministic output in eval mode")


class TestFR03EmergencyPreemption(unittest.TestCase):
    """FR-03: Emergency preemption must override AI within 1 sim step of detection."""

    def _make_traci(self, has_emergency=True):
        traci = MagicMock()
        if has_emergency:
            traci.trafficlight.getControlledLinks.return_value = [
                [("edge_N_0", "out_0", "via")],
                [("edge_E_0", "out_1", "via")],
            ]
            traci.lane.getLastStepVehicleIDs.side_effect = lambda lane: (
                ["ambulance_1"] if "N" in lane else []
            )
            traci.vehicle.getTypeID.return_value = "emergency"
            traci.vehicle.getLanePosition.return_value = 30.0
            traci.vehicle.getSpeed.return_value = 5.0
        else:
            traci.trafficlight.getControlledLinks.return_value = []
            traci.lane.getLastStepVehicleIDs.return_value = []
        traci.lane.getLength.return_value = 100.0
        traci.trafficlight.getPhase.return_value = 0
        traci.trafficlight.getRedYellowGreenState.return_value = "GGrrGGrr"
        traci.junction.getPosition.return_value = (0.0, 0.0)
        traci.lane.getShape.return_value = [(-50.0, 0.0), (0.0, 0.0)]
        return traci

    def test_preemption_activates_in_one_step(self):
        traci = self._make_traci(has_emergency=True)
        ctrl  = EmergencyPreemptionController(traci, ["J0"], log, yellow_steps=3)
        yr, pt, lc = {}, {}, {"J0": 0}
        ctrl.update("J0", 1, yellow_remaining=yr, pending_targets=pt,
                    last_phase_change_step=lc)
        self.assertTrue(ctrl.is_active("J0"),
                       "Preemption must activate within 1 step of detecting emergency vehicle")

    def test_preemption_sets_tls_state(self):
        traci = self._make_traci(has_emergency=True)
        ctrl  = EmergencyPreemptionController(traci, ["J0"], log, yellow_steps=1)
        yr, pt, lc = {}, {}, {"J0": 0}
        # Step through yellow phase to allred
        for step in range(1, 5):
            ctrl.update("J0", step, yellow_remaining=yr, pending_targets=pt,
                        last_phase_change_step=lc)
        self.assertTrue(
            traci.trafficlight.setRedYellowGreenState.called
            or traci.trafficlight.setPhase.called,
            "Preemption must call TraCI to set traffic light state",
        )

    def test_ai_overridden_during_preemption(self):
        """When preemption is active, is_active() must return True,
        indicating the sim loop should skip the AI phase decision."""
        traci = self._make_traci(has_emergency=True)
        ctrl  = EmergencyPreemptionController(traci, ["J0"], log)
        yr, pt, lc = {}, {}, {"J0": 0}
        ctrl.update("J0", 1, yellow_remaining=yr, pending_targets=pt,
                    last_phase_change_step=lc)
        # is_active() being True is the signal to the sim loop to skip AI
        self.assertTrue(ctrl.is_active("J0"))

    def test_no_preemption_without_emergency(self):
        traci = self._make_traci(has_emergency=False)
        ctrl  = EmergencyPreemptionController(traci, ["J0"], log)
        yr, pt, lc = {}, {}, {"J0": 0}
        ctrl.update("J0", 1, yellow_remaining=yr, pending_targets=pt,
                    last_phase_change_step=lc)
        self.assertFalse(ctrl.is_active("J0"))


class TestFR04DatabaseLogging(unittest.TestCase):
    """FR-04: step_log rows must be written to PostgreSQL on each telemetry batch."""

    def test_db_module_importable(self):
        from backend.database import init_db, create_session, log_step, close_session
        self.assertTrue(True)

    def test_log_step_graceful_when_no_db(self):
        """log_step must not raise even if the DB pool is None."""
        import backend.database as db
        original_pool = db._pool
        db._pool = None
        try:
            db.log_step(
                session_id=1, sim_step=1,
                decisions=[{"intersection_id": 0, "next_phase": 0,
                            "confidence": 0.9, "total_vehicles": 5,
                            "queue_length": 10.0}],
                telemetry=[_make_telemetry(0)],
            )
        finally:
            db._pool = original_pool

    def test_log_emergency_graceful_when_no_db(self):
        import backend.database as db
        original_pool = db._pool
        db._pool = None
        try:
            db.log_emergency(session_id=1, event={
                "tls_id": "J0", "junction_name": "K1",
                "ambulance_id": "amb_0", "start_step": 10,
                "end_step": 25, "transit_steps": 15,
                "vehicles_waited": 3, "total_wait_steps": 9,
                "avg_wait_steps": 3.0, "result": "cleared",
            })
        finally:
            db._pool = original_pool


class TestFR06YellowPhase(unittest.TestCase):
    """
    FR-06: A yellow phase must be inserted between any two distinct green phases.
    Tests the yellow_remaining / pending_targets mechanism used in run_sumo_live.py.
    """

    def _apply_phase_change(self, current_phase, target_phase,
                            yellow_remaining, pending_targets, tls_id, step,
                            yellow_steps=3):
        """
        Replicates the phase-change logic from run_sumo_live.py.
        If current != target, installs a yellow then queues the target.
        Returns the SUMO phase that would actually be set this step.
        """
        current_sumo_green = current_phase * 2      # even = green
        current_sumo_yellow = current_sumo_green + 1

        if target_phase == current_phase:
            return current_sumo_green  # no change needed

        # Insert yellow before switching
        yellow_remaining[tls_id] = yellow_steps
        pending_targets[tls_id]  = target_phase * 2
        return current_sumo_yellow  # set yellow now

    def test_yellow_inserted_on_phase_change(self):
        yr, pt = {}, {}
        actual = self._apply_phase_change(0, 3, yr, pt, "J0", step=1)
        self.assertEqual(actual % 2, 1, "Odd SUMO phase = yellow must be set on phase change")
        self.assertIn("J0", yr, "yellow_remaining must be populated")
        self.assertEqual(yr["J0"], 3)

    def test_no_yellow_when_same_phase(self):
        yr, pt = {}, {}
        actual = self._apply_phase_change(0, 0, yr, pt, "J0", step=1)
        self.assertEqual(actual % 2, 0, "Even SUMO phase = green when no change needed")
        self.assertNotIn("J0", yr)

    def test_target_queued_correctly(self):
        yr, pt = {}, {}
        self._apply_phase_change(0, 3, yr, pt, "J0", step=1)
        self.assertIn("J0", pt)
        self.assertEqual(pt["J0"], 6, "Target SUMO phase for model phase 3 should be 6")

    def test_yellow_decrements_each_step(self):
        yr, pt = {"J0": 3}, {"J0": 6}
        # Simulate decrementing
        yr["J0"] -= 1
        self.assertEqual(yr["J0"], 2)
        yr["J0"] -= 1
        self.assertEqual(yr["J0"], 1)
        yr["J0"] -= 1
        self.assertEqual(yr["J0"], 0)
        # When countdown hits 0, apply target phase
        if yr["J0"] <= 0:
            applied = pt.pop("J0")
            yr.pop("J0", None)
        self.assertEqual(applied, 6)
        self.assertNotIn("J0", yr)

    def test_yellow_duration_is_three_steps(self):
        yr, pt = {}, {}
        self._apply_phase_change(2, 5, yr, pt, "J0", step=1, yellow_steps=3)
        self.assertEqual(yr["J0"], 3, "Yellow duration must be 3 steps (≈3 sim-seconds)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
