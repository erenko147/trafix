"""
Unit tests — EmergencyPreemptionController state machine.
Uses a mock TraCI so no SUMO instance is needed.
Run: python tests/mandatory/test_unit_preemption.py
"""
import sys, pathlib, logging
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import unittest
from unittest.mock import MagicMock, patch

# Import the class under test
from sumo.emergency_preemption import EmergencyPreemptionController

log = logging.getLogger("test_preemption")
logging.basicConfig(level=logging.CRITICAL)   # suppress test noise

TLS_IDS = ["J0", "J1", "J2", "J3", "J4"]
YELLOW_STEPS = 3
ALL_RED_STEPS = 2


def _make_traci(has_emergency=False, emergency_lane="edge_N_0", lane_len=100.0):
    """
    Build a minimal mock traci module.
    If has_emergency=True, one vehicle of type 'emergency' is on the given lane.
    """
    traci = MagicMock()

    # getControlledLinks returns a list of link tuples per TLS signal
    # Each link: [(from_lane, to_lane, via), ...]
    links = [[(f"{emergency_lane}", "out_0", "via_0")]]
    traci.trafficlight.getControlledLinks.return_value = links

    if has_emergency:
        traci.lane.getLastStepVehicleIDs.return_value = ["ambulance_0"]
        traci.vehicle.getTypeID.return_value = "emergency"
        traci.vehicle.getLanePosition.return_value = 50.0   # 50 m along lane
        traci.vehicle.getSpeed.return_value = 8.0           # m/s, not stuck
    else:
        traci.lane.getLastStepVehicleIDs.return_value = []
        traci.vehicle.getTypeID.return_value = "passenger"

    traci.lane.getLength.return_value = lane_len
    traci.trafficlight.getPhase.return_value = 0
    traci.trafficlight.getRedYellowGreenState.return_value = "GGrrGGrr"
    traci.junction.getPosition.return_value = (0.0, 0.0)
    traci.lane.getShape.return_value = [(-50.0, 0.0), (0.0, 0.0)]

    return traci


class TestInitialState(unittest.TestCase):

    def setUp(self):
        self.traci = _make_traci(has_emergency=False)
        self.ctrl = EmergencyPreemptionController(
            self.traci, TLS_IDS, log,
            yellow_steps=YELLOW_STEPS, all_red_steps=ALL_RED_STEPS,
        )

    def test_initially_inactive(self):
        for tls in TLS_IDS:
            self.assertFalse(self.ctrl.is_active(tls),
                            f"Controller should be inactive at start for {tls}")

    def test_pop_completed_sessions_empty_at_start(self):
        completed = self.ctrl.pop_completed_sessions()
        self.assertEqual(completed, [])

    def test_update_no_emergency_stays_inactive(self):
        yellow_rem = {}
        pending    = {}
        last_chg   = {t: 0 for t in TLS_IDS}
        self.ctrl.update("J0", step=1,
                         yellow_remaining=yellow_rem,
                         pending_targets=pending,
                         last_phase_change_step=last_chg)
        self.assertFalse(self.ctrl.is_active("J0"))


class TestPreemptionActivation(unittest.TestCase):

    def setUp(self):
        self.traci = _make_traci(has_emergency=True, emergency_lane="edge_N")
        self.ctrl = EmergencyPreemptionController(
            self.traci, TLS_IDS, log,
            yellow_steps=YELLOW_STEPS, all_red_steps=ALL_RED_STEPS,
        )

    def test_activates_on_emergency(self):
        yellow_rem = {}
        pending    = {}
        last_chg   = {t: 0 for t in TLS_IDS}
        self.ctrl.update("J0", step=1,
                         yellow_remaining=yellow_rem,
                         pending_targets=pending,
                         last_phase_change_step=last_chg)
        self.assertTrue(self.ctrl.is_active("J0"),
                       "Controller should be active after detecting emergency vehicle")

    def test_stage_is_yellow_after_activation(self):
        yellow_rem = {}
        pending    = {}
        last_chg   = {t: 0 for t in TLS_IDS}
        self.ctrl.update("J0", step=1,
                         yellow_remaining=yellow_rem,
                         pending_targets=pending,
                         last_phase_change_step=last_chg)
        self.assertEqual(self.ctrl.stage["J0"], "yellow")

    def test_cancels_pending_yellow(self):
        yellow_rem = {"J0": 2}
        pending    = {"J0": 3}
        last_chg   = {t: 0 for t in TLS_IDS}
        self.ctrl.update("J0", step=1,
                         yellow_remaining=yellow_rem,
                         pending_targets=pending,
                         last_phase_change_step=last_chg)
        self.assertNotIn("J0", yellow_rem,
                        "Pending yellow should be cancelled when preemption starts")

    def test_other_junctions_not_affected(self):
        yellow_rem = {}
        pending    = {}
        last_chg   = {t: 0 for t in TLS_IDS}
        self.ctrl.update("J0", step=1,
                         yellow_remaining=yellow_rem,
                         pending_targets=pending,
                         last_phase_change_step=last_chg)
        for tls in TLS_IDS[1:]:
            self.assertFalse(self.ctrl.is_active(tls),
                            f"Preemption of J0 should not affect {tls}")


class TestStateTransitions(unittest.TestCase):
    """Step through yellow→allred→green manually."""

    def setUp(self):
        self.traci = _make_traci(has_emergency=True, emergency_lane="edge_N")
        self.ctrl = EmergencyPreemptionController(
            self.traci, TLS_IDS, log,
            yellow_steps=YELLOW_STEPS,
            all_red_steps=ALL_RED_STEPS,
            max_emergency_green=60,
        )
        self._yellow_rem = {}
        self._pending    = {}
        self._last_chg   = {t: 0 for t in TLS_IDS}

    def _step(self, step):
        return self.ctrl.update(
            "J0", step=step,
            yellow_remaining=self._yellow_rem,
            pending_targets=self._pending,
            last_phase_change_step=self._last_chg,
        )

    def test_yellow_to_allred(self):
        self._step(1)                     # activates → yellow
        self.assertEqual(self.ctrl.stage["J0"], "yellow")
        for s in range(2, 2 + YELLOW_STEPS):
            self._step(s)
        self.assertEqual(self.ctrl.stage["J0"], "allred",
                        "Should transition from yellow to allred after YELLOW_STEPS")

    def test_allred_to_green(self):
        # Advance through yellow
        for s in range(1, 1 + YELLOW_STEPS + 1):
            self._step(s)
        self.assertEqual(self.ctrl.stage["J0"], "allred")
        for s in range(1 + YELLOW_STEPS + 1, 1 + YELLOW_STEPS + 1 + ALL_RED_STEPS):
            self._step(s)
        self.assertEqual(self.ctrl.stage["J0"], "green",
                        "Should transition from allred to green after ALL_RED_STEPS")


class TestMetricsCollection(unittest.TestCase):

    def setUp(self):
        # Set up emergency present then gone after green phase
        self.traci_active = _make_traci(has_emergency=True, emergency_lane="edge_N")
        self.traci_clear  = _make_traci(has_emergency=False)

    def test_session_opened_on_activation(self):
        ctrl = EmergencyPreemptionController(
            self.traci_active, TLS_IDS, log,
            yellow_steps=1, all_red_steps=1,
        )
        yr, pt, lc = {}, {}, {t: 0 for t in TLS_IDS}
        ctrl.update("J0", 1, yellow_remaining=yr, pending_targets=pt, last_phase_change_step=lc)
        self.assertIn("J0", ctrl._session, "Session should open on preemption start")

    def test_completed_event_recorded(self):
        ctrl = EmergencyPreemptionController(
            self.traci_active, TLS_IDS, log,
            yellow_steps=1, all_red_steps=1, max_emergency_green=60,
        )
        yr, pt, lc = {}, {}, {t: 0 for t in TLS_IDS}
        # Advance to green stage
        for step in range(1, 10):
            ctrl.update("J0", step, yellow_remaining=yr, pending_targets=pt,
                        last_phase_change_step=lc)
            if ctrl.stage.get("J0") == "green":
                break

        # Now swap to no-emergency traci so the controller thinks vehicle left
        ctrl.traci = self.traci_clear
        ctrl.traci.trafficlight.getControlledLinks.return_value = [
            [("edge_N_0", "out_0", "via_0")]
        ]
        ctrl.traci.lane.getLastStepVehicleIDs.return_value = []
        for step in range(10, 20):
            ctrl.update("J0", step, yellow_remaining=yr, pending_targets=pt,
                        last_phase_change_step=lc)
            completed = ctrl.pop_completed_sessions()
            if completed:
                rec = completed[0]
                self.assertIn("transit_steps", rec)
                self.assertIn("vehicles_waited", rec)
                self.assertIn("result", rec)
                return

        self.fail("No completed session recorded after emergency cleared")


class TestBuildState(unittest.TestCase):

    def setUp(self):
        traci = _make_traci(has_emergency=False)
        traci.trafficlight.getControlledLinks.return_value = [
            [("edge_N_0", "out_0", "v0")],
            [("edge_N_1", "out_1", "v1")],
            [("edge_E_0", "out_2", "v2")],
        ]
        self.ctrl = EmergencyPreemptionController(traci, TLS_IDS, log)

    def test_approach_edge_gets_green(self):
        state = self.ctrl._build_state("J0", "edge_N", "G")
        self.assertEqual(state[0], "G")
        self.assertEqual(state[1], "G")
        self.assertEqual(state[2], "r")

    def test_non_approach_edges_get_red(self):
        state = self.ctrl._build_state("J0", "edge_E", "G")
        self.assertEqual(state[0], "r")
        self.assertEqual(state[1], "r")
        self.assertEqual(state[2], "G")


if __name__ == "__main__":
    unittest.main(verbosity=2)
