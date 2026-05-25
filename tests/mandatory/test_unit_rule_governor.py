"""
Unit tests — RuleGovernor (min-green, max-green, anti-flicker, pressure boost).
Run: python tests/mandatory/test_unit_rule_governor.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import unittest
import torch
from trafix_v6.rule_governor import RuleGovernor, MIN_GREEN_THROUGH, MIN_GREEN_LEFT, _NEG_INF

J = 5
P = 6
D = 20


def _make_obs(phase=0, duration=0.0, queue=0.0, through_ns=0.0, through_ew=0.0):
    """Build a [J, D] observation tensor with controlled values."""
    obs = torch.zeros(J, D)
    for j in range(J):
        obs[j, 13 + phase] = 1.0                  # phase one-hot
        obs[j, 19] = min(duration / 120.0, 3.0)   # duration
        obs[j, 12] = queue / 200.0                # queue
        obs[j, 1]  = through_ns / 30.0            # north through
        obs[j, 4]  = through_ns / 30.0            # south through
        obs[j, 7]  = through_ew / 30.0            # east through
        obs[j, 10] = through_ew / 30.0            # west through
    return obs


def _uniform_logits(batch=1):
    return [torch.zeros(batch, P) for _ in range(J)]


class TestHardMaskMinGreen(unittest.TestCase):

    def setUp(self):
        self.gov = RuleGovernor(num_junctions=J, num_phases=P, min_green_s=10.0, max_green_s=90.0)

    def test_phase_locked_before_min_green_through(self):
        """Phase 0 (NS-through) held for 5 s < 10 s min → must not switch."""
        obs = _make_obs(phase=0, duration=5.0)
        logits_in = _uniform_logits()
        logits_out = self.gov.apply(logits_in, obs)
        for j in range(J):
            l = logits_out[j][0]
            # All non-current phases should be masked to -inf
            for p in range(P):
                if p != 0:
                    self.assertLess(l[p].item(), -1e8,
                                   f"Junction {j} phase {p} not masked at t=5s")

    def test_phase_locked_before_min_green_left(self):
        """Phase 1 (N-left) held for 4 s < 8 s min → must not switch."""
        obs = _make_obs(phase=1, duration=4.0)
        logits_in = _uniform_logits()
        logits_out = self.gov.apply(logits_in, obs)
        for j in range(J):
            l = logits_out[j][0]
            for p in range(P):
                if p != 1:
                    self.assertLess(l[p].item(), -1e8,
                                   f"Junction {j} phase {p} not masked at t=4s (left)")

    def test_phase_free_after_min_green(self):
        """Phase 0 held for 15 s > 10 s → switch is allowed."""
        obs = _make_obs(phase=0, duration=15.0)
        logits_in = _uniform_logits()
        logits_out = self.gov.apply(logits_in, obs)
        for j in range(J):
            l = logits_out[j][0]
            # At least one non-current phase must be unmasked
            any_open = any(l[p].item() > -1e8 for p in range(P) if p != 0)
            self.assertTrue(any_open, f"Junction {j}: all phases masked after min-green")


class TestHardMaskMaxGreen(unittest.TestCase):

    def setUp(self):
        self.gov = RuleGovernor(num_junctions=J, num_phases=P, min_green_s=10.0, max_green_s=90.0)

    def test_current_phase_blocked_after_max_green(self):
        """Phase 0 held for 100 s > 90 s max → staying on phase 0 must be masked."""
        obs = _make_obs(phase=0, duration=100.0)
        logits_in = _uniform_logits()
        logits_out = self.gov.apply(logits_in, obs)
        for j in range(J):
            l = logits_out[j][0]
            self.assertLess(l[0].item(), -1e8,
                           f"Junction {j} current phase not blocked after max-green")

    def test_other_phases_available_after_max_green(self):
        """After max-green, at least one alternative phase must remain open."""
        obs = _make_obs(phase=0, duration=100.0)
        logits_in = _uniform_logits()
        logits_out = self.gov.apply(logits_in, obs)
        for j in range(J):
            l = logits_out[j][0]
            any_open = any(l[p].item() > -1e8 for p in range(P) if p != 0)
            self.assertTrue(any_open, f"Junction {j}: no alternative phase open after max-green")


class TestAntiFlicker(unittest.TestCase):

    def setUp(self):
        self.gov = RuleGovernor(
            num_junctions=J, num_phases=P,
            flicker_window=2, flicker_penalty=10.0
        )

    def test_reversal_penalised(self):
        """A→B→A pattern: returning to A should be penalised."""
        # Record: chose phase 3, then phase 0 → reverting to 3 is penalised
        self.gov.update_state(torch.full((J,), 3, dtype=torch.long))
        self.gov.update_state(torch.full((J,), 0, dtype=torch.long))

        obs = _make_obs(phase=0, duration=15.0)
        logits_in = _uniform_logits()
        logits_out = self.gov.apply(logits_in, obs)

        for j in range(J):
            l = logits_out[j][0]
            # Phase 3 (the reversal target) should have negative adjustment
            self.assertLess(l[3].item(), 0.0,
                           f"Junction {j}: reversal target not penalised")

    def test_no_penalty_without_reversal(self):
        """A→B→C (no reversal) should carry no flicker penalty on C."""
        self.gov.update_state(torch.full((J,), 0, dtype=torch.long))
        self.gov.update_state(torch.full((J,), 1, dtype=torch.long))

        obs = _make_obs(phase=1, duration=15.0)
        logits_in = _uniform_logits()
        logits_out = self.gov.apply(logits_in, obs)

        for j in range(J):
            l = logits_out[j][0]
            # Phase 0 (reversal target would be 0 only if we went B→C→B)
            # Here the previous two actions were 0, 1 — no reversal toward 0 yet
            # All non-masked phases should be ≥ 0 (no penalty applied to 2,3,4,5)
            for p in (2, 3, 4, 5):
                self.assertGreaterEqual(l[p].item(), 0.0,
                                       f"Junction {j} phase {p} incorrectly penalised")

    def test_reset_clears_history(self):
        """After reset(), flicker history is gone and no penalty is applied."""
        self.gov.update_state(torch.full((J,), 3, dtype=torch.long))
        self.gov.update_state(torch.full((J,), 0, dtype=torch.long))
        self.gov.reset()

        obs = _make_obs(phase=0, duration=15.0)
        logits_in = _uniform_logits()
        logits_out = self.gov.apply(logits_in, obs)

        for j in range(J):
            l = logits_out[j][0]
            # After reset, phase 3 should not be penalised
            self.assertGreaterEqual(l[3].item(), 0.0,
                                   f"Junction {j}: flicker penalty persists after reset")


class TestPressureBonus(unittest.TestCase):

    def setUp(self):
        self.gov = RuleGovernor(
            num_junctions=J, num_phases=P,
            pressure_boost=2.0, pressure_thresh=0.3
        )

    def test_congested_ns_gets_boost(self):
        """Heavy NS traffic → phase 0 (NS-through) should get positive boost."""
        obs = _make_obs(phase=0, duration=15.0, through_ns=25.0, through_ew=2.0)
        logits_in = _uniform_logits()
        logits_out = self.gov.apply(logits_in, obs)

        for j in range(J):
            l_in  = logits_in[j][0]
            l_out = logits_out[j][0]
            self.assertGreater(l_out[0].item(), l_in[0].item(),
                              f"Junction {j}: NS phase not boosted under congestion")

    def test_congested_ew_gets_boost(self):
        """Heavy EW traffic → phase 3 (EW-through) should get positive boost."""
        obs = _make_obs(phase=0, duration=15.0, through_ns=2.0, through_ew=25.0)
        logits_in = _uniform_logits()
        logits_out = self.gov.apply(logits_in, obs)

        for j in range(J):
            l_in  = logits_in[j][0]
            l_out = logits_out[j][0]
            self.assertGreater(l_out[3].item(), l_in[3].item(),
                              f"Junction {j}: EW phase not boosted under congestion")


class TestApplyOutputShape(unittest.TestCase):

    def setUp(self):
        self.gov = RuleGovernor(num_junctions=J, num_phases=P)

    def test_output_list_length(self):
        obs = _make_obs(phase=0, duration=15.0)
        out = self.gov.apply(_uniform_logits(), obs)
        self.assertEqual(len(out), J)

    def test_output_logit_shape(self):
        obs = _make_obs(phase=0, duration=15.0)
        out = self.gov.apply(_uniform_logits(), obs)
        for j, l in enumerate(out):
            self.assertEqual(l.shape, (1, P), f"Wrong logit shape at junction {j}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
