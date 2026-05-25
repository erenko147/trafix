"""
Unit tests — TraFixV6 model internals (GRU, GATConv, PPO actor-critic).
Run: python tests/mandatory/test_unit_model.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import unittest
import torch
from trafix_v6.trafix_v6 import TraFixV6, _TemporalEncoder, _GraphEncoder, _make_chain_edge_index

J = 5   # junctions
T = 30  # time window
D = 20  # obs_dim
P = 6   # phases


class TestTemporalEncoder(unittest.TestCase):

    def setUp(self):
        self.enc = _TemporalEncoder(obs_dim=D, hidden_dim=128)
        self.enc.eval()

    def test_output_shape(self):
        obs = torch.zeros(1, T, J, D)
        out = self.enc(obs)
        self.assertEqual(out.shape, (1, J, 128))

    def test_batch_shape(self):
        obs = torch.zeros(4, T, J, D)
        out = self.enc(obs)
        self.assertEqual(out.shape, (4, J, 128))

    def test_no_nan(self):
        obs = torch.randn(1, T, J, D)
        out = self.enc(obs)
        self.assertFalse(torch.isnan(out).any(), "NaN in GRU output")

    def test_different_inputs_different_outputs(self):
        a = self.enc(torch.zeros(1, T, J, D))
        b = self.enc(torch.ones(1, T, J, D))
        self.assertFalse(torch.allclose(a, b), "GRU output identical for different inputs")


class TestGraphEncoder(unittest.TestCase):

    def setUp(self):
        self.enc = _GraphEncoder(in_channels=128, heads=4, head_dim=32)
        self.enc.eval()
        self.edge_index = _make_chain_edge_index(J)

    def test_output_shape(self):
        x = torch.zeros(J, 128)
        out = self.enc(x, self.edge_index)
        self.assertEqual(out.shape, (J, 128))  # 4 heads × 32 = 128

    def test_no_nan(self):
        x = torch.randn(J, 128)
        out = self.enc(x, self.edge_index)
        self.assertFalse(torch.isnan(out).any(), "NaN in GATConv output")


class TestTraFixV6Forward(unittest.TestCase):

    def setUp(self):
        self.model = TraFixV6(obs_dim=D, num_phases=P)
        self.model.eval()

    def _window(self, batch=1):
        return torch.zeros(batch, T, J, D)

    def test_logits_list_length(self):
        logits, _ = self.model(self._window())
        self.assertEqual(len(logits), J)

    def test_logits_shape(self):
        logits, _ = self.model(self._window())
        for j, l in enumerate(logits):
            self.assertEqual(l.shape, (1, P), f"Junction {j} logits shape wrong")

    def test_value_shape(self):
        _, value = self.model(self._window())
        self.assertEqual(value.shape, (1, J))

    def test_no_nan_in_logits(self):
        logits, _ = self.model(torch.randn(1, T, J, D))
        for j, l in enumerate(logits):
            self.assertFalse(torch.isnan(l).any(), f"NaN in logits for junction {j}")

    def test_no_nan_in_value(self):
        _, value = self.model(torch.randn(1, T, J, D))
        self.assertFalse(torch.isnan(value).any(), "NaN in value output")

    def test_actor_probabilities_sum_to_one(self):
        logits, _ = self.model(self._window())
        for j, l in enumerate(logits):
            probs = torch.softmax(l, dim=-1)
            self.assertAlmostEqual(probs.sum().item(), 1.0, places=5,
                                   msg=f"Probs don't sum to 1 for junction {j}")

    def test_actor_probabilities_non_negative(self):
        logits, _ = self.model(self._window())
        for j, l in enumerate(logits):
            probs = torch.softmax(l, dim=-1)
            self.assertTrue((probs >= 0).all(),
                            f"Negative probability for junction {j}")

    def test_chosen_phase_in_valid_range(self):
        logits, _ = self.model(self._window())
        for j, l in enumerate(logits):
            phase = int(torch.argmax(l, dim=-1).item())
            self.assertGreaterEqual(phase, 0)
            self.assertLess(phase, P)

    def test_batch_inference(self):
        batch = 8
        logits, value = self.model(self._window(batch))
        for l in logits:
            self.assertEqual(l.shape[0], batch)
        self.assertEqual(value.shape, (batch, J))


class TestTraFixV6GetAction(unittest.TestCase):

    def setUp(self):
        self.model = TraFixV6(obs_dim=D, num_phases=P)
        self.model.eval()

    def test_action_shape(self):
        obs = torch.zeros(1, T, J, D)
        actions, log_probs, value = self.model.get_action(obs)
        self.assertEqual(actions.shape, (1, J))
        self.assertEqual(log_probs.shape, (1, J))
        self.assertEqual(value.shape, (1, J))

    def test_actions_in_valid_range(self):
        obs = torch.zeros(1, T, J, D)
        actions, _, _ = self.model.get_action(obs)
        self.assertTrue((actions >= 0).all())
        self.assertTrue((actions < P).all())

    def test_log_probs_finite(self):
        obs = torch.zeros(1, T, J, D)
        _, log_probs, _ = self.model.get_action(obs)
        self.assertTrue(torch.isfinite(log_probs).all())


class TestTraFixV6EvaluateActions(unittest.TestCase):

    def setUp(self):
        self.model = TraFixV6(obs_dim=D, num_phases=P)
        self.model.eval()

    def test_evaluate_actions_shapes(self):
        obs = torch.zeros(4, T, J, D)
        actions = torch.zeros(4, J, dtype=torch.long)
        log_probs, entropy, value = self.model.evaluate_actions(obs, actions)
        self.assertEqual(log_probs.shape, (4, J))
        self.assertEqual(entropy.shape, (4, J))
        self.assertEqual(value.shape, (4, J))

    def test_entropy_non_negative(self):
        obs = torch.randn(4, T, J, D)
        actions = torch.randint(0, P, (4, J))
        _, entropy, _ = self.model.evaluate_actions(obs, actions)
        self.assertTrue((entropy >= 0).all(), "Entropy must be non-negative")


class TestTraFixV6Checkpoint(unittest.TestCase):
    """Verify that the final checkpoint loads and produces valid output."""

    CKPT = pathlib.Path(__file__).resolve().parents[2] / "trafix_v6/checkpoints/trafix_v6_final.pt"

    def test_checkpoint_exists(self):
        self.assertTrue(self.CKPT.exists(), f"Checkpoint not found: {self.CKPT}")

    def test_checkpoint_loads(self):
        if not self.CKPT.exists():
            self.skipTest("Checkpoint not found")
        model = TraFixV6(obs_dim=D, num_phases=P)
        ckpt = torch.load(self.CKPT, map_location="cpu", weights_only=True)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
        model.eval()

        obs = torch.zeros(1, T, J, D)
        logits, value = model(obs)
        self.assertEqual(len(logits), J)
        self.assertFalse(torch.isnan(value).any())


if __name__ == "__main__":
    unittest.main(verbosity=2)
