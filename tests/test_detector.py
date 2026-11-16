import unittest

from wakeword_trainer.detector import ScoreGate


class ScoreGateTests(unittest.TestCase):
    def test_requires_a_consecutive_run_and_emits_once(self):
        gate = ScoreGate(threshold=0.5, confirmation_frames=4)
        scores = [0.9, 0.9, 0.2, 0.9, 0.9, 0.9, 0.9, 0.9, 0.1, 0.9, 0.9, 0.9, 0.9]
        self.assertEqual(
            [i for i, score in enumerate(scores) if gate.update(score)], [6, 12],
        )

    def test_explicit_reset_clears_an_incomplete_run(self):
        gate = ScoreGate(confirmation_frames=2)
        self.assertFalse(gate.update(0.8))
        gate.reset()
        self.assertFalse(gate.update(0.8))
        self.assertTrue(gate.update(0.8))


if __name__ == "__main__":
    unittest.main()
