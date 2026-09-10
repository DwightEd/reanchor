import unittest

from reanchor.effects import BranchMargins, compute_effects


class ComputeEffectsTest(unittest.TestCase):
    def test_separates_source_and_prefix_main_effects(self) -> None:
        effects = compute_effects(
            BranchMargins(
                onset_a=2.0,
                onset_b=-2.0,
                world_a_after_a=2.0,
                world_a_after_b=-2.0,
                world_b_after_a=2.0,
                world_b_after_b=-2.0,
            )
        )

        self.assertEqual(effects.source_onset, 2.0)
        self.assertEqual(effects.source_followup, 0.0)
        self.assertEqual(effects.prefix_followup, 2.0)
        self.assertEqual(effects.source_prefix_coupling, 0.0)

    def test_coupling_is_invariant_to_interaction_sign(self) -> None:
        positive = compute_effects(BranchMargins(1.0, -1.0, 2.0, 0.0, 0.0, 2.0))
        negative = compute_effects(BranchMargins(1.0, -1.0, 0.0, 2.0, 2.0, 0.0))

        self.assertEqual(positive.source_prefix_coupling, 1.0)
        self.assertEqual(negative.source_prefix_coupling, 1.0)


if __name__ == "__main__":
    unittest.main()
