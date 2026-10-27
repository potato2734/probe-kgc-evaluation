import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Testing/src"))


def load_probe(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PROBE


IMPLEMENTATIONS = (
    load_probe(ROOT / "PROBE.py", "training_probe_root"),
    load_probe(ROOT / "Testing/src/PROBE.py", "training_probe_testing"),
)


class TrainingEpsilonTests(unittest.TestCase):
    def make_probe(self, cls, records, **epsilons):
        return cls("Fixture", "Fixture", records, 6, 2, 1, 0, 0, 8, {},
                   **epsilons)

    def test_full_training_minima_include_values_absent_from_evaluation(self):
        # Degrees from (0,0,1), (0,0,2), (0,1,2), (3,1,3).
        entity_probabilities = [3/8, 1/8, 2/8, 2/8, 0, 0]
        relation_probabilities = {
            0: [2/3, 1/3], 1: [1, 0], 2: [1/2, 1/2], 3: [0, 1],
            4: [0, 0], 5: [0, 0],
        }
        subset = [[[0, 0, 2], "h", 2, 1/2, 2/8, 0],
                  [[0, 0, 2], "t", 3, 2/3, 3/8, 0]]
        extra = [[[0, 1, 1], "h", 1, 0, 1/8, 0],
                 [[0, 1, 1], "t", 2, 1/3, 3/8, 0]]
        for cls in IMPLEMENTATIONS:
            with self.subTest(implementation=cls.__module__):
                eps_e, eps_r = cls.training_epsilons(
                    iter(entity_probabilities), relation_probabilities)
                self.assertEqual(eps_e, 1/8)
                self.assertEqual(eps_r, 1/3)
                kwargs = dict(epsilon_entity=eps_e, epsilon_relation=eps_r)
                small = self.make_probe(cls, list(subset), **kwargs)
                large = self.make_probe(cls, list(subset + extra), **kwargs)
                for met in (small, large):
                    met.set_entity_weight(0.4)
                    met.set_relation_weight(0.6)
                self.assertGreater(min(small.t_p_e), eps_e)
                self.assertGreater(min(small.t_p_r_e), eps_r)
                np.testing.assert_allclose(small.entity_raw_weights,
                                           (eps_e + small.t_p_e)**-0.4)
                np.testing.assert_allclose(small.relation_raw_weights,
                                           (eps_r + small.t_p_r_e)**-0.6)
                for i, (query, mode) in enumerate(zip(small.query, small.mode)):
                    j = next(j for j, pair in enumerate(zip(large.query, large.mode))
                             if pair == (query, mode))
                    self.assertEqual(small.entity_raw_weights[i],
                                     large.entity_raw_weights[j])
                    self.assertEqual(small.relation_raw_weights[i],
                                     large.relation_raw_weights[j])

    def test_zero_probability_targets_have_finite_weights_and_metric(self):
        records = [[[4, 0, 5], "h", 2, 0, 0, 0],
                   [[4, 0, 5], "t", 3, 0, 0, 0]]
        for cls in IMPLEMENTATIONS:
            with self.subTest(implementation=cls.__module__):
                met = self.make_probe(cls, list(records),
                                      epsilon_entity=1/8, epsilon_relation=1/3)
                self.assertTrue(np.isfinite(met.calculate_final_metric(1, 0.8, 0.8)))
                self.assertTrue(np.isfinite(met.entity_raw_weights).all())
                self.assertTrue(np.isfinite(met.relation_raw_weights).all())
                np.testing.assert_allclose(met.norm_W, [0.5, 0.5])

    def test_missing_reference_cannot_silently_use_evaluation_minimum(self):
        records = [[[0, 0, 2], "h", 2, 0.5, 0.25, 0],
                   [[0, 0, 2], "t", 3, 0.5, 0.375, 0]]
        for cls in IMPLEMENTATIONS:
            with self.subTest(implementation=cls.__module__):
                met = self.make_probe(cls, list(records))
                self.assertAlmostEqual(met.calculate_final_metric(1, 0, 0, "f"),
                                       (1/2 + 1/3)/2)
                with self.assertRaisesRegex(ValueError, "training-referenced"):
                    met.set_entity_weight(0.4)
                with self.assertRaisesRegex(ValueError, "training-referenced"):
                    met.set_relation_weight(0.4)

    def test_invalid_or_empty_training_reference_fails_explicitly(self):
        for cls in IMPLEMENTATIONS:
            for values in ([], [0, 0], [-1, 0.5], [np.nan], [np.inf]):
                with self.subTest(implementation=cls.__module__, values=values):
                    with self.assertRaises(ValueError):
                        cls.training_epsilons(values, {0: [0.5, 0.5]})
                    with self.assertRaises(ValueError):
                        cls.training_epsilons([0.5], {0: values})


if __name__ == "__main__":
    unittest.main()
