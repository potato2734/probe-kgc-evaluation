import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rank_metrics import CANDIDATE_SWEEP_METRICS, candidate_size_sweep
from vistool import VisTool
from main import parse_args


class CandidateSizeSweepTests(unittest.TestCase):
    def test_single_rank_against_analytic_metrics(self):
        scales = [0.8, 1.0, 1.2]
        result = candidate_size_sweep([10], [100], scales)
        scores = result['scores']
        for i, n in enumerate([80, 100, 120]):
            self.assertAlmostEqual(scores['PROBE(a=1.0)'][i], (n / 10 - 1) / (n - 1))
            endpoint = n ** -0.5
            self.assertAlmostEqual(scores['PROBE(a=0.5)'][i], (10 ** -0.5 - endpoint) / (1 - endpoint))
            self.assertAlmostEqual(scores['PROBE(a=0.0)'][i], 1 - np.log(10) / np.log(n))
            self.assertAlmostEqual(scores['P2_3'][i], 10 ** (-2 / 3))
            self.assertAlmostEqual(scores['P1_3'][i], 10 ** (-1 / 3))
            self.assertAlmostEqual(scores['AMRI'][i], 1 - 18 / (n - 1))
            self.assertAlmostEqual(scores['WMR_BAR'][i], 1 - 20 / (n + 1))
            mu = sum(1 / j for j in range(1, n + 1)) / n
            variance = sum(1 / j**2 for j in range(1, n + 1)) / n - mu**2
            self.assertAlmostEqual(scores['ZMRR'][i], (0.1 - mu) / variance**0.5)
        for name, values in scores.items():
            if name in ('P2_3', 'P1_3'):
                np.testing.assert_array_equal(np.diff(values), 0)
            else:
                self.assertTrue(np.all(np.diff(values) > 0))

    def test_fixed_cohort_rounding_fractional_ranks_and_unchanged_inputs(self):
        ranks = np.array([1.5, 9., 1.])
        candidates = np.array([5, 10, 1])
        saved_ranks, saved_candidates = ranks.copy(), candidates.copy()
        result = candidate_size_sweep(ranks, candidates, [0.8, 0.9, 1.0, 1.2])
        self.assertEqual(result['retained_indices'], [0])
        self.assertEqual(result['excluded_count'], 2)
        # 5 * 0.9 = 4.5 is rounded up to five candidates.
        self.assertAlmostEqual(result['scores']['AMRI'][1], 0.75)
        self.assertAlmostEqual(result['scores']['AMRI'][2], 0.75)
        np.testing.assert_array_equal(ranks, saved_ranks)
        np.testing.assert_array_equal(candidates, saved_candidates)
        # The removed rank 9 is excluded even at scales where it becomes valid.
        single = candidate_size_sweep([1.5], [5], [0.8, 0.9, 1.0, 1.2])
        self.assertEqual(result['scores'], single['scores'])

    def test_invalid_scales_and_no_valid_cohort(self):
        for scales in ([], [1, 0.8], [1, 1], [0], [float('nan')], [[1]]):
            with self.subTest(scales=scales), self.assertRaises(ValueError):
                candidate_size_sweep([2], [10], scales)
        with self.assertRaisesRegex(ValueError, 'No records'):
            candidate_size_sweep([9], [10], [0.8, 1.0])

    def test_real_figures_and_repeated_runs(self):
        vis = VisTool({}, {}, [], [], 100, 1, 0, {}, 'fixture')
        runs = [SimpleNamespace(model='RotatE', data='fixture', nentity=100,
                                rank=np.array([rank]), filtered=np.array([0]))
                for rank in [5., 10.]]
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            with patch.object(vis, '_save_reb_m15c', wraps=vis._save_reb_m15c) as render:
                paths = vis.draw_reb_m15c([runs], output_dir=directory)
            summary = render.call_args.args[0]
            self.assertEqual(len(paths), len(CANDIDATE_SWEEP_METRICS))
            for path in paths:
                self.assertTrue(Path(path).read_bytes().startswith(b'%PDF'))
            self.assertEqual(len(summary['scales']), 7)
            self.assertEqual(summary['pivot_scale'], 1.0)
            pivot = summary['scales'].index(1.0)
            model = summary['models']['RotatE']
            raw = np.array([run['scores']['AMRI'] for run in model['runs']])
            values = raw - raw[:, pivot, None]
            np.testing.assert_allclose([run['differences']['AMRI'] for run in model['runs']], values)
            self.assertEqual(model['metrics']['AMRI']['mean'][pivot], 0.0)
            self.assertEqual(model['metrics']['AMRI']['std'][pivot], 0.0)
            np.testing.assert_allclose(model['metrics']['AMRI']['mean'], values.mean(axis=0))
            np.testing.assert_allclose(model['metrics']['AMRI']['std'], values.std(axis=0, ddof=1))
            for metric, stats in model['metrics'].items():
                self.assertEqual(stats['mean'][pivot], 0.0)
                self.assertEqual(stats['std'][pivot], 0.0)
            for metric in ('P2_3', 'P1_3'):
                np.testing.assert_array_equal(model['metrics'][metric]['mean'], 0)
                np.testing.assert_array_equal(model['metrics'][metric]['std'], 0)
                for run in model['runs']:
                    np.testing.assert_array_equal(run['differences'][metric], 0)
            # A custom sweep omitting the pivot must still use the scale-1 score.
            with patch.object(vis, '_save_reb_m15c', wraps=vis._save_reb_m15c) as render:
                vis.draw_reb_m15c([runs], scales=[0.8, 1.2], output_dir=directory)
            custom = render.call_args.args[0]
            self.assertEqual(custom['scales'], [0.8, 1.0, 1.2])
            delta = custom['models']['RotatE']['metrics']['AMRI']['mean']
            expected = np.mean([1 - 2 * (rank - 1) / 119 - (1 - 2 * (rank - 1) / 99)
                                for rank in [5., 10.]])
            self.assertAlmostEqual(delta[-1], expected)
            self.assertLess(delta[0], 0)
            self.assertGreater(delta[-1], 0)
        np.testing.assert_array_equal(runs[0].rank, [5.])
        np.testing.assert_array_equal(runs[1].filtered, [0])
        self.assertEqual(plt.get_fignums(), [])
        self.assertTrue(parse_args(['--m15c']).m15c)


if __name__ == '__main__':
    unittest.main()
