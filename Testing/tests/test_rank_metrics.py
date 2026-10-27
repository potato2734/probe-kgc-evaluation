import contextlib
import io
import itertools
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rank_metrics import RankMetrics, calculate_rank_metrics, metrics_from_records, record_weights, METRIC_NAMES
from owa_history import build_owa_histories, write_json
from vistool import VisTool, OWA_MODEL_COLORS
from main import parse_args, main


class RankMetricTests(unittest.TestCase):
    def test_exact_null_mean_and_z_variance_by_exhaustive_enumeration(self):
        for weights in (None, [0.2, 0.8]):
            calculator = RankMetrics([2, 4], weights)
            combinations = list(itertools.product(range(1, 3), range(1, 5)))
            results = [calculator.calculate(ranks) for ranks in combinations]
            for name in METRIC_NAMES:
                self.assertAlmostEqual(np.mean([result[name] for result in results]), 0, places=12)
            self.assertAlmostEqual(np.var([result['ZMRR'] for result in results]), 1, places=12)
            raw_weights = np.array([1, 3]) * calculator.weights
            wmr_weights = raw_weights / raw_weights.sum()
            expected_wmr = np.mean([np.prod(np.array(ranks, dtype=float)**wmr_weights)
                                    for ranks in combinations])
            self.assertAlmostEqual(np.exp(calculator.log_expected_wmr), expected_wmr, places=12)

    def test_amri_endpoints_and_fractional_tie_ranks(self):
        self.assertAlmostEqual(calculate_rank_metrics([1, 1], [3, 5])['AMRI'], 1)
        self.assertAlmostEqual(calculate_rank_metrics([3, 5], [3, 5])['AMRI'], -1)
        self.assertAlmostEqual(calculate_rank_metrics([2, 3], [3, 5])['AMRI'], 0)
        self.assertTrue(all(np.isfinite(list(calculate_rank_metrics([1.5, 2.5], [3, 4]).values()))))
        single = calculate_rank_metrics([1, 2], [1, 3])
        self.assertAlmostEqual(single['AMRI'], 0)

    def test_invalid_counts_ranks_weights_and_degenerate_null(self):
        for candidates in ([], [0], [2.5], [np.nan], [[3]]):
            with self.subTest(candidates=candidates), self.assertRaises(ValueError):
                RankMetrics(candidates)
        for weights in ([0, 0], [-1, 1], [np.inf, 1], [1]):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                RankMetrics([3, 4], weights)
        for ranks in ([0, 2], [4, 2], [np.nan, 2], [1]):
            with self.subTest(ranks=ranks), self.assertRaises(ValueError):
                RankMetrics([3, 4]).calculate(ranks)
        with self.assertRaisesRegex(ValueError, 'undefined'):
            RankMetrics([1, 1]).calculate([1, 1])

    def test_record_candidates_and_query_weights(self):
        records = [[[0, 0, 2], 't', 2.5, 1., .5, 2],
                   [[0, 0, 3], 't', 2.5, 1., .5, 2],
                   [[1, 0, 4], 't', 3., 1., .2, 1]]
        np.testing.assert_allclose(record_weights(records, 'query'), [0.5, 0.5, 1])
        self.assertEqual(metrics_from_records(records, 6), calculate_rank_metrics([2.5, 2.5, 3], [4, 4, 5]))
        macro = metrics_from_records(records, 6, 'query')
        expected = calculate_rank_metrics([2.5, 2.5, 3], [4, 4, 5], [0.25, 0.25, 0.5])
        self.assertEqual(macro, expected)
        self.assertAlmostEqual(macro['AMRI'], 0)


class OwaPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def create_completed_run(self):
        root = self.root / 'all10_fixture'
        full = [[[0, 0, 2], 't', 2.5, 1., .5, 2],
                [[0, 0, 3], 't', 2.5, 1., .5, 2],
                [[1, 0, 4], 't', 3., 1., .2, 1]]
        sparse = [[[0, 0, 2], 't', 3., 1., .5, 1]]
        for model, seeds in [('RotatE', range(10, 15)), ('ComplEx', range(20, 25))]:
            for seed in seeds:
                run = root / ('%s_seed%d' % (model, seed))
                session = run / 'probe' / 'session'
                session.mkdir(parents=True)
                write_json(run / 'config.json', {'max_steps': 2, 'valid_steps': 1})
                for phase, step in [('validation', 0), ('validation', 1), ('final', 1)]:
                    for split, rows in [('full', full), ('sparse', sparse)]:
                        basename = '%s_step_%09d_%s' % (phase, step, split)
                        write_json(session / (basename + '.json'), rows)
                        write_json(session / (basename + '.metrics.json'), {
                            'complete_split': True, 'tie_policy': 'average', 'records': basename + '.json',
                            'record_count': len(rows), 'query_count': 2 if split == 'full' else 1,
                            'phase': phase, 'step': step, 'split': split, 'nentity': 6,
                            'model': model, 'seed': seed})
        return root

    def test_converter_covers_all_ten_steps_splits_and_metrics(self):
        run = self.create_completed_run()
        data = self.root / 'owa'
        with contextlib.redirect_stdout(io.StringIO()):
            build_owa_histories(run, data)
        manifest = json.loads((data / 'manifest.json').read_text())
        self.assertTrue(manifest['complete'])
        self.assertEqual(len(manifest['models']), 10)
        for model in OWA_MODEL_COLORS:
            history = json.loads((data / (model + '.json')).read_text())
            self.assertEqual(set(history), set(METRIC_NAMES))
            self.assertEqual(json.loads((data / (model + '.steps.json')).read_text())['steps'], [0, 1, 1])
            for values in history.values():
                self.assertEqual(np.array(values).shape, (3, 2))
                self.assertTrue(np.isfinite(values).all())
        missing = next((run / 'RotatE_seed10' / 'probe').glob('*/validation*full.metrics.json'))
        missing.unlink()
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            build_owa_histories(run, self.root / 'broken')

    def create_plot_histories(self):
        data = self.root / 'histories'
        for i, model in enumerate(OWA_MODEL_COLORS):
            write_json(data / (model + '.json'), {
                'ZMRR': [[-3 - i, -2 - i], [1000 + i*20, 800 + i*10], [2000 + i*20, 1600 + i*10]],
                'AMRI': [[-0.2, -0.3], [0.4 + i/100, 0.3 + i/100], [0.9 - i/100, 0.8 + i/100]],
                'WMR_BAR': [[-0.1, -0.2], [0.6, 0.5], [0.95 - i/100, 0.8 + i/100]],
            })
            write_json(data / (model + '.steps.json'), {'steps': [0, 100, 99999],
                                                       'phases': ['validation', 'validation', 'final']})
        return data

    def test_plotters_signed_scores_steps_filenames_and_ranking(self):
        data = self.create_plot_histories()
        captured = {}
        def capture(figure, filename, **kwargs):
            ax = figure.axes[0]
            captured[Path(filename).name] = {'xlabel': ax.get_xlabel(), 'scale': ax.get_yscale(),
                                            'xlim': ax.get_xlim(), 'ylim': ax.get_ylim(),
                                            'lines': [np.asarray(line.get_xdata()).copy() for line in ax.lines]}
        with patch.object(Figure, 'savefig', capture), contextlib.redirect_stdout(io.StringIO()):
            for metric in METRIC_NAMES:
                output = VisTool.reb_m14(data, self.root / 'm14', metric)
                self.assertEqual(Path(output).name, metric + '.pdf')
                plot = captured[metric + '.pdf']
                self.assertEqual(plot['scale'], 'symlog')
                self.assertIn('Training step', plot['xlabel'])
                np.testing.assert_array_equal(plot['lines'][0], [0, 100, 99999])
            summary = VisTool.draw_owa(data, self.root / 'fig12', METRIC_NAMES)
        self.assertGreater(captured['ZMRR.pdf']['xlim'][1], 2000)
        self.assertLess(captured['ZMRR.pdf']['xlim'][0], -3)
        self.assertGreater(captured['ZMRR.pdf']['ylim'][1], 1600)
        self.assertAlmostEqual(summary['AMRI']['kendall_tau'], -1)
        self.assertAlmostEqual(summary['AMRI']['spearman_rho'], -1)
        self.assertEqual(summary['AMRI']['selected_models']['RotatE1']['step'], 99999)
        self.assertEqual(len(captured), 6)  # Three plots and three rank-change boxes.

    def test_standalone_main_uses_histories_without_dataset_infos(self):
        data = self.create_plot_histories()
        args = parse_args(['--m14', '--fig12', '--metrics', 'ZMRR', 'AMRI', 'WMR_BAR',
                           '--owa-data-dir', str(data), '--m14-output-dir', str(self.root / 'm14'),
                           '--owa-output-dir', str(self.root / 'fig12')])
        with patch('main.get_e2id', side_effect=AssertionError('OWA must not read dataset dictionaries')), \
                patch.object(Figure, 'savefig'), contextlib.redirect_stdout(io.StringIO()):
            main(args)
        self.assertEqual(len(list((self.root / 'm14').glob('*.summary.json'))), 3)
        self.assertTrue((self.root / 'fig12' / 'summary.json').exists())


if __name__ == '__main__':
    unittest.main()
