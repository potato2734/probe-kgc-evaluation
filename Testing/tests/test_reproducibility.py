"""Public defaults, compatibility, baseline exports, and analysis CLI routing."""
import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from test_training_epsilon import IMPLEMENTATIONS
from baseline_metrics import BASELINE_NAMES, main as baseline_main, score_records
from main import main, parse_args
from vistool import VisTool


RECORDS = [
    [[0, 0, 2], 'h', 2, 0.5, 0.25, 0],
    [[0, 0, 2], 't', 3, 0.5, 0.375, 0],
    [[0, 1, 1], 'h', 1, 0, 0.125, 0],
    [[0, 1, 1], 't', 2, 1/3, 0.375, 0],
]


class ReproducibilityTests(unittest.TestCase):
    def test_public_defaults_and_legacy_gamma_preserve_scores(self):
        for cls in IMPLEMENTATIONS:
            with self.subTest(implementation=cls.__module__):
                met = cls('fixture', 'fixture', list(RECORDS), 6, 2,
                          epsilon_entity=1/8, epsilon_relation=1/3)
                self.assertEqual((met.alpha, met.beta), (1.0, 0.0))
                expected = np.mean([(6 / row[2] - 1) / 5 for row in RECORDS])
                self.assertAlmostEqual(met.calculate_final_metric(), expected)
                self.assertAlmostEqual(met.calculate_final_metric(tmode='f'),
                                       np.mean([1 / row[2] for row in RECORDS]))
                score = met.calculate_final_metric(alpha=0.5, beta=0.4)
                weights = (met.t_p_e + 1/8)**-0.4 * (met.t_p_r_e + 1/3)**-0.4
                np.testing.assert_allclose(met.norm_W, weights / weights.sum())
                with self.assertWarnsRegex(DeprecationWarning, 'ignored'):
                    legacy = met.calculate_final_metric(0.5, 0.4, 99)
                self.assertAlmostEqual(score, legacy)
                with self.assertWarnsRegex(DeprecationWarning, 'ignored'):
                    old = cls('fixture', 'fixture', list(RECORDS), 6, 2,
                              1, 0, 99, 8, {})
                self.assertAlmostEqual(old.calculate_final_metric(), expected)

    def test_answer_and_query_averaging_use_the_same_cohort(self):
        records = [[[0, 0, 1], 't', 1, 0, 0, 0],
                   [[0, 0, 2], 't', 2, 0, 0, 0],
                   [[1, 0, 2], 't', 4, 0, 0, 0]]
        answer = score_records(records, 6)
        query = score_records(records, 6, 'query')
        self.assertAlmostEqual(answer['MRR'], (1 + 1/2 + 1/4) / 3)
        self.assertAlmostEqual(query['MRR'], ((1 + 1/2) / 2 + 1/4) / 2)
        self.assertAlmostEqual(answer['HITS@1'], 1/3)
        self.assertAlmostEqual(query['HITS@1'], 1/4)
        self.assertAlmostEqual(query['P2_3'], (1 + 2**(-2/3))/4 + 4**(-2/3)/2)
        self.assertEqual(set(answer), set(BASELINE_NAMES))
        # Fractional ranks must not be rounded or tie-broken again.
        fractional = [[[0, 0, 1], 't', 1.5, 0, 0, 0]]
        self.assertAlmostEqual(score_records(fractional, 6)['MRR'], 2/3)
        self.assertEqual(score_records(fractional, 6)['HITS@1'], 0.0)

    def test_standalone_export_records_sources_and_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / 'run.json', root / 'out' / 'metrics.json'
            source.write_text(json.dumps(RECORDS))
            original = source.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                baseline_main(['--records', str(source), '--nentity', '6', '--output', str(output)])
            report = json.loads(output.read_text())
            self.assertEqual(report['aggregation'], 'answer')
            run = report['models']['run']['runs'][0]
            self.assertEqual(run['sha256'], hashlib.sha256(original).hexdigest())
            self.assertEqual(run['record_count'], len(RECORDS))
            self.assertEqual(run['scores'], score_records(RECORDS, 6))
            self.assertTrue(output.with_suffix('.csv').is_file())
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(report['models']['run']['summary']['MRR']['std'], 0.0)

    def test_dataset_cli_reaches_requested_m15c_analysis(self):
        args = parse_args(['--data', 'fixture', '--models', 'RotatE', '--m15c'])
        with patch('main.get_e2id', return_value=({str(i): i for i in range(6)}, {i: str(i) for i in range(6)})), \
                patch('main.get_r2id', return_value=({'r': 0, 's': 1}, {0: 'r', 1: 's'})), \
                patch('main.get_triples', return_value=([], [], [])), \
                patch('main.count_entities', return_value={0: 3, 1: 1, 2: 2, 3: 2}), \
                patch('main.count_relations', return_value={0: 2, 1: 2}), \
                patch('main.get_rel_distribution', return_value={0: [2/3, 1/3]}), \
                patch('main.get_infos', return_value=[list(RECORDS)]), \
                patch.object(VisTool, 'draw_reb_m15c') as draw, \
                contextlib.redirect_stdout(io.StringIO()):
            main(args)
        draw.assert_called_once()
        met = draw.call_args.args[0][0][0]
        self.assertAlmostEqual(met.epsilon_entity, 1/8)
        self.assertAlmostEqual(met.epsilon_relation, 1/3)

    def test_m8_redraw_does_not_load_unrelated_dataset(self):
        with patch('main.get_e2id', side_effect=AssertionError('No dataset required')), \
                patch.object(VisTool, 'reb_m8') as draw:
            main(parse_args(['--m8', '--m8-from-json', '/tmp/previous.json']))
        draw.assert_called_once()


if __name__ == '__main__':
    unittest.main()
