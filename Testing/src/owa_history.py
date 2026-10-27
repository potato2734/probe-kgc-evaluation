"""Convert completed full/sparse PROBE exports to OWA metric histories."""
import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from rank_metrics import METRIC_NAMES, RankMetrics, record_weights


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.', suffix='.tmp', delete=False) as stream:
            temporary = stream.name
            json.dump(value, stream, allow_nan=False, indent=2)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def load_completed_export(path, aggregation='answer'):
    with path.open(encoding='utf-8') as stream:
        metadata = json.load(stream)
    if not metadata.get('complete_split') or metadata.get('tie_policy') != 'average':
        raise ValueError('Expected a complete average-rank export: %s' % path)
    record_path = path.parent / metadata['records']
    with record_path.open(encoding='utf-8') as stream:
        records = json.load(stream)
    if not records or len(records) != metadata['record_count']:
        raise ValueError('Record count mismatch or empty export: %s' % record_path)
    keys = [(tuple(row[0]), row[1]) for row in records]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate PROBE answer records: %s' % record_path)
    if any(row[1] != 't' for row in records):
        raise ValueError('Family OWA history expects tail-only records: %s' % record_path)
    ranks = np.array([row[2] for row in records], dtype=np.float64)
    filtered = np.array([row[5] for row in records], dtype=np.float64)
    if not np.isfinite(filtered).all() or np.any(filtered < 0) or np.any(filtered != np.floor(filtered)):
        raise ValueError('Invalid filtered counts: %s' % record_path)
    candidates = metadata['nentity'] - filtered
    weights = record_weights(records, aggregation=aggregation)
    # Every model/split has static targets and filtering; detect changes, do not silently reuse moments.
    signature = hashlib.sha256(json.dumps(keys, separators=(',', ':')).encode() +
                               candidates.tobytes()).hexdigest()
    return metadata, records, ranks, candidates, weights, signature


def build_owa_histories(probe_run, output_dir, aggregation='answer'):
    """Require all ten completed configurations; retain true training steps."""
    if aggregation not in ('answer', 'query'):
        raise ValueError('Aggregation must be answer or query')
    probe_run, output_dir = Path(probe_run).resolve(), Path(output_dir).resolve()
    model_specs = [(name, seed, '%s%d' % (name, i + 1))
                   for name, seeds in [('RotatE', range(10, 15)), ('ComplEx', range(20, 25))]
                   for i, seed in enumerate(seeds)]
    output_dir.mkdir(parents=True, exist_ok=True)
    details = {'source_run': str(probe_run), 'aggregation': aggregation,
               'null_model': 'independent discrete-uniform ranks on [1, N]',
               'wmr_weights': 'negative corruptions N - 1, multiplied by aggregation weights',
               'wmr_adjustment': '1 - WMR / E[WMR]; exact independent discrete expectation',
               'metric_names': list(METRIC_NAMES), 'complete': False, 'models': {}}
    moments = {}
    for model, seed, label in model_specs:
        run = probe_run / ('%s_seed%d' % (model, seed))
        config = json.loads((run / 'config.json').read_text())
        sessions = [path for path in (run / 'probe').iterdir() if path.is_dir()]
        if len(sessions) != 1:
            raise ValueError('Expected exactly one PROBE session for %s; found %d' % (run, len(sessions)))
        session = sessions[0]
        entries = {}
        for path in session.glob('*.metrics.json'):
            metadata = json.loads(path.read_text())
            key = (metadata['phase'], metadata['step'], metadata['split'])
            if key in entries:
                raise ValueError('Duplicate evaluation phase/step/split in %s' % session)
            entries[key] = path
        expected_steps = list(range(0, config['max_steps'], config['valid_steps']))
        evaluations = [('validation', step) for step in expected_steps] + [('final', config['max_steps'] - 1)]
        expected_keys = {(phase, step, split) for phase, step in evaluations for split in ('full', 'sparse')}
        if set(entries) != expected_keys:
            raise ValueError('Incomplete/unexpected history in %s: %d missing, %d extra split exports' %
                             (session, len(expected_keys - set(entries)), len(set(entries) - expected_keys)))
        history = {metric: [] for metric in METRIC_NAMES}
        steps, phases, baselines, signatures = [], [], {}, {}
        for phase, step in evaluations:
            scores = {}
            for split in ('full', 'sparse'):
                metadata, records, ranks, candidates, weights, signature = load_completed_export(entries[phase, step, split], aggregation)
                if metadata['model'] != model or metadata['seed'] != seed:
                    raise ValueError('Model/seed mismatch in %s' % entries[phase, step, split])
                if split in signatures and signatures[split] != signature:
                    raise ValueError('Query targets or candidate counts changed during %s/%s' % (label, split))
                signatures[split] = signature
                key = (signature, aggregation)
                if key not in moments:
                    moments[key] = RankMetrics(candidates, weights)
                calculator = moments[key]
                scores[split] = calculator.calculate(ranks)
                if split not in baselines:
                    baselines[split] = {
                        'record_count': len(records), 'query_count': metadata['query_count'],
                        'nentity': metadata['nentity'], 'candidate_count_min': int(candidates.min()),
                        'candidate_count_max': int(candidates.max()),
                        'expected_mrr': calculator.expected_mrr,
                        'mrr_variance': calculator.mrr_variance,
                        'expected_wmr': float(np.exp(calculator.log_expected_wmr)),
                    }
            for metric in METRIC_NAMES:
                history[metric].append([scores['full'][metric], scores['sparse'][metric]])
            steps.append(step)
            phases.append(phase)
        write_json(output_dir / (label + '.json'), history)
        write_json(output_dir / (label + '.steps.json'), {'steps': steps, 'phases': phases})
        details['models'][label] = {'model': model, 'seed': seed, 'session': str(session),
                                    'validation_evaluations': len(expected_steps),
                                    'history_points': len(steps), 'baselines': baselines,
                                    'final_metrics': {metric: history[metric][-1] for metric in METRIC_NAMES}}
        write_json(output_dir / 'manifest.json', details)
        print('%s: %d validation + final evaluations; full=%d, sparse=%d records' %
              (label, len(expected_steps), baselines['full']['record_count'], baselines['sparse']['record_count']), flush=True)
    details['complete'] = True
    write_json(output_dir / 'manifest.json', details)
    return output_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-run', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--aggregation', choices=('answer', 'query'), default='answer')
    args = parser.parse_args()
    build_owa_histories(args.probe_run, args.output_dir, args.aggregation)
