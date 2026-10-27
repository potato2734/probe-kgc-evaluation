"""Recompute baseline metrics from saved PROBE records, without model inference."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from rank_metrics import METRIC_NAMES, metrics_from_records, record_weights


BASELINE_NAMES = ('MRR', 'HITS@1', 'HITS@3', 'HITS@10', 'P2_3', 'P1_3', *METRIC_NAMES)
ROOT = Path(__file__).resolve().parents[1]


def score_records(records, nentity, aggregation='answer'):
    """Use the same saved ranks, candidate counts, and averaging for all metrics."""
    scores = metrics_from_records(records, nentity, aggregation)
    ranks = np.asarray([row[2] for row in records], dtype=float)
    weights = record_weights(records, aggregation)
    values = {'MRR': 1 / ranks, 'P2_3': ranks ** (-2 / 3),
              'P1_3': ranks ** (-1 / 3)}
    values.update({f'HITS@{k}': ranks <= k for k in (1, 3, 10)})
    scores.update({key: float(np.average(value, weights=weights))
                   for key, value in values.items()})
    return {name: scores[name] for name in BASELINE_NAMES}


def build_report(sources, nentity, aggregation='answer'):
    """Retain each source/score and summarize independent runs within each model."""
    if isinstance(nentity, bool) or not isinstance(nentity, int) or nentity < 2:
        raise ValueError('nentity must be an integer of at least two')
    if not sources:
        raise ValueError('At least one model is required')
    report = {'nentity': nentity, 'aggregation': aggregation,
              'candidates': 'N = nentity - filtered_count; includes target',
              'rank_policy': 'Use saved ranks, including fractional tie ranks, unchanged',
              'null_model': 'Independent discrete-uniform ranks on 1..N',
              'wmr_adjustment': '1 - WMR / E[WMR], exact independent discrete expectation',
              'std_ddof': 1, 'metrics': list(BASELINE_NAMES), 'models': {}}
    for model, paths in sources.items():
        if not paths:
            raise ValueError(f'No saved PROBE JSON files for {model}')
        runs = []
        for path in sorted(map(Path, paths)):
            payload = path.read_bytes()
            records = json.loads(payload)
            runs.append({'source': str(path.resolve()),
                         'sha256': hashlib.sha256(payload).hexdigest(),
                         'record_count': len(records),
                         'scores': score_records(records, nentity, aggregation)})
        summary = {}
        for name in BASELINE_NAMES:
            values = [run['scores'][name] for run in runs]
            summary[name] = {'mean': float(np.mean(values)),
                             'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}
        report['models'][model] = {'runs': runs, 'summary': summary}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', help='Dataset under Testing/data (with --models)')
    parser.add_argument('--models', nargs='+', help='Models under Testing/infos')
    parser.add_argument('--records', type=Path, nargs='+', help='Explicit PROBE JSON files (with --nentity)')
    parser.add_argument('--nentity', type=int)
    parser.add_argument('--aggregation', choices=('answer', 'query'), default='answer')
    parser.add_argument('--output', type=Path, help='JSON report; also writes a CSV with the same stem')
    args = parser.parse_args(argv)
    if args.records:
        if args.data or args.models or args.nentity is None:
            parser.error('--records requires --nentity and cannot be combined with --data/--models')
        if len({path.stem for path in args.records}) != len(args.records):
            parser.error('Explicit record filenames must have unique stems')
        sources = {path.stem: [path] for path in args.records}
        nentity = args.nentity
    else:
        if not args.data or not args.models or args.nentity is not None:
            parser.error('Use --data DATASET --models MODEL ... or --records FILE ... --nentity N')
        with (ROOT / 'data' / args.data / 'entities.dict').open(encoding='utf-8') as stream:
            nentity = sum(1 for line in stream if line.strip())
        sources = {model: list((ROOT / 'infos' / model / args.data).glob('*.json'))
                   for model in args.models}
    report = build_report(sources, nentity, args.aggregation)
    if args.data:
        report['dataset'] = args.data
    output = args.output or ROOT / 'figs' / 'baselines' / f'{args.data or "records"}_{args.aggregation}.json'
    if output.suffix.lower() != '.json':
        parser.error('--output must have a .json extension')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    csv_path = output.with_suffix('.csv')
    with csv_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model', 'source', 'record_count', *BASELINE_NAMES])
        for model, result in report['models'].items():
            for run in result['runs']:
                writer.writerow([model, run['source'], run['record_count'],
                                 *(run['scores'][name] for name in BASELINE_NAMES)])
    print(f'Saved {output} and {csv_path}')
    return report


if __name__ == '__main__':
    main()
