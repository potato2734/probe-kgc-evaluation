"""Run selected paper analyses across datasets using the current interpreter."""
import argparse
from pathlib import Path
import subprocess
import sys


DATASETS = ('FB15k237', 'wn18rr', 'YAGO3-10', 'family', 'umls', 'kinship')
MODELS = ('RotatE', 'ComplEx', 'HousE', 'TuckER', 'pLogicNet', 'RNNLogic')
FLAGS = ('table4', 'table5', 'fig11', 'm9', 'm15c', 'm16')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--models', nargs='+', default=None)
    parser.add_argument('--baselines', action='store_true', help='Export baseline JSON/CSV reports')
    for flag in FLAGS:
        parser.add_argument('--' + flag, action='store_true')
    args = parser.parse_args(argv)
    flags = ['--' + flag for flag in FLAGS if getattr(args, flag)]
    if not flags and not args.baselines:
        flags = ['--table4', '--table5']
    source_dir = Path(__file__).resolve().parent
    for dataset in args.datasets:
        models = args.models or [model for model in MODELS
                                 if not (dataset == 'YAGO3-10' and model == 'RNNLogic')]
        commands = []
        if flags:
            commands.append([sys.executable, str(source_dir / 'main.py'),
                             '--data', dataset, '--models', *models, *flags])
        if args.baselines:
            commands.append([sys.executable, str(source_dir / 'baseline_metrics.py'),
                             '--data', dataset, '--models', *models])
        for command in commands:
            print('Running:', ' '.join(command), flush=True)
            subprocess.run(command, cwd=source_dir, check=True)


if __name__ == '__main__':
    main()
