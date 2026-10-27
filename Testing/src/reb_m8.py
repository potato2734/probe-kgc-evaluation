"""Write the six-dataset epsilon sensitivity table as LaTeX."""
import json
from pathlib import Path

import numpy as np

from epsilon_sensitivity import DATASETS, ROOT, analyze_dataset
from epsilon_sensitivity_by_beta import BETAS, FACTORS, SCALES, DATASET_LABELS


SCALE_LABELS = (r'$1/4\times$', r'$1/2\times$', r'$2\times$', r'$4\times$')


def create_table(output_dir=None, from_json=None):
    """Write only m8.tex, recomputing rho or reading saved per-beta correlations.

    Both training epsilons are scaled together; alpha=1 and beta=gamma.
    Rank full-precision seed-mean scores, exclude AStarNet, and display raw rho.
    Return the LaTeX file path. No existing analysis artifacts are modified.
    """
    output_dir = Path(output_dir) if output_dir is not None else ROOT / 'figs/m8'
    if from_json is None:
        result = {
            'alpha': 1.0, 'betas': list(BETAS), 'epsilon_reference': 'train',
            'epsilon_factors': dict(FACTORS), 'datasets': {},
        }
        for dataset in DATASETS:
            result['datasets'][dataset] = analyze_dataset(dataset, betas=BETAS, factors=FACTORS)
            print(f'M8: calculated {dataset}', flush=True)
    else:
        result = json.loads(Path(from_json).read_text(encoding='utf-8'))
    if result['betas'] != list(BETAS) or result['epsilon_factors'] != FACTORS:
        raise ValueError('M8 requires betas 0.2/0.4/0.6/0.8 and epsilon factors 1/4, 1/2, 2, 4')
    if result.get('alpha') != 1.0 or result.get('epsilon_reference') != 'train':
        raise ValueError('M8 requires alpha=1 and training-derived epsilons')

    values = {}
    for dataset in DATASETS:
        columns = []
        for state in SCALES:
            correlations = np.asarray(result['datasets'][dataset]['comparisons'][f'BASE-{state}']['per_beta'], dtype=float)
            if (correlations.shape != (len(BETAS),) or not np.isfinite(correlations).all()
                    or np.any(np.abs(correlations) > 1)):
                raise ValueError(f'Invalid correlations for {dataset}/{state}')
            columns.append(correlations)
        correlations = np.column_stack(columns)
        values[dataset] = np.column_stack((correlations, correlations.mean(axis=1)))

    lines = [r'% Requires \usepackage{booktabs}', r'\begin{table*}[t]',
             r'\centering', r'\small', r'\setlength{\tabcolsep}{4pt}',
             r'\renewcommand{\arraystretch}{0.7}', '',
             r'\caption{',
             'Sensitivity of model rankings to smoothing parameters.',
             r'Spearman correlations ($\rho$) between BASE and jointly scaled',
             r'training-derived $\epsilon_e$ and $\epsilon_{r|e}$,',
             r'with $\alpha=1$.',
             'Models are ranked by their mean scores over three seeds.',
             'Avg is the arithmetic mean across the four scaling factors for each dataset and beta.',
             '}', r'\label{tab:epsilon-sensitivity}', '']
    for index, group in enumerate((DATASETS[:3], DATASETS[3:])):
        if index:
            lines.extend([r'\vspace{0.9em}', ''])
        lines.extend([f'% ==================== Panel {"AB"[index]} ====================',
                      r'\begin{tabular*}{\textwidth}{',
                      r'@{}c@{\extracolsep{\fill}}rrrrr rrrrr rrrrr@{}',
                      '}', r'\toprule'])
        for k, dataset in enumerate(group):
            line = r'& \multicolumn{5}{c}{\textbf{' + DATASET_LABELS[dataset] + '}}'
            lines.append(line + (r' \\' if k == len(group) - 1 else ''))
        lines.extend([r'\cmidrule(lr){2-6}', r'\cmidrule(lr){7-11}',
                      r'\cmidrule(l){12-16}', '', r'$\beta$'])
        for k in range(len(group)):
            line = '& ' + ' & '.join((*SCALE_LABELS, 'Avg'))
            lines.append(line + (r' \\' if k == len(group) - 1 else ''))
        lines.append(r'\midrule')
        for i, beta in enumerate(BETAS):
            lines.append(f'{beta:.1f}')
            for k, dataset in enumerate(group):
                line = '& ' + ' & '.join(f'{value:.4f}' for value in values[dataset][i])
                lines.append(line + (r' \\' if k == len(group) - 1 else ''))
            if i < len(BETAS) - 1:
                lines.append('')
        lines.extend([r'\bottomrule', r'\end{tabular*}', ''])
    lines.extend([r'\end{table*}', ''])
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / 'm8.tex'
    path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Saved M8 LaTeX table (96 correlations and 24 averages) to {path}', flush=True)
    return path
