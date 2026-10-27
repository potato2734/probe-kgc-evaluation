"""Produce per-beta epsilon sensitivity results and a grouped LaTeX table."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from epsilon_sensitivity import DATASETS, ROOT, analyze_dataset


BETAS = (0.2, 0.4, 0.6, 0.8)
FACTORS = {"BASE": 1.0, "DOWN": 0.25, "DOWN_2": 0.5, "UP_2": 2.0, "UP": 4.0}
SCALES = tuple(state for state in FACTORS if state != "BASE")
SCALE_LABELS = (r"$\times\frac{1}{4}$", r"$\times\frac{1}{2}$", r"$\times 2$", r"$\times 4$")
DATASET_LABELS = {"FB15k237": "FB15k-237", "wn18rr": "WN18RR",
                  "YAGO3-10": "YAGO3-10", "family": "Family",
                  "umls": "UMLS", "kinship": "Kinship"}


def write_outputs(result, output_dir, normalization):
    assert result["betas"] == list(BETAS)
    result["normalization"] = {
        "mode": normalization,
        "formula": "50 * (rho + 1)" if normalization == "range" else "100 * rho",
        "display_decimals": 2,
    }
    cells = []
    for dataset, entry in result["datasets"].items():
        for state in SCALES:
            key = f"BASE-{state}"
            correlations = entry["comparisons"][key]["per_beta"]
            if len(correlations) != len(BETAS) or not np.isfinite(correlations).all():
                raise ValueError(f"Invalid correlations for {dataset}/{state}")
            normalized = [float(np.clip(50 * (rho + 1), 0, 100))
                          if normalization == "range" else float(100 * rho)
                          for rho in correlations]
            entry["comparisons"][key] = {
                "per_beta": correlations, "normalized_per_beta": normalized}
            for beta, rho, value in zip(BETAS, correlations, normalized):
                cells.append((dataset, beta, FACTORS[state], rho, value))
    assert len(cells) == len(result["datasets"]) * 4 * 4
    result["correlation_count"] = len(cells)
    output_dir.mkdir(parents=True, exist_ok=True)
    basename = "epsilon_sensitivity_by_beta"
    (output_dir / f"{basename}.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n")
    with (output_dir / f"{basename}.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["dataset", "beta", "epsilon_factor", "spearman_rho", "normalized"])
        writer.writerows(cells)

    normalization_latex = r"$50(\rho+1)$" if normalization == "range" else r"$100\rho$"
    lines = [
        r"% Requires \usepackage{booktabs,graphicx}",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Spearman correlations between BASE and jointly scaled training",
        r"epsilons for each $\beta$, with $\alpha=1$. Values are displayed as "
        + normalization_latex + r". Models are ranked by their mean score across",
        r"three seeds; AStarNet is excluded.}",
        r"\label{tab:epsilon-sensitivity}",
        r"\resizebox{\textwidth}{!}{%",
    ]
    datasets = list(result["datasets"])
    for start in range(0, len(datasets), 3):
        group = datasets[start:start + 3]
        if start > 0:
            lines.extend([r"\par\medskip", r"\resizebox{\textwidth}{!}{%"])
        lines.extend([r"\begin{tabular}{c|" + "|".join("rrrr" for _ in group) + "}",
                      r"\toprule"])
        headers = [""]
        for i, dataset in enumerate(group):
            spec = "c|" if i < len(group)-1 else "c"
            headers.append(r"\multicolumn{4}{" + spec + "}{" + DATASET_LABELS[dataset] + "}")
        lines.append(" & ".join(headers) + r" \\")
        lines.append(" ".join(r"\cmidrule(lr){" + str(2+4*i) + "-" + str(5+4*i) + "}"
                              for i in range(len(group))))
        lines.append(" & ".join([r"$\beta$"] + list(SCALE_LABELS) * len(group)) + r" \\")
        lines.append(r"\midrule")
        for i, beta in enumerate(BETAS):
            row = [f"{beta:.1f}"]
            for dataset in group:
                for state in SCALES:
                    value = result["datasets"][dataset]["comparisons"][f"BASE-{state}"]["normalized_per_beta"][i]
                    row.append(f"{value:.2f}")
            lines.append(" & ".join(row) + r" \\")
        lines.extend([r"\bottomrule", r"\end{tabular}", "}"])
    lines.extend([r"\end{table*}", ""])
    (output_dir / f"{basename}.tex").write_text("\n".join(lines))
    print(f"Saved {len(cells)} correlations and LaTeX table to {output_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "figs/epsilon_sensitivity_by_beta")
    parser.add_argument("--normalization", choices=("range", "percent"), default="range")
    parser.add_argument("--from-json", type=Path,
                        help="Reformat saved raw correlations without rerunning scores")
    args = parser.parse_args()
    if args.from_json:
        result = json.loads(args.from_json.read_text())
    else:
        result = {
            "alpha": 1.0, "betas": list(BETAS), "epsilon_reference": "train",
            "epsilon_factors": FACTORS, "excluded_models": ["AStarNet"],
            "ranking": "Descending full-precision mean score across three seeds; average ranks for ties",
            "aggregation": "One Spearman coefficient per dataset, epsilon scale, and beta; beta=0 excluded",
            "datasets": {},
        }
        for dataset in DATASETS:
            entry = analyze_dataset(dataset, betas=BETAS, factors=FACTORS)
            result["datasets"][dataset] = entry
            print(dataset, {key: comparison["per_beta"]
                            for key, comparison in entry["comparisons"].items()}, flush=True)
    write_outputs(result, args.output_dir, args.normalization)


if __name__ == "__main__":
    main()
