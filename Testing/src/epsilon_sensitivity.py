"""Compare model rankings under jointly scaled training epsilons for Table 5."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

from PROBE import PROBE
from utils import count_entities, get_e2id, get_r2id, get_rel_distribution, read_triple


DATASETS = ("FB15k237", "wn18rr", "YAGO3-10", "family", "umls", "kinship")
MODELS = ("RotatE", "ComplEx", "HousE", "TuckER", "pLogicNet", "RNNLogic")
BETAS = (0.0, 0.2, 0.4, 0.6, 0.8)
FACTORS = {"BASE": 1.0, "UP": 4.0, "DOWN": 0.25}
ROOT = Path(__file__).resolve().parents[1]


def analyze_dataset(dataset, betas=BETAS, factors=FACTORS):
    data_dir = ROOT / "data" / dataset
    entity_ids, _ = get_e2id(str(data_dir))
    relation_ids, _ = get_r2id(str(data_dir))
    train_triples = read_triple(str(data_dir / "train.txt"), entity_ids, relation_ids)
    degrees = count_entities(train_triples)
    total_degree = sum(degrees.values())
    relation_probabilities = get_rel_distribution(train_triples, len(relation_ids))
    epsilon_entity, epsilon_relation = PROBE.training_epsilons(
        (degree / total_degree for degree in degrees.values()), relation_probabilities)
    models = [model for model in MODELS
              if not (dataset == "YAGO3-10" and model == "RNNLogic")]
    scores = {state: {} for state in factors}
    run_files = {}
    for model in models:
        files = sorted((ROOT / "infos" / model / dataset).glob("*.json"))
        if not files:
            raise ValueError(f"No result files for {dataset}/{model}")
        run_files[model] = [str(path.relative_to(ROOT)) for path in files]
        for state in factors:
            scores[state][model] = []
        for path in files:
            records = json.loads(path.read_text())
            met = PROBE(model, dataset, records, len(entity_ids), len(relation_ids),
                        total_degree=total_degree, rel=relation_probabilities,
                        epsilon_entity=epsilon_entity, epsilon_relation=epsilon_relation)
            for state, factor in factors.items():
                met.epsilon_entity = epsilon_entity * factor
                met.epsilon_relation = epsilon_relation * factor
                values = [float(met.calculate_final_metric(1.0, beta))
                          for beta in betas]
                if not np.isfinite(values).all():
                    raise ValueError(f"Nonfinite metric for {dataset}/{model}/{state}")
                scores[state][model].append(values)
    means = {state: np.array([np.mean(scores[state][model], axis=0)
                              for model in models]) for state in factors}
    ranks = {state: np.column_stack([rankdata(-means[state][:, i], method="average")
                                     for i in range(len(betas))])
             for state in factors}
    comparisons = {}
    for state in factors:
        if state == "BASE":
            continue
        correlations = []
        for i in range(len(betas)):
            base_rank = ranks["BASE"][:, i]
            changed_rank = ranks[state][:, i]
            rho = float(spearmanr(base_rank, changed_rank).statistic)
            if not np.isfinite(rho):
                raise ValueError(f"Undefined Spearman coefficient for {dataset}/{state}")
            # Independently verify rho from the already ranked vectors.
            np.testing.assert_allclose(rho, np.corrcoef(base_rank, changed_rank)[0, 1],
                                       atol=1e-14, rtol=0)
            if len(set(base_rank)) == len(models) and len(set(changed_rank)) == len(models):
                n = len(models)
                exact = 1 - 6 * np.sum((base_rank - changed_rank)**2) / (n * (n*n - 1))
                np.testing.assert_allclose(rho, exact, atol=1e-14, rtol=0)
            correlations.append(rho)
        comparisons[f"BASE-{state}"] = {
            "per_beta": correlations, "average": float(np.mean(correlations))}
        # Uniform weighting at beta=0 must be invariant to epsilon scaling.
        if 0.0 in betas:
            zero_index = betas.index(0.0)
            np.testing.assert_array_equal(means["BASE"][:, zero_index],
                                          means[state][:, zero_index])
            np.testing.assert_allclose(correlations[zero_index], 1.0, atol=1e-14, rtol=0)
    return {
        "models": models, "run_files": run_files,
        "epsilons": {state: {"entity": epsilon_entity * factor,
                              "relation": epsilon_relation * factor}
                     for state, factor in factors.items()},
        "scores_by_run": scores,
        "mean_scores": {state: {model: means[state][i].tolist()
                                 for i, model in enumerate(models)} for state in factors},
        "model_ranks": {state: {model: ranks[state][i].tolist()
                                 for i, model in enumerate(models)} for state in factors},
        "comparisons": comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figs/epsilon_sensitivity")
    args = parser.parse_args()
    result = {
        "alpha": 1.0, "betas": list(BETAS), "epsilon_reference": "train",
        "epsilon_factors": FACTORS, "excluded_models": ["AStarNet"],
        "ranking": "Descending full-precision mean score across available seeds; average ranks for ties",
        "aggregation": "Arithmetic mean of five per-beta Spearman coefficients, including beta=0",
        "datasets": {},
    }
    for dataset in args.datasets:
        entry = analyze_dataset(dataset)
        result["datasets"][dataset] = entry
        print(dataset, {mode: comparison["average"]
                        for mode, comparison in entry["comparisons"].items()}, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "epsilon_sensitivity.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n")
    with (args.output_dir / "epsilon_sensitivity.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["dataset", "BASE-UP", "BASE-DOWN"])
        for dataset, entry in result["datasets"].items():
            writer.writerow([dataset, entry["comparisons"]["BASE-UP"]["average"],
                             entry["comparisons"]["BASE-DOWN"]["average"]])
    print(f"Saved results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
