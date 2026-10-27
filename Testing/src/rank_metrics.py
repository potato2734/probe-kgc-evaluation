"""Chance-adjusted rank metrics under independent discrete-uniform null ranks.

Candidates include the target. WMR weights count negative corruptions (N - 1).
The WMR adjustment uses the exact expected geometric mean, not the geometric
mean of expected ranks. All three metrics are oriented so larger is better.
"""
import numpy as np


METRIC_NAMES = ('ZMRR', 'AMRI', 'WMR_BAR')
METRIC_LABELS = {'ZMRR': 'ZMRR', 'AMRI': 'AMRI', 'WMR_BAR': r'$\overline{\mathrm{WMR}}$'}

CANDIDATE_SWEEP_METRICS = ('PROBE(a=1.0)', 'PROBE(a=0.5)', 'PROBE(a=0.0)',
                           'P2_3', 'P1_3', *METRIC_NAMES)


class RankMetrics:
    """Cache the null moments for a fixed set of ranking tasks."""

    def __init__(self, num_candidates, weights=None):
        candidates = np.asarray(num_candidates, dtype=np.float64)
        if (candidates.ndim != 1 or not candidates.size or
                not np.isfinite(candidates).all() or
                np.any(candidates < 1) or np.any(candidates != np.floor(candidates))):
            raise ValueError('Candidate counts must be a nonempty vector of positive integers')
        self.num_candidates = candidates.astype(np.int64)
        if weights is None:
            weights = np.ones(candidates.size, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        if (weights.shape != candidates.shape or not np.isfinite(weights).all() or
                np.any(weights < 0) or weights.sum() <= 0):
            raise ValueError('Metric weights must be finite, nonnegative, and have positive total mass')
        self.weights = weights / weights.sum()

        indices = np.arange(1, self.num_candidates.max() + 1, dtype=np.float64)
        harmonic = np.cumsum(1 / indices)
        harmonic2 = np.cumsum(1 / indices**2)
        mu = harmonic[self.num_candidates - 1] / candidates
        variance = np.maximum(harmonic2[self.num_candidates - 1] / candidates - mu**2, 0)
        self.expected_mrr = float(np.dot(self.weights, mu))
        self.mrr_variance = float(np.dot(self.weights**2, variance))
        self.expected_rank_minus_one = float(np.dot(self.weights, (candidates - 1) / 2))

        corruption_weights = self.weights * (candidates - 1)
        if corruption_weights.sum() > 0:
            corruption_weights /= corruption_weights.sum()
        self.wmr_weights = corruption_weights
        self.log_expected_wmr = 0.0
        # Many answers share N and w; compute their exact null moment once.
        tasks, counts = np.unique(np.column_stack((candidates, corruption_weights)),
                                  axis=0, return_counts=True)
        for (candidate_count, weight), count in zip(tasks, counts):
            if weight == 0:
                continue
            log_ranks = np.log(indices[:int(candidate_count)])
            # E[R**w] is close to 1 for small w. expm1/log1p preserve precision.
            log_moment = np.log1p(np.expm1(weight * log_ranks).mean())
            self.log_expected_wmr += int(count) * float(log_moment)

    def calculate(self, ranks):
        ranks = np.asarray(ranks, dtype=np.float64)
        if (ranks.shape != self.num_candidates.shape or not np.isfinite(ranks).all() or
                np.any(ranks < 1) or np.any(ranks > self.num_candidates)):
            raise ValueError('Ranks must be finite and lie in [1, number of candidates]')
        if self.mrr_variance <= 0 or self.expected_rank_minus_one <= 0:
            raise ValueError('Chance-adjusted metrics are undefined when all weighted candidate sets have size one')
        zmrr = (np.dot(self.weights, 1 / ranks) - self.expected_mrr) / np.sqrt(self.mrr_variance)
        amri = 1 - np.dot(self.weights, ranks - 1) / self.expected_rank_minus_one
        log_wmr = np.dot(self.wmr_weights, np.log(ranks))
        wmr_bar = -np.expm1(log_wmr - self.log_expected_wmr)
        return {'ZMRR': float(zmrr), 'AMRI': float(amri), 'WMR_BAR': float(wmr_bar)}


def calculate_rank_metrics(ranks, num_candidates, weights=None):
    return RankMetrics(num_candidates, weights).calculate(ranks)


def candidate_size_sweep(ranks, num_candidates, scales):
    """Keep ranks and the evaluation cohort fixed while scaling candidate counts.

    Counts include the target and are rounded half up. The fixed cohort contains
    only records with N >= 2 and rank <= N at every scale. PROBE uses alpha
    1, 0.5, and 0 with uniform answer weights (beta=gamma=0). The 2/3-MRR
    and 1/3-MRR scores depend only on fixed ranks and remain constant. WMR
    corruption weights and all null moments are recomputed at each scale.
    """
    ranks = np.asarray(ranks, dtype=np.float64)
    candidates = np.asarray(num_candidates, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    if (scales.ndim != 1 or not scales.size or not np.isfinite(scales).all()
            or np.any(scales <= 0) or np.any(np.diff(scales) <= 0)):
        raise ValueError('Scales must be finite, positive, and strictly increasing')
    if (candidates.ndim != 1 or not candidates.size
            or not np.isfinite(candidates).all() or np.any(candidates < 1)
            or np.any(candidates != np.floor(candidates))):
        raise ValueError('Candidate counts must be a nonempty vector of positive integers')
    if (ranks.shape != candidates.shape or not np.isfinite(ranks).all()
            or np.any(ranks < 1) or np.any(ranks > candidates)):
        raise ValueError('Ranks must be finite and lie in [1, number of candidates]')
    smallest = np.floor(candidates * scales[0] + 0.5)
    retained = (smallest >= 2) & (ranks <= smallest)
    if not retained.any():
        raise ValueError('No records are valid at every candidate-size scale')
    fixed_ranks, fixed_candidates = ranks[retained], candidates[retained]
    scores = {name: [] for name in CANDIDATE_SWEEP_METRICS}
    pmrr = {name: float(np.mean((1 / fixed_ranks) ** power))
            for name, power in (('P2_3', 2 / 3), ('P1_3', 1 / 3))}
    for scale in scales:
        scaled = np.floor(fixed_candidates * scale + 0.5).astype(np.int64)
        metrics = calculate_rank_metrics(fixed_ranks, scaled)
        # Equivalent to PROBE.set_transform_function(alpha=1, mode='f*').
        scores['PROBE(a=1.0)'].append(float(np.mean(
            (scaled / fixed_ranks - 1) / (scaled - 1))))
        endpoint = np.sqrt(1 / scaled)
        scores['PROBE(a=0.5)'].append(float(np.mean(
            (np.sqrt(1 / fixed_ranks) - endpoint) / (1 - endpoint))))
        scores['PROBE(a=0.0)'].append(float(np.mean(
            1 - np.log(fixed_ranks) / np.log(scaled))))
        for name, value in pmrr.items():
            scores[name].append(value)
        for name in METRIC_NAMES:
            scores[name].append(metrics[name])
    return {'record_count': int(ranks.size), 'retained_count': int(retained.sum()),
            'excluded_count': int((~retained).sum()),
            'retained_indices': np.flatnonzero(retained).tolist(), 'scores': scores}


def record_weights(records, aggregation='answer'):
    """Uniform answer weights, or equal weight for each head/relation query."""
    if aggregation == 'answer':
        return None
    if aggregation != 'query':
        raise ValueError('Aggregation must be answer or query')
    from collections import Counter
    keys = [(row[1], row[0][0] if row[1] == 't' else row[0][2], row[0][1])
            for row in records]
    counts = Counter(keys)
    return np.array([1 / counts[key] for key in keys], dtype=np.float64)


def metrics_from_records(records, nentity, aggregation='answer'):
    if not records:
        raise ValueError('Cannot calculate metrics from an empty PROBE export')
    ranks = np.array([row[2] for row in records], dtype=np.float64)
    filtered = np.array([row[5] for row in records], dtype=np.float64)
    if np.any(filtered < 0) or np.any(filtered != np.floor(filtered)):
        raise ValueError('Filtered counts must be nonnegative integers')
    return calculate_rank_metrics(ranks, nentity - filtered, record_weights(records, aggregation))
