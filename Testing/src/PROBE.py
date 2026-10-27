import numpy as np
import math
import warnings
from collections import defaultdict
from rank_metrics import METRIC_NAMES, metrics_from_records

DEFAULT_ALPHA = 1.0
DEFAULT_BETA = 0.0

class PROBE():
    """PROBE(alpha, beta), with beta shared by entity and relation weights.

    The legacy gamma argument is accepted but ignored, as in earlier releases.
    Training-referenced epsilons are required when beta is nonzero.
    """
    def __init__(self, model, data, info_dump, nentity, nrelation, alpha=DEFAULT_ALPHA,
                 beta=DEFAULT_BETA, gamma=None, total_degree=None, rel=None, w_mode='t',
                 *, epsilon_entity=None, epsilon_relation=None):
        self._check_legacy_gamma(gamma)
        self.total_degree = total_degree
        self.model = model
        self.data = data
        self.info = info_dump
        self.w_mode = w_mode
        self.query, self.mode, self.piv_ls, self.e2p, self.rank, self.p_r_e, self.p_e, self.filtered = self.unpack_info(self.info)
        self.nentity = nentity
        self.nrelation = nrelation
        self.alpha = alpha
        self.beta = beta
        self.rel = rel
        self.W = None
        # These constants must be computed over the full training statistics.
        self.epsilon_entity = epsilon_entity
        self.epsilon_relation = epsilon_relation
        for name, epsilon in (('epsilon_entity', epsilon_entity),
                              ('epsilon_relation', epsilon_relation)):
            if epsilon is not None and (not math.isfinite(epsilon) or epsilon <= 0):
                raise ValueError(f'{name} must be finite and positive')

    def unpack_info(self, info):
        query_ls = []
        mode_ls = []
        piv_ls = []
        e2p = {}
        rank_ls = []
        p_r_e_ls = []
        p_e_ls = []
        filtered_ls = []
        re2p = defaultdict(list)
        info.sort()
        for query_info in info:
            query, mode, rank, p_r_e, p_e, filt_num = query_info
            query_ls.append(query)
            mode_ls.append(mode)
            piv_ls.append(query[0] if mode == 't' else query[2])
            e2p[query[0] if mode == 't' else query[2]] = p_e
            re2p[tuple(query)].append(p_r_e)
            rank_ls.append(rank)
            p_r_e_ls.append(p_r_e)
            p_e_ls.append(p_e)
            filtered_ls.append(filt_num)
        if self.w_mode == 't':
            t_p_e_ls = []
            t_p_r_e_ls = []
            piv_ls = []
            visited_query = set()
            for query_info in info:
                query, mode, rank, p_r_e, p_e, filt_num = query_info
                target = query[0] if mode == 'h' else query[2]
                piv_ls.append(target)
                t_p_e_ls.append(e2p[target])
                if tuple(query) not in visited_query:
                    t_p_r_e_ls.append(re2p[tuple(query)][1])
                    visited_query.add(tuple(query))
                else:
                    t_p_r_e_ls.append(re2p[tuple(query)][0])
            self.t_p_e = np.array(t_p_e_ls)
            self.t_p_r_e = np.array(t_p_r_e_ls)

        return query_ls, mode_ls, np.array(piv_ls), e2p, np.array(rank_ls), np.array(p_r_e_ls), np.array(p_e_ls), np.array(filtered_ls)

    def set_transform_function(self, alpha, mode='f*'):

        '''
        pre-requisite : self.raw_ranks, self.alpha

        calculation result : self.transformed_ranks
        '''
        self.alpha = alpha
        if mode == 'f':
            self.transformed_ranks = (1 / self.rank) ** self.alpha
            return

        elif mode == 'f*':
            if math.isclose(self.alpha, 0, abs_tol=1e-9):
                self.transformed_ranks = 1 - np.log(self.rank) / np.log(self.nentity - self.filtered)
            else:
                N_coeff = np.array([self.nentity] * len(self.filtered)) - self.filtered
                N_coeff = (1 / N_coeff) ** self.alpha
                self.transformed_ranks = ((1 / self.rank) ** self.alpha - N_coeff) / (1 - N_coeff)
        else:
            raise Exception(f'No mode supports \'{mode}\'')

    @staticmethod
    def normalize_array(arr):
        norm_arr = arr / np.sum(arr)
        assert math.isclose(sum(norm_arr), 1.0, rel_tol=1e-10)
        return norm_arr

    def w_entity_function(self, arr, param, epsilon):
        if param == 0:
            return np.ones_like(arr, dtype=float)
        if epsilon is None:
            raise ValueError('Nonuniform weights require a training-referenced epsilon')
        return 1 / (epsilon + arr) ** param

    def w_relation_function(self, arr, param, epsilon):
        return self.w_entity_function(arr, param, epsilon)

    @staticmethod
    def nonezero_min(x):
        nz_min = np.inf
        for probability in x:
            if not math.isfinite(probability) or probability < 0:
                raise ValueError('Training probabilities must be finite and nonnegative')
            if 0 < probability < nz_min:
                nz_min = probability
        if not math.isfinite(nz_min):
            raise ValueError('Training probabilities must contain a positive value')
        return nz_min

    @staticmethod
    def training_epsilons(entity_probabilities, relation_probabilities):
        """Minima over all training P(e) and P(r | e), excluding zeros."""
        return (
            PROBE.nonezero_min(entity_probabilities),
            PROBE.nonezero_min(probability
                              for row in relation_probabilities.values()
                              for probability in row),
        )

    def set_entity_weight(self, beta):
        self.beta = beta
        self.entity_raw_weights = self.w_entity_function(self.t_p_e, self.beta, self.epsilon_entity)

    @staticmethod
    def _check_legacy_gamma(gamma):
        if gamma is not None:
            warnings.warn(
                'PROBE gamma is deprecated and ignored; beta controls both weight factors',
                DeprecationWarning, stacklevel=3,
            )

    def set_relation_weight(self, beta):
        self.relation_raw_weights = self.w_relation_function(
            self.t_p_r_e, beta, self.epsilon_relation)

    def calculate_final_metric(self, alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA,
                               gamma=None, tmode='f*'):
        """Score with normalized ranks by default; tmode='f' uses raw rank powers."""
        self._check_legacy_gamma(gamma)
        self.set_transform_function(alpha, mode=tmode)
        self.set_entity_weight(beta)
        self.set_relation_weight(beta)

        self.norm_W = PROBE.normalize_array(self.entity_raw_weights * self.relation_raw_weights)

        assert len(self.transformed_ranks) == len(self.norm_W), print(len(self.transformed_ranks), len(self.norm_W))

        final_metric = np.dot(self.transformed_ranks, self.norm_W)

        return final_metric

    def calculate_rank_metric(self, metric, aggregation='answer'):
        """Calculate ZMRR, AMRI or WMR_BAR from this object's PROBE records."""
        if metric not in METRIC_NAMES:
            raise ValueError('Unknown rank metric: %s' % metric)
        return metrics_from_records(self.info, self.nentity, aggregation)[metric]
