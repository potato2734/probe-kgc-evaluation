import numpy as np
import math
from collections import defaultdict


class PROBE():
    def __init__(self, model, data, info_dump, nentity, nrelation, alpha, beta, gamma, total_degree, rel, w_mode='t'):
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
        self.gamma = gamma
        self.rel = rel
        self.W = None



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
            e2p[query[0] if mode == 't' else query[2]] = p_e # universal truth
            re2p[tuple(query)].append(p_r_e) # universal truth
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
        return 1 / (epsilon + arr) ** param

    def w_relation_function(self, arr, param, epsilon):
        return 1 / (epsilon + arr) ** param

    def nonezero_min(self, x):
        nz_min = np.inf
        for i in x:
            if i != 0.0 and nz_min > i:
                nz_min = i
        return nz_min

    def set_entity_weight(self, beta):
        self.beta = beta
        self.entity_raw_weights = self.w_entity_function(self.t_p_e, self.beta, self.nonezero_min(self.t_p_e))

    def set_relation_weight(self, gamma):
        self.gamma = gamma
        self.relation_raw_weights = self.w_relation_function(self.t_p_r_e, self.beta, self.nonezero_min(self.t_p_r_e))

    def calculate_final_metric(self, alpha, beta, gamma, tmode='f*'):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.set_transform_function(alpha, mode=tmode)
        self.set_entity_weight(self.beta)
        self.set_relation_weight(self.beta)

        self.norm_W = PROBE.normalize_array(self.entity_raw_weights * self.relation_raw_weights)

        assert len(self.transformed_ranks) == len(self.norm_W), print(len(self.transformed_ranks), len(self.norm_W))

        final_metric = np.dot(self.transformed_ranks, self.norm_W)

        return final_metric


