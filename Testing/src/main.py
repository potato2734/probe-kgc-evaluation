import argparse
from utils import *
from collections import defaultdict
from vistool import VisTool
from PROBE import PROBE


def parse_args(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str)
    parser.add_argument('--models', nargs='+', type=str, default=[])

    parser.add_argument('--fig2_obs1', action='store_true')
    parser.add_argument('--fig2_obs2', action='store_true')
    parser.add_argument('--fig4', action='store_true')
    parser.add_argument('--fig5', action='store_true')
    parser.add_argument('--fig6_a', action='store_true')
    parser.add_argument('--fig6_bcd', action='store_true')
    parser.add_argument('--fig7', action='store_true')
    parser.add_argument('--table4', action='store_true')
    parser.add_argument('--table5', action='store_true')
    parser.add_argument('--fig9', action='store_true')
    parser.add_argument('--fig10_top', action='store_true')
    parser.add_argument('--fig10_bottom', action='store_true')
    parser.add_argument('--fig11', action='store_true')
    parser.add_argument('--fig12', action='store_true')

    return parser.parse_args(args)


def get_problem_count(ranks, tst_triples):
    ret_dict = defaultdict(int)
    for tup in tst_triples:
        h, r, t = tup
        ret_dict[h] += 1
        ret_dict[t] += 1
    return ret_dict

def main(args):
    data_path = f'../data/{args.data}/'

    e2id, id2e = get_e2id(data_path)
    r2id, id2r = get_r2id(data_path)
    nentity = len(e2id)
    nrelation = len(r2id)
    trn_triples, val_triples, tst_triples = get_triples(data_path, e2id, r2id)
    count_info_dict_trn = count_entities(trn_triples)
    count_info_dict_tst = get_problem_count(None, tst_triples)
    total_degree = sum(i for i in count_info_dict_trn.values())
    count_info_rel = count_relations(trn_triples)
    rel_prob = get_rel_distribution(trn_triples, len(r2id))

    info_ls = []

    print(args.models)

    met_ls = [None] * len(args.models)
    for idx, model in enumerate(args.models):
        info_ls = get_infos(model, args.data)
        met_ls[idx] = [PROBE(model=model, data=args.data, info_dump=info, nentity=nentity, nrelation=nrelation, alpha=1, beta=0,
                            gamma=0, total_degree=total_degree, rel=count_info_rel, w_mode='t') for
                            info in info_ls]


    alphas = [1.0, 0.5, 0.0, -0.5, -1.0]
    betas = [0.8, 0.6, 0.4, 0.2, 0.0]

    vis = VisTool(count_info_dict_trn, count_info_dict_tst, trn_triples, tst_triples, nentity, nrelation, total_degree, rel_prob, args.data)

    if args.fig2_obs1:
        vis.draw_rank_cluster_hist(met_ls,
                                cuts=(1, 2, 6, 21, 51, 100),
                                density=False,
                                both=True,
                                cut_last=True,
                                diff=True,
                                format='pdf',
                                xlabel=False)
        
    if args.fig9:
        vis.draw_rank_cluster_hist(met_ls,
                                cuts=(1, 3, 6, 11, 21, 51, 100), 
                                density=False,
                                both=True,
                                cut_last=True,
                                diff=True,
                                format='pdf',
                                xlabel=False)

    if args.fig2_obs2:
        vis.draw_bar_chart(5,
                        met_ls,
                        mode='zenon',
                        is_bin=True,
                        split_ax=True,
                        ref='t',
                        view='b',
                        p_r_e=rel_prob,
                        backoff=1,
                        format='pdf',
                        custom=[4, 7, 13, 25])

    if args.fig4:
        vis.draw_multiple_unit_alphas([1.0, 0.5, 0.25, 0.125], 
                                        ver='f')
        vis.draw_multiple_unit_alphas([1.0, 0.5, 0.25, 0.125], 
                                        ver='f*')

    if args.fig5:
        vis.draw_two_w(max_lines=50000,
                        plot=True,
                        bar=True)

    if args.fig6_a:
        vis.draw_multiple_unit_betas(met_ls[0][0],
                                        [0.0, 0.1, 0.2, 0.4, 0.8])

    if args.fig6_bcd:
        vis.draw_3d_weights(met=met_ls[0][0],
                            betas=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

    if args.fig7:
        VisTool.draw_vectors()

    if args.table4:
        vis.write_tables(met_ls,
                        [1.0, 0.5, 0.0, -0.5, -1.0],
                        [0.0],
                        None,
                        mode='a')

    if args.table5:
        vis.write_tables(met_ls,
                        [1.0],
                        [0.0, 0.2, 0.4, 0.6, 0.8],
                        None,
                        mode='b')

    if args.fig10_top:
        assert len(args.models) == 2, f'Model list length should be 2 for case study. Currently {len(args.models)}'
        vis.draw_e_cases(args,
                        e2id,
                        r2id,
                        threshold=20)

    if args.fig10_bottom:
        assert len(args.models) == 2, f'Model list length should be 2 for case study. Currently {len(args.models)}'
        vis.draw_pre_cases(met_ls,
                            id2e,
                            id2r)

    if args.fig11:
        vis.draw__hyperPlane(met_ls,
                            alphas,
                            betas,
                            None,
                            'ab')

    if args.fig12:
        VisTool.draw_owa()




if __name__ == '__main__':
    main(parse_args())