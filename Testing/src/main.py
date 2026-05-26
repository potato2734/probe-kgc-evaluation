import argparse
from utils import *
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
    return {k: len(v) for k, v in ranks.items()}

def main(args):
    ranking_per_model = []
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
    # for model in args.models:
    #     info_ls.append(get_infos(model, args.data))
    # probe_ls = []
    # for info in info_ls:
    #     probe_ls.append(PROBE(info_dump=info[0], nentity=nentity, alpha=1, beta=0, gamma=0))
    print(args.models)
    # print(round(total_degree / nentity, 1), max(count_info_dict_trn.values()))
    # exit()
    met_ls = [None] * len(args.models)
    for idx, model in enumerate(args.models):
        info_ls = get_infos(model, args.data)
        met_ls[idx] = [PROBE(model=model, data=args.data, info_dump=info, nentity=nentity, nrelation=nrelation, alpha=1, beta=0,
                             gamma=0, total_degree=total_degree, rel=count_info_rel, w_mode='t') for
                            info in info_ls]


    alphas = [1.0, 0.5, 0.0, -0.5, -1.0]
    betas = [0.8, 0.6, 0.4, 0.2, 0.0]

    vis = VisTool(count_info_dict_trn, count_info_dict_tst, trn_triples, tst_triples, nentity, nrelation, total_degree, rel_prob, args.data)

    # #vis.draw_pop_dependency(tst_triples, count_info_dict_trn)
    #
    # # vis.draw_evolution(met_ls, alphas, 'a')
    # # vis.draw_evolution(met_ls, betas, 'b')
    # # vis.draw_evolution(met_ls, gammas, 'g')
    #
    # #vis.fit_degree()
    # '''Etc simulation'''
    # #vis.draw_two_w()
    # # exit()
    # #vis.draw_re_trn_tst(met_ls[0][0])
    # #vis.draw_trn_tst(args.data)
    # #vis.draw_degree_points()
    # # vis.draw_multiple_unit_alphas([1.0, 0.5, 0.25], ver='f')
    # # vis.draw_multiple_unit_alphas([1.0, 0.5, 0.25, 0.0, -1.0], ver='f*')
    # #for b in betas: vis.draw_weight_scales(met_ls, b, b)
    # #vis.draw_multiple_unit_betas(met_ls[0][0], [0.0,0.1,0.2,0.4,0.8])
    # # vis.draw_weight_correlation(met_ls, 0.2, 0.2, mode='add', transform='None', plot_kind='hexbin')
    # # vis.draw_weight_correlation(met_ls, 0.2, 0.2, mode='mul', transform='None', plot_kind='hexbin')
    # # vis.draw_corr_evolution(met_ls, [i*0.1 for i in range(1,10)], [i*0.1 for i in range(1,10)])
    #
    # #vis.draw_multiple_models_withAlphas(met_ls, [i*0.1 for i in range(-10, 21)], 0.0, 0.0, mode='n')
    #
    # # for a in [1.0]: vis.draw_multiple_models_withBetas(met_ls, a, [i * 0.05 for i in range(20, -1, -1)], 1, mode='n')
    # #vis.draw_gain_alphaWise(met_ls, [i*0.1 for i in range(-10, 21)])
    # # vis.draw_scores_bar_models(met_ls, [i*1 for i in range(-1, 3)])
    # # for a in alphas:
    # #     vis.draw_multiple_models_withBetas(met_ls, a, betas, 1, mode='n')
    # # for b in betas:
    # #     vis.draw_multiple_models_withAlphas(met_ls, alphas, b, 1, mode='n')
    # # vis.draw_plane(met_ls)
    # ''' Case studies & two of us figs '''
    # if 0:
    #     key = 'a'
    #     if key == 'a':
    #         # (1,3,5,7,9,11,16,21,41,61,81,100)
    #         # vis.draw_rank_cluster_hist(met_ls, cuts=(1,3,6,11,21,51,100), density=False, both=True, cut_last=True,
    #         #                            diff=True, format='pdf', xlabel=False)
    #         for a in [round(-1 + i*0.25, 2) for i in range(9)]:
    #             vis.draw_for_table(met_ls, a, 0.0, 0.0, target='a', figh=1.5, figw=1.5)
    #     if key == 'b':
    #         vis.draw_e_cases(args, e2id, r2id, threshold=20)
    #         # for b in [round(i*0.1, 1) for i in range(9)]:
    #         #     vis.draw_for_table(met_ls, 1.0, b, 0.0, target='b', figh=1.5, figw=1.5)
    #     if key == 'g':
    #         vis.draw_pre_cases(met_ls, id2e, id2r)
    #         # for g in gammas:
    #         #     vis.draw_for_table(met_ls, 1.0, 0.0, g, target='g')
    #
    # ''' Global case '''
    # # assert len(args.models) == 2, f'Model list length should be 2. Currently {len(args.models)}'
    # # vis.draw_e_cases(args, e2id, r2id, threshold=20)
    #
    # ''' Local case '''
    # #vis.draw_pre_cases(met_ls, id2e, id2r)
    #
    # #vis.draw_conventions(10)
    # #vis.draw_alphas(alphas, x=60)
    #
    # #vis.draw_data_points(4, args.data)
    #
    # #vis.draw_piv_tar_heatmap(5, met_ls, 'zenon')
    #
    # #vis.draw_piv_target_accuracy(met_ls)
    # #vis.draw_multiple_models(met_ls, 1.0, 0.0, 0.0, mode='g')
    # ''' Bar plots '''
    # if 0:
    #     # vis.draw_bar_chart(6, met_ls, mode='zenon', is_bin=True, split_ax=True, ref='t', view='b', p_r_e=rel_prob, backoff=1)
    #     # vis.draw_bar_chart(4, met_ls, mode='zenon', is_bin=True, split_ax=True, ref='t', view='g', p_r_e=rel_prob)
    #     if 0 and args.data == 'FB15k237':
    #         vis.draw_bar_chart(6, met_ls, mode='zenon', is_bin=True, split_ax=True, ref='t', view='b', p_r_e=rel_prob,
    #                            backoff=1, format='svg')
    #         #vis.draw_bar_chart(6, met_ls, mode='zenon', is_bin=False, split_ax=True, ref='t', view='g', p_r_e=rel_prob, backoff=2)
    #     else:
    #         vis.draw_bar_chart(3, met_ls, mode='custom', is_bin=True, split_ax=False, ref='t', view='b',
    #                              p_r_e=rel_prob, custom=[20,40,60])
    #         vis.draw_bar_chart(4, met_ls, mode='custom', is_bin=True, split_ax=False, ref='t', view='g',
    #                            p_r_e=rel_prob, custom=[])
    #
    # # vis.draw_cumulative(20, met_ls, 'even')
    # #vis.draw_ranks(met_ls, 1)
    # ''' Hypers '''
    # # for mode in ['ab']:
    # #    vis.draw__hyperPlane(met_ls, alphas, betas, gammas, mode, )
    # # vis.draw_cube_hyper(met_ls, alphas, betas, gammas)
    # ''' Tables '''
    # if 1:
    #     alphas = [1.0, 0.5, 0.0, -0.5, -1.0]
    #     betas = [0.0,0.2,0.4,0.6,0.8]
    #     # if args.data == 'FB15k237':
    #     #     target_dir = '../figs/excell_tables'
    #     #     os.makedirs(target_dir, exist_ok=True)  # ensure it exists
    #     #
    #     #     for name in os.listdir(target_dir):
    #     #         path = os.path.join(target_dir, name)
    #     #         if os.path.isfile(path) or os.path.islink(path):
    #     #             os.unlink(path)  # remove file or symlink
    #     #         elif os.path.isdir(path):
    #     #             shutil.rmtree(path)  # remove subdirectory
    #
    #     vis.write_tables(met_ls, alphas, [0.0], None, mode='a')
    #     vis.write_tables(met_ls, [1.0], betas, None, mode='b')
    #     # vis.write_tables(met_ls, [1.0], None, gammas, mode='g')

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