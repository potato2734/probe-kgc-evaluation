import argparse
from pathlib import Path
from utils import *
from collections import defaultdict
from vistool import VisTool
from PROBE import PROBE
import numpy as np

def parse_args(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str)
    parser.add_argument('--models', nargs='+', type=str, default=[])

    parser.add_argument('--m8', '--rebm8', action='store_true',
                        help='Write the six-dataset epsilon-scale Spearman correlation table')
    parser.add_argument('--m8-output-dir', type=Path, default=None)
    parser.add_argument('--m8-from-json', type=Path, default=None,
                        help='Render saved per-beta epsilon correlations without rescoring')
    parser.add_argument('--m6', action='store_true')
    parser.add_argument('--m9', action='store_true')
    parser.add_argument('--m14', action='store_true')
    parser.add_argument('--m15c', action='store_true',
                        help='Plot fixed-rank metric differences from candidate size scale 1')
    parser.add_argument('--m15c-output-dir', type=Path, default=None)
    parser.add_argument('--m16', action='store_true')
    parser.add_argument('--m16-dispersion', choices=('variance', 'std'), default='variance',
                        help='Y axis of the M16 beta plot at alpha=1.0 (sample variance or std divided by each model/beta mean)')
    parser.add_argument('--metrics', nargs='+', default=None,
                        help='OWA metric keys, e.g. ZMRR AMRI WMR_BAR')
    parser.add_argument('--probe-run', type=Path, default=None,
                        help='Completed Open-World-KG all10 output root; convert its PROBE histories')
    parser.add_argument('--metric-aggregation', choices=('answer', 'query'), default='answer')
    parser.add_argument('--owa-data-dir', type=Path, default=None)
    parser.add_argument('--owa-output-dir', type=Path, default=None)
    parser.add_argument('--m14-output-dir', type=Path, default=None)

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

def run_owa_analysis(args):
    """OWA histories do not need another dataset's dictionaries or paired PROBE infos."""
    root = Path(__file__).resolve().parents[1]
    suffix = '' if args.metric_aggregation == 'answer' else '_query'
    run_name = args.probe_run.name + suffix if args.probe_run else None
    data_dir = args.owa_data_dir or root / 'data' / 'owa'
    m14_dir = args.m14_output_dir or root / 'figs' / 'm14'
    owa_dir = args.owa_output_dir or root / 'figs' / 'fig12'
    metrics = args.metrics
    if args.probe_run:
        from owa_history import build_owa_histories
        from rank_metrics import METRIC_NAMES
        if args.owa_data_dir is None:
            data_dir = data_dir / run_name
        if args.m14_output_dir is None:
            m14_dir = m14_dir / run_name
        if args.owa_output_dir is None:
            owa_dir = owa_dir / run_name
        build_owa_histories(args.probe_run, data_dir, args.metric_aggregation)
        metrics = metrics or list(METRIC_NAMES)
    if args.m14:
        for metric in metrics or ['PROBE(a=-0.5)']:
            VisTool.reb_m14(data_dir=data_dir, output_dir=m14_dir, metric=metric)
    if args.fig12:
        VisTool.draw_owa(data_dir=data_dir, output_dir=owa_dir, metrics=metrics)

def main(args):
    
    if args.probe_run or args.m14 or args.fig12:
        run_owa_analysis(args)
        kg_flags = ('m8', 'm6', 'm9', 'm15c', 'm16', 'fig2_obs1', 'fig2_obs2', 'fig4', 'fig5', 'fig6_a',
                    'fig6_bcd', 'fig7', 'table4', 'table5', 'fig9', 'fig10_top', 'fig10_bottom', 'fig11')
        if not any(getattr(args, flag) for flag in kg_flags):
            return
    if args.m8:
        VisTool.reb_m8(output_dir=args.m8_output_dir, from_json=args.m8_from_json)
        # M8 loads its own six-dataset inputs, or renders an existing JSON.
        if not any(getattr(args, flag) for flag in (
                'm6', 'm9', 'm15c', 'm16', 'fig2_obs1', 'fig2_obs2', 'fig4', 'fig5',
                'fig6_a', 'fig6_bcd', 'fig7', 'table4', 'table5', 'fig9',
                'fig10_top', 'fig10_bottom', 'fig11')):
            return
    if not args.data or not args.models:
        raise ValueError('Dataset analyses require --data and --models')
    data_path = Path(__file__).resolve().parents[1] / 'data' / args.data

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
    epsilon_entity, epsilon_relation = PROBE.training_epsilons(
        (degree / total_degree for degree in count_info_dict_trn.values()),
        rel_prob,
    )
    print(f'{args.data}: train epsilon_entity={epsilon_entity:.12g}, '
          f'epsilon_relation={epsilon_relation:.12g}')

    info_ls = []

    print(args.models)

    met_ls = [None] * len(args.models)
    for idx, model in enumerate(args.models):
        info_ls = get_infos(model, args.data)
        met_ls[idx] = [PROBE(model=model, data=args.data, info_dump=info, nentity=nentity, nrelation=nrelation, total_degree=total_degree, rel=count_info_rel, w_mode='t',
                            epsilon_entity=epsilon_entity, epsilon_relation=epsilon_relation) for
                            info in info_ls]
    alphas = [1.0, 0.5, 0.0, -0.5, -1.0]
    betas = [0.0, 0.2, 0.4, 0.6, 0.8]

    vis = VisTool(count_info_dict_trn, count_info_dict_tst, trn_triples, tst_triples, nentity, nrelation, total_degree, rel_prob, args.data)
           
    if args.m9:  
        vis.reb_m9(met_ls)
    if args.m15c:
        vis.draw_reb_m15c(met_ls, output_dir=args.m15c_output_dir or '../figs/m15c')
    if args.m16:  
        vis.reb_m16(met_ls, dispersion=args.m16_dispersion)
    if args.m6:  
        vis.reb_m6(Es=[20000, 40000, 80000], ls=[0.1, 0.2, 0.4, 0.8]) 
        
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

if __name__ == '__main__':
    main(parse_args())