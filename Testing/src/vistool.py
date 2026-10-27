import math
import os
import json
from PROBE import  PROBE
from typing import List, Sequence
import numpy as np
from collections import defaultdict
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, MultipleLocator, NullLocator, NullFormatter, ScalarFormatter, PercentFormatter
from matplotlib import gridspec
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from tqdm import tqdm
import networkx as nx
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib import colors
from collections import Counter
from utils import get_ship, get_decoded
from scipy.stats import kendalltau, spearmanr, ttest_ind, rankdata
from rank_metrics import METRIC_NAMES, METRIC_LABELS, CANDIDATE_SWEEP_METRICS, candidate_size_sweep
from matplotlib.patches import Patch
from dataclasses import dataclass
from nltk.corpus import wordnet as wn
import warnings
from functools import lru_cache
import pandas as pd

warnings.filterwarnings(
    "ignore",
    message="No WordNet synset found for pos="
)

OWA_MODEL_COLORS = {
    'RotatE1': '#073B4C', 'RotatE2': '#009999', 'RotatE3': '#6062B8',
    'RotatE4': '#0099FF', 'RotatE5': '#2149C9',
    'ComplEx1': '#996633', 'ComplEx2': '#808000', 'ComplEx3': '#FF9900',
    'ComplEx4': '#A50021', 'ComplEx5': '#CCCC00',
}

def load_owa_histories(data_dir, metrics=None):
    """Read the ten model files; manifests/step sidecars are not model results."""
    plt.rcParams.update({'font.family': 'serif',
                         'font.serif': ['DejaVu Serif', 'Liberation Serif']})
    histories = {}
    for model in OWA_MODEL_COLORS:
        path = os.path.join(data_dir, model + '.json')
        with open(path, encoding='utf-8') as stream:
            histories[model] = json.load(stream)
    metrics = list(metrics) if metrics is not None else list(histories['RotatE1'])
    if not metrics:
        raise ValueError('No OWA metrics requested')
    for model, history in histories.items():
        for metric in metrics:
            if metric not in history:
                raise KeyError('%r is missing from %s/%s.json' % (metric, data_dir, model))
            values = np.asarray(history[metric], dtype=float)
            if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != 2 or not np.isfinite(values).all():
                raise ValueError('Expected nonempty finite full/sparse pairs for %s/%s' % (model, metric))
            history[metric] = values
    return histories, metrics

def owa_training_steps(data_dir, model, size):
    path = os.path.join(data_dir, model + '.steps.json')
    if not os.path.exists(path):
        return np.arange(size), 'Evaluation step'
    with open(path, encoding='utf-8') as stream:
        steps = np.asarray(json.load(stream)['steps'])
    if steps.shape != (size,) or np.any(np.diff(steps) < 0):
        raise ValueError('Invalid training step sidecar: %s' % path)
    return steps, 'Training step (zero-based)'

def owa_metric_label(metric):
    if metric in METRIC_LABELS:
        return METRIC_LABELS[metric]
    if 'PROBE' in metric:
        return metric.replace('a=', r'$\alpha$=')
    if metric in ('MRR', 'log', 'sqrt'):
        return {'MRR': '1.0-MRR', 'log': 'log-MRR', 'sqrt': '0.5-MRR'}[metric]
    if metric.startswith('P') and '_' in metric:
        numerator, denominator = metric[1:].split('_')
        return '%s-MRR' % round(int(numerator) / int(denominator), 2)
    return metric

@dataclass
class Vector:
    v: np.ndarray
    epop: np.ndarray
    rpop: np.ndarray
    color: str

class VisTool():
    def __init__(self, trn_count_info, tst_count_info, trains, tests, nentity, nrelation, total_degree, rel_prob, data):
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Liberation Serif"],
        })
        self.trn_count_info = trn_count_info
        self.tst_count_info = tst_count_info
        self.markers = {'RotatE': 'o', 'ComplEx': 's', 'HousE':'^', 'TuckER':'v', 'pLogicNet':'D',
                        'RNNLogic':'p', 'A*Net':'x'}
        self.mods = {'RotatE':'#073b4c', 'ComplEx':'#118ab2','HousE':'#06d6a0', 'TuckER':'#FFB715',
                      'pLogicNet':'#F78C6B', 'RNNLogic':'#ef476f','A*Net':"#710730"}

        self.trains = trains[:]
        self.tests = tests[:]
        self.nentity = nentity
        self.nrelation = nrelation
        self.total_degree = total_degree
        self.rel_prob = rel_prob
        self.data = data

        self.edges = defaultdict(set)
        for q in self.trains:
            h, r, t = q
            self.edges[h].add((r, t, 'h'))
            self.edges[t].add((r, h, 't'))

    def get_model_colors(self, mets: List[List[PROBE]]):
        model_names = [_mets[0].model for _mets in mets]
        colors = [self.mods[model_name] for model_name in model_names]
        return colors

    def draw__hyperPlane(self, mets: List[List[PROBE]], alphas: list, betas: list, gammas=None, mode='ab',
                        vis_axis=False, vis_cbar=False):
        if mode != 'ab':
            raise ValueError('PROBE has two parameters; only alpha/beta contours are supported')
        target1, target2 = alphas, betas

        data = defaultdict(list)
        model_name_ls = []
        num_models = len(mets)
        axis_font = 13

        for a_i, alpha in enumerate(target1):
            for b_i, beta in enumerate(target2):
                for _met in mets:
                    values = []
                    if len(model_name_ls) < num_models:
                        model_name_ls.append(_met[0].model)

                    for met in _met:
                        values.append(met.calculate_final_metric(alpha, beta))

                    data[(a_i, b_i)].append(sum(values) / len(values))

        for xy, z_ls in data.items():
            z_ls = np.array(z_ls, dtype=np.float64)
            denom = (z_ls.max() - z_ls.min())
            data[xy] = (z_ls - z_ls.min()) / (denom + 1e-12)

        n1, n2 = len(target1), len(target2)
        Zs = np.full((num_models, n2, n1), np.nan, dtype=np.float64)
        for (xi, yi), z_vals in data.items():
            for m_idx, z in enumerate(z_vals):
                Zs[m_idx, xi, yi] = z

        X_idx, Y_idx = np.meshgrid(np.arange(n1), np.arange(n2))

        for m_idx in range(num_models):
            if np.isnan(Zs[m_idx]).any():
                missing = np.argwhere(np.isnan(Zs[m_idx]))
                raise ValueError(
                    f"Missing grid points for model {model_name_ls[m_idx]} at indices (target1_idx, target2_idx): "
                    f"{missing.tolist()[:10]}{' ...' if len(missing) > 10 else ''}"
                )

        zmin = float(np.nanmin(Zs))
        zmax = float(np.nanmax(Zs))
        levels = np.linspace(zmin, zmax, 256)

        plot_models = list(enumerate(model_name_ls))
        if self.data == 'YAGO3-10' and 'RNNLogic' not in model_name_ls:
            # Keep the missing RNNLogic panel separate from actual model results.
            plot_models.insert(5, (None, 'RNNLogic'))

        mag = 1.3
        cmap_scheme = 'RdYlBu_r'
        for panel_idx, (m_idx, model_name) in enumerate(plot_models):
            fig, ax2 = plt.subplots(figsize=(mag, mag), constrained_layout=True)
            if m_idx is None:
                ax2.set_facecolor("white")
                ax2.set_xticks([])
                ax2.set_yticks([])
            else:
                Z = Zs[m_idx]
                csf = ax2.contourf(
                    Y_idx, X_idx, Z,
                    levels=levels,
                    cmap=cmap_scheme,
                    antialiased=True
                )
                ax2.contour(Y_idx, X_idx, Z,
                            levels=levels,
                            cmap=cmap_scheme,
                            linewidths=0.6)

            x_mid = sum(ax2.get_xlim()) / 2
            y_mid = sum(ax2.get_ylim()) / 2

            if m_idx is not None:
                ax2.axvline(x_mid, color='grey', lw=1, ls='--')
                ax2.axhline(y_mid, color='grey', lw=1, ls='--')

            if m_idx is None:
                ax2.text(
                    0.5, 0.5,
                    "O.O.M",
                    ha='center',
                    va='center',
                    fontsize=15,
                    transform=plt.gca().transAxes
                )

            if panel_idx == 0:
                origin = (0.9, 0.1)

                ax2.annotate(
                    '',
                    xy=(0.7, 0.1), 
                    xytext=origin,
                    arrowprops=dict(arrowstyle='-|>', lw=1, fc='black'),
                    xycoords='axes fraction'
                )

                ax2.annotate(
                    '',
                    xy=(0.9, 0.3),
                    xytext=origin,
                    arrowprops=dict(arrowstyle='-|>', lw=1, fc='black'),
                    xycoords='axes fraction'
                )

                ax2.text(0.69, 0.1, r'$\alpha$', transform=ax2.transAxes, ha='right', va='center')
                ax2.text(0.9, 0.31, r'$\beta$', transform=ax2.transAxes, ha='center', va='bottom')

            ax2.tick_params(axis='both', which='both', length=0)
            ax2.set_xticks(np.arange(n1))
            ax2.set_yticks(np.arange(n2))
 
            if vis_axis:
                if m_idx is not None:
                    ax2.set_yticklabels([f"↑" if v in {target1[len(target1) // 2]} else "" for v in target1], fontsize=axis_font)
                else:
                    ax2.set_yticklabels([])
                ax2.set_xticklabels([f"→" if v in {target2[len(target2) // 2]} else "" for v in target2], fontsize=axis_font)
            else:
                ax2.set_xticks([])
                ax2.set_yticks([])

            ax2.set_xlim(0, n1 - 1)
            ax2.set_ylim(0, n2 - 1)
            ax2.set_aspect("equal", adjustable="box")
            os.makedirs(f'../figs/fig11', exist_ok=True)
            file_model_name = 'AStarNet' if model_name == 'A*Net' else model_name
            save_path = f'../figs/fig11/{self.data}_{file_model_name}_contour_ab.pdf'
            fig.savefig(save_path, bbox_inches='tight', pad_inches=0.01, dpi=300)
            plt.close(fig)

    def draw_3d_weights(self, met: PROBE, betas: list):
        axis_font = 12
        cmap = cm.inferno 
        wfunc = met.w_entity_function
        min_r_e = min_e = epsilon = 1e-3
        max_e = min_e * 50

        num_axis_points = 20

        for beta in betas:
            e = np.array([i / num_axis_points for i in range(1, num_axis_points + 1)])
            r_e = np.array([i / num_axis_points for i in range(1, num_axis_points + 1)])

            X, Y = np.meshgrid(e, r_e, indexing='ij')

            Wx = wfunc(e, beta, epsilon) 
            Wy = wfunc(r_e, beta, epsilon) 

            Z = np.outer(Wx, Wy) 

            norm = colors.LogNorm(vmin=Z.min(), vmax=Z.max())
            facecolors = cmap(norm(Z))

            fig = plt.figure(figsize=(3, 3))
            ax = fig.add_subplot(111, projection='3d')

            ax.plot_surface(
                X, Y, Z,
                facecolors=facecolors,
                linewidth=0,
                antialiased=True,
                shade=False  
            )

            labelpad = -13
            ax.set_xlabel(r'$\delta_e$ (→)', fontsize=axis_font, labelpad=labelpad)
            ax.set_ylabel(r'(←) $\delta_{r|e}$', fontsize=axis_font, labelpad=labelpad)
            ax.set_zlabel(r'$w_t$ (→)', fontsize=axis_font, labelpad=labelpad - 1)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])

            ax.invert_yaxis()

            os.makedirs(f'../figs/fig6/', exist_ok=True)
            plt.savefig(f'../figs/fig6/w_hyper_{round(beta, 1)}.pdf', bbox_inches='tight', pad_inches=0.01, dpi=300)
            plt.close(fig)
        fig, ax = plt.subplots(figsize=(1, 1.5))
        ax.remove()
        cax = fig.add_axes([0.45, 0.05, 0.15, 0.9])

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        cbar = fig.colorbar(sm, cax=cax)

        cbar.ax.yaxis.set_major_locator(NullLocator())
        cbar.ax.yaxis.set_minor_locator(NullLocator())
        cbar.ax.yaxis.set_major_formatter(NullFormatter())
        cbar.ax.yaxis.set_minor_formatter(NullFormatter())
        cbar.ax.tick_params(which='both', length=0, labelleft=False, labelright=False)

        cbar.ax.text(0.5, 1.02, 'High', ha='center', va='bottom',
                     transform=cbar.ax.transAxes, fontsize=axis_font)
        cbar.ax.text(0.5, -0.02, 'Low', ha='center', va='top',
                     transform=cbar.ax.transAxes, fontsize=axis_font)

        plt.savefig('../figs/fig6/w_hyper_cbar.pdf',
                    bbox_inches='tight', pad_inches=0.01, dpi=300)
        plt.close(fig)

    def draw_multiple_unit_betas(self, met: PROBE, betas):
        fig = plt.figure(figsize=(4,3))
        cmap = plt.cm.Set1

        axis_font = 18
        epsilon = 1e-3
        num_points = 100
        x = np.array([i / num_points for i in range(num_points + 1)])
        for idx, beta in enumerate(sorted(betas, reverse=True)):
            color = cmap(idx)
            y = met.w_entity_function(x, beta, epsilon)
            plt.plot(x, y, color=color, label=rf'$\beta$={beta}', lw=2)
        plt.legend(fontsize=axis_font - 6)
        plt.tick_params(axis='both', labelsize=axis_font - 3)
        plt.yscale('log')
        plt.xlabel('Popularity', fontsize=axis_font)
        plt.ylabel('Weights', fontsize=axis_font)
        img_name = f'Unit_weights'
        plt.grid(True, which='both', linestyle='-', linewidth=0.3, color='grey', alpha=0.5)
        os.makedirs(f'../figs/fig6', exist_ok=True)
        fig.savefig(f'../figs/fig6/{img_name}.pdf', bbox_inches='tight', pad_inches=0.01)

    def draw_multiple_unit_alphas(self, alphas, ver='f'):
        fig = plt.figure(figsize=(4, 3))
        axis_font = 18
        E = 8000

        x = np.arange(1, E)  
        inv_x = 1.0 / x

        for idx, a in enumerate(alphas):
            if ver == 'f*':
                if abs(a) < 1e-9:
                    y = 1 - np.log(x) / np.log(E)
                else:
                    N = (1.0 / E) ** a 
                    y = (inv_x ** a - N) / (1 - N)
            else:
                y = inv_x ** a

            plt.plot(x, y, color=f'C{idx}', label=rf'$\alpha$={a}')

        plt.legend(fontsize=axis_font - 4)
        plt.tick_params(axis='both', labelsize=axis_font - 3)
        plt.xlabel('Rank Value', fontsize=axis_font)
        plt.xticks([0, 2500, 5000, 7500], ['0k', '2.5k', '5k', '7.5k'])
        plt.ylabel('Transformed Score', fontsize=axis_font)
        plt.grid(True, which='both', linestyle='-', linewidth=0.5, color='gray', alpha=0.7)

        img_name = 'Affine RT' if ver == 'f*' else 'Original RT'
        os.makedirs('../figs/fig4', exist_ok=True)
        fig.savefig(f'../figs/fig4/{img_name}.pdf', bbox_inches='tight', pad_inches=0.01)

    def draw_two_w(self, max_lines=50000, plot=True, bar=True):
        axis_font = 8
        gamma = {
            'FB15k237': 0.3,
            'wn18rr': 0.3,
            'YAGO3-10': 0.2,
            'family': 1,
            'umls': 1,
            'kinship': 1,
        }[self.data] 
        gamma = 1.0

        norm = colors.Normalize(vmin=0, vmax=1.1)
        cmap = plt.cm.binary_r

        fig, ax = plt.subplots(figsize=(3, 1.2))

        e_color = '#3D65A1'
        r_color = '#689236'

        plot_gap = 0.10
        e_line = plot_gap
        r_line = 1.0 - plot_gap
        hist_height = 0.3

        e_occ_list = []
        r_occ_list = []

        unique_entities = set()
        unique_er_pairs = set()

        for h, r, t in self.trains:
            e_occ_list.extend([
                self.trn_count_info[h] / self.total_degree,
                self.trn_count_info[t] / self.total_degree,
            ])
            r_occ_list.extend([
                self.rel_prob[h][r],
                self.rel_prob[t][r],
            ])

            unique_entities.add(h)
            unique_entities.add(t)

            unique_er_pairs.add((h, r))
            unique_er_pairs.add((t, r))

        e_occ = np.asarray(e_occ_list, dtype=np.float64)
        r_occ = np.asarray(r_occ_list, dtype=np.float64)

        e_min, e_max = e_occ.min(), e_occ.max()
        r_min, r_max = r_occ.min(), r_occ.max()

        def map_entity_pop(v):
            v = np.asarray(v, dtype=np.float64)
            e_pop = (v - e_min) / (e_max - e_min + 1e-12)
            x = 1.0 - e_pop
            x = 1.0 - (1.0 - x) ** gamma
            return x

        def map_rel_pop(v):
            v = np.asarray(v, dtype=np.float64)
            r_pop = (v - r_min) / (r_max - r_min + 1e-12)
            x = 1.0 - r_pop
            x = 1.0 - (1.0 - x) ** 1.0
            return x

        if plot:
            if len(e_occ) > max_lines:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(e_occ), size=max_lines, replace=False)
                e_occ_plot = e_occ[idx]
                r_occ_plot = r_occ[idx]
            else:
                e_occ_plot = e_occ
                r_occ_plot = r_occ

            x0_line = map_entity_pop(e_occ_plot)
            x1_line = map_rel_pop(r_occ_plot)

            segments = np.stack([
                np.stack([x0_line, np.full_like(x0_line, e_line)], axis=1),
                np.stack([x1_line, np.full_like(x1_line, r_line)], axis=1)
            ], axis=1)

            lc = LineCollection(
                segments,
                cmap=cmap,
                norm=norm,
                linewidths=0.1,
                alpha=0.3
            )
            lc.set_array(x0_line)
            ax.add_collection(lc)

        if bar:
            e_hist_vals = np.asarray(
                [self.trn_count_info[e] / self.total_degree for e in unique_entities],
                dtype=np.float64
            )

            r_hist_vals = np.asarray(
                [self.rel_prob[e][r] for (e, r) in unique_er_pairs],
                dtype=np.float64
            )

            x0_hist = map_entity_pop(e_hist_vals)
            x1_hist = map_rel_pop(r_hist_vals)

            bins = 100
            hist0, edges0 = np.histogram(x0_hist, bins=bins, range=(0.0, 1.0), density=True)
            hist1, edges1 = np.histogram(x1_hist, bins=bins, range=(0.0, 1.0), density=True)

            centers0 = 0.5 * (edges0[:-1] + edges0[1:])
            centers1 = 0.5 * (edges1[:-1] + edges1[1:])
            width0 = edges0[1] - edges0[0]
            width1 = edges1[1] - edges1[0]

            hist0_draw = hist0.astype(np.float64)
            hist1_draw = hist1.astype(np.float64)

            if hist0_draw.max() > 0:
                hist0_draw /= hist0_draw.max()
            if hist1_draw.max() > 0:
                hist1_draw /= hist1_draw.max()

            bump = 0.00
            ax.bar(
                centers0,
                -hist0_draw * hist_height -bump,
                width=width0,
                bottom=e_line,
                align='center',
                linewidth=0,
                alpha=0.8,
                color=e_color,
                antialiased=False
            )

            ax.bar(
                centers1,
                hist1_draw * hist_height + bump,
                width=width1,
                bottom=r_line,
                align='center',
                linewidth=0,
                alpha=0.8,
                color=r_color,
                antialiased=False
            )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.axhline(y=e_line, xmin=0, xmax=1, color='black', lw=0.1)
        ax.axhline(y=r_line, xmin=0, xmax=1, color='black', lw=0.1)

        bar_gap = 0.02
        txt_gap = 0.02

        ax.text(
            0.5, r_line+bar_gap+txt_gap, r"Normalized $\delta(r|e)$",
            ha='center', va='bottom',
            transform=ax.get_xaxis_transform(), fontsize=axis_font
        )

        ax.text(
            0.5, -e_line - bar_gap - txt_gap + 0.02, r"Normalized $\delta(e)$",
            ha='center', va='bottom',
            transform=ax.get_xaxis_transform(), fontsize=axis_font
        )

        ax.axis('off')

        os.makedirs(f'../figs/fig5/', exist_ok=True)
        plt.savefig(
            f'../figs/fig5/{self.data}_e_r_corr.pdf',
            dpi=300,
            bbox_inches='tight',
            pad_inches=0.01
        )
        plt.close(fig)

    def draw_bar_chart(self,
                       chunk,
                       mets: List[List[PROBE]],
                       mode,
                       is_bin=False,
                       split_ax=False,
                       ref='t',
                       view='b',
                       p_r_e=None,
                       backoff=0,
                       format='pdf',
                       custom=None
                       ):
        dataset = mets[0][0].data
        alpha, beta = {
            'FB15k237': (1, 0),
            'wn18rr': (1, 0),
            'YAGO3-10': (1, 0),
            'family': (1, 0),
            'umls': (1, 0),
            'kinship': (1, 0),
        }[dataset]

        if not custom:
            if view == 'b':
                custom = {
                'FB15k237': [25, 50, 75],
                'wn18rr': [30, 60, 90],
                'YAGO3-10': [30, 60, 90],
                'family': [25, 50, 75],
                'umls': [30, 60, 90],
                'kinship': [25, 50, 75],
                }[dataset]
            elif view == 'g':
                custom = {
                'FB15k237': [25, 50, 75],
                'wn18rr': [30, 60, 90],
                'YAGO3-10': [30, 60, 90],
                'family': [25, 50, 75],
                'umls': [30, 60, 90],
                'kinship': [25, 50, 75],
                }[dataset]

        def in_bin(cnt, bin_idx):
            if bin_idx == 0:
                return cnt > walls[0] if walls else True
            elif bin_idx == len(walls):
                return cnt <= walls[-1]
            else:
                return (cnt > walls[bin_idx]) and (cnt <= walls[bin_idx - 1])

        if view == 'b':
            trn_cnts = list(self.trn_count_info.values())
            trn_len = len(trn_cnts)
            walls = []
            total_degree = mets[0][0].total_degree
            if custom:
                for c in custom:
                    walls.append(trn_cnts[int((c / 100) * trn_len)] / total_degree)
                chunk = len(custom) + 1
            else:
                for i in range(backoff + 1, chunk):
                    if mode == 'even':
                        walls.append(trn_cnts[(trn_len // chunk) * i] / total_degree)
                    elif mode == 'zenon':
                        walls.append(trn_cnts[int(trn_len * (0.5) ** i)] / total_degree)
            walls.sort(reverse=True) 

            post_chunk = len(walls) + 1
            percentages = [np.count_nonzero(np.array([in_bin(cnt / total_degree, i) for cnt in trn_cnts])) / trn_len for
                           i in range(post_chunk)]
            chunk_range = [-1]
            for i in range(post_chunk):
                print(sum(percentages[:i + 1]))
                chunk_range.append(int(sum(percentages[:i + 1]) * 100))
            chunk_range.append(100)
        elif view == 'g':
            walls = []
            p_r_e_ls = []
            for ls in p_r_e.values():
                temp_ls = [el for el in ls if el]
                p_r_e_ls += temp_ls
            p_r_e_ls.sort(reverse=True)
            pre_len = len(p_r_e_ls)
            if custom:
                for c in custom:
                    walls.append(p_r_e_ls[int((c / 100) * pre_len)])
                chunk = len(custom) + 1
            else:
                for i in range(1, chunk):
                    if mode == 'even':
                        walls.append(p_r_e_ls[(pre_len // chunk) * i])
                    elif mode == 'zenon':
                        walls.append(p_r_e_ls[int(pre_len * (0.5) ** i)])
            walls.sort(reverse=True)
            print(walls)
            print([a for a in walls])

            percentages = [np.count_nonzero(np.array([in_bin(cnt, i) for cnt in p_r_e_ls])) / pre_len for
                           i in range(chunk)]
            chunk_range = [-1]
            for i in range(chunk):
                print(sum(percentages[:i + 1]))
                chunk_range.append(int(sum(percentages[:i + 1]) * 100))
            chunk_range.append(100)
            post_chunk = chunk

        def get_reference(met : PROBE):
            if view == 'b' and ref == 't': return met.t_p_e
            if view == 'b' and ref == 'p': return met.p_e
            if view == 'g' and ref == 't': return met.t_p_r_e
            if view == 'g' and ref == 'p': return met.p_r_e

        # Example: chunk=4 → walls=[w0,w1,w2] with w0>=w1>=w2
        # 0~12.5%/12.5~25%/25~50%/50~100%

        models = []
        num_models = len(mets)
        model_metric_chunks = [0 for _ in range(num_models)]
        colors = self.get_model_colors(mets)
        markers = [self.markers[mets[i][0].model] for i in range(len(mets))]
        final_metrics = [[] for _ in range(num_models)]

        bin_samples = [[[] for _ in range(post_chunk)] for __ in range(num_models)]

        final_results = [None for __ in range(num_models)]
        min_y, max_y = 1, 0
        err_results = [[0.0 for _ in range(post_chunk)] for __ in range(num_models)]
        for i, _met in enumerate(mets):
            models.append(_met[0].model)
            temp_results = [[0 for _ in range(post_chunk)] for __ in range(len(_met))]
            for j, met in enumerate(_met): 
                score = met.calculate_final_metric(alpha, beta)
                for bin_idx in range(post_chunk):
                    mask = np.array([in_bin(p, bin_idx)
                                     for p in get_reference(met)], dtype=bool)
                    list_per_entity = np.array(met.transformed_ranks, dtype=object)
                    filtered_list = list_per_entity[mask]  

                    per_entity_means = float(np.mean(filtered_list))
                    bin_samples[i][bin_idx] = per_entity_means
                    min_y = min(min_y, per_entity_means)
                    max_y = max(max_y, per_entity_means)
                    size = len(filtered_list)
                    agg = sum(filtered_list)
                    temp_results[j][bin_idx] = (agg / size) if size > 0 else np.nan

                final_metrics[i].append(score)

            final_results[i] = [
                np.nanmean([temp_results[_k][k] for _k in range(len(temp_results))])
                for k in range(post_chunk)
            ]
            for bin_idx in range(post_chunk):
                s = np.array([temp_results[seed][bin_idx] for seed in range(len(temp_results))], dtype=float)
                err_results[i][bin_idx] = float(np.nanstd(s, ddof=1)) if s.size >= 2 else 0.0

        x = np.arange(post_chunk)
        bar_width = 0.8 / num_models
        avg_final_metrics = [sum(ls) / len(ls) for ls in final_metrics]
        data = np.array(final_results).T
        max_val = float(np.nanmax(data))
        min_val = float(np.nanmin(data))

        axis_font = 15

        hatch_dict = {
            'RotatE': '///',
            'ComplEx': '+',
            'HousE': 'O',
            'TuckER': 'x',
            'pLogicNet': '|',
            'RNNLogic': '...',
        }

        def _plot_on_axis(ax):
            for i in range(num_models):
                if is_bin:
                    ax.bar(
                        x + i * bar_width,
                        [final_results[i][bar_idx] for bar_idx in range(post_chunk)],
                        width=bar_width, label=models[i],
                        yerr=[err_results[i][bar_idx] for bar_idx in range(chunk)],
                        capsize=3,
                        color=to_rgba(colors[i], alpha=1), edgecolor=colors[i],
                        linewidth=2, zorder=0,
                    )
                else:
                    ax.scatter(
                        x, [final_results[i][bar_idx] for bar_idx in range(post_chunk)],
                        label=models[i], color=colors[i],
                        linewidth=1, marker=markers[i], s=100, zorder=3
                    )
                    ax.errorbar(
                        x,
                        [final_results[i][bar_idx] for bar_idx in range(chunk)],
                        yerr=[err_results[i][bar_idx] for bar_idx in range(chunk)],
                        fmt='none',
                        ecolor='black',
                        elinewidth=1,
                        capsize=3,
                        zorder=4
                    )

        if split_ax:
            fig = plt.figure(figsize=(5.8, 2.5))
            gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2], hspace=0.05, figure=fig)

            ax_high = fig.add_subplot(gs[0])
            ax_low = fig.add_subplot(gs[1], sharex=ax_high)

            def find_cut(arr):
                cut_1, cut_2 = 0, 1
                for i in range(len(arr)):
                    cut_1 = max(arr[i][1], cut_1)
                    cut_2 = min(arr[i][0], cut_2)
                return cut_1 + 0.02, cut_2 - 0.02

            cut_1, cut_2 = find_cut(final_results)
            ax_low.set_ylim(min_y - 0.015, cut_1)  
            ax_high.set_ylim(cut_2, max_y + 0.01)
            ax_high.axhline(cut_2, color='black', linewidth=2)
            ax_low.axhline(cut_1, color='black', linewidth=2)

            _plot_on_axis(ax_low)
            _plot_on_axis(ax_high)
            for ax in (ax_low, ax_high):
                ax.tick_params('y', labelsize=axis_font)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.tick_params(axis="y", length=0)
                ax.tick_params(axis="x", length=0)
            ax_high.spines['bottom'].set_visible(False)

            from matplotlib.ticker import MultipleLocator

            for ax in (ax_low, ax_high):
                ax.yaxis.set_major_locator(MultipleLocator(0.1))

            ax_low.yaxis.grid(True, color="grey", linestyle="-", linewidth=0.5, alpha=0.5, zorder=0)
            ax_high.yaxis.grid(True, color="grey", linestyle="-", linewidth=0.5, alpha=0.5, zorder=0)

            ax_high.tick_params(labelbottom=False)  
            ax_low.set_xlabel('Popularity (%)', fontsize=axis_font, labelpad=0)
            ax_low.set_xticks(x + (bar_width * (num_models - 1) / 2 if is_bin else 0))
            ax_low.set_xticklabels([f'{chunk_range[i] + 1}~{chunk_range[i + 1]}%' for i in range(post_chunk)],
                                   fontsize=axis_font)

            ax_low.set_ylabel('MRR', fontsize=axis_font, labelpad=0)
            ax_low.yaxis.set_label_coords(-0.1, 0.7)

        else:
            fig, ax = plt.subplots(figsize=(5, 2.5))
            ax.tick_params('x', labelsize=axis_font)
            ax.tick_params('y', labelsize=axis_font)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.tick_params(axis="y", length=0)
            ax.tick_params(axis="x", length=0)
            _plot_on_axis(ax)

            ax.yaxis.grid(True, color="grey", linestyle="-", linewidth=0.5, alpha=0.5, zorder=0)
            ax.yaxis.grid(True, color="grey", linestyle="-", linewidth=0.5, alpha=0.5, zorder=0)

            ax.set_xticks(x + (bar_width * (num_models - 1) / 2 if is_bin else 0))
            ax.set_xticklabels([f'{chunk_range[i] + 1}~{chunk_range[i + 1]}%' for i in range(post_chunk)],
                               fontsize=axis_font)

            ax.set_ylabel('MRR', fontsize=axis_font, labelpad=2)
            ax.tick_params(axis='x', pad=2)

        model_strs = '_'.join(models)
        os.makedirs(f'../figs/fig2_obs2/{mode}', exist_ok=True)
        fig.savefig(f'../figs/fig2_obs2/{mode}/{dataset}_{mode}bar_{ref}_{view}_{model_strs}.{format}', bbox_inches='tight', pad_inches=0.01)

    def _stars(self, p):
        if p < 1e-3: return "***"
        if p < 1e-2: return "**"
        if p < 5e-2: return "*"
        return f"n.s."

    def draw_for_table(self, mets: List[List[PROBE]], alpha, beta, gamma=None, target='a', figw=2, figh=1.5):
        data = mets[0][0].data

        if target not in ('a', 'b'):
            raise ValueError('Table target must be alpha (a) or beta (b)')
        ret_dict = {'a': alpha, 'b': beta}
        name_dict = {'a': 'alpha', 'b': 'beta'}

        colors = {_mets[0].model: self.mods[_mets[0].model] for _mets in mets}

        model_results = {}
        for _mets in mets:
            model_name = _mets[0].model
            scores = [met.calculate_final_metric(alpha, beta) for met in _mets]
            model_results[model_name] = scores

        model_names = list(model_results.keys())
        means = {m: float(np.mean(model_results[m])) for m in model_names}
        stds = {m: float(np.std(model_results[m], ddof=1)) if len(model_results[m]) > 1 else 0.0
                for m in model_names}

        order = sorted(model_names, key=lambda m: means[m], reverse=True)

        upperbound, lower_bound = max([*means.values()]), min([*means.values()])

        x = np.arange(len(order))
        y = np.array([means[m] for m in order])
        yerr = np.array([stds[m] for m in order])
        bar_colors = [colors[m] for m in order]
        max_yerr = max(yerr)
        plt.xticks([])

        y_max = float(max(y + yerr)) if len(y) else 1.0
        y_min = float(min(y - yerr)) if len(y) else 0.0
        base_offset = 0.03 * y_max
        line_height = 0.01
        upperbound, lower_bound = y_max * 1.05, y_min * 0.95

        for i in range(len(order) - 1):
            m1, m2 = order[i], order[i + 1]
            a, b = model_results[m1], model_results[m2]

            _, p = ttest_ind(a, b, equal_var=False)
            label = self._stars(p)

            axis_font = 16
            part = 0.35 
            xs = [part, 1 - part]
            fig1, ax1 = plt.subplots(figsize=(figw, figh), constrained_layout=True)
            ax1.set_facecolor("#E9E9F1")
            name_m0, name_m1 = model_names
            mean_m0, mean_m1 = np.mean(model_results[name_m0]), np.mean(model_results[name_m1])
            std_m0, std_m1 = stds[name_m0], stds[name_m1]

            ax1.errorbar(xs[0], mean_m0, yerr=std_m0, color=colors[name_m0], capsize=5, fmt=self.markers[name_m0],
                         markersize=12, elinewidth=2, capthick=2, label=name_m0)
            ax1.errorbar(xs[1], mean_m1, yerr=std_m1, color=colors[name_m1], capsize=5, fmt=self.markers[name_m1],
                         markersize=12, elinewidth=2, capthick=2, label=name_m1)

            ax1.tick_params(axis='y', direction='in', pad=0)
            for lbl in ax1.get_yticklabels():
                lbl.set_horizontalalignment('left') 
                lbl.set_verticalalignment('center')

            ax1.set_xticks(xs)
            ax1.set_xticklabels([])
            ax1.set_xlim(0, 1)
            ax1.set_title(fr'[$\{name_dict[target]}$ = {ret_dict[target]}]{label}', fontsize=axis_font)
            diff = abs(mean_m0 - mean_m1)
            if diff < 0.04: unit_tick, ntick = 0.01, 100
            elif diff < 0.08: unit_tick, ntick = 0.02, 50
            else: unit_tick, ntick = 0.05, 20
            ax1.set_yticks([round(unit_tick * ii, 2) for ii in range(ntick)])
            ax1.set_yticklabels([str(round(unit_tick * ii, 2)) for ii in range(ntick)], color='grey', fontsize=axis_font - 2)

            def adjust_ylim(ax, margin=0.001):
                ymin, ymax = ax.get_ylim()
                ticks = np.asarray(ax.get_yticks())

                above = ticks[ticks > ymin]
                if above.size > 0:
                    nearest_above = above.min()
                    lower_gap = nearest_above - ymin
                    if lower_gap < margin:
                        ymin = nearest_above + 0.001

                below = ticks[ticks < ymax]
                if below.size > 0:
                    nearest_below = below.max()
                    upper_gap = ymax - nearest_below
                    if upper_gap < margin:
                        ymax = nearest_below - 0.001

                ax.set_ylim(ymin, ymax)

            ax1.set_ylim(y_min - 0.007, min(1, y_max + 0.01)) 
            adjust_ylim(ax1, margin=0.003)

            for spine in ["top", "right", "left", "bottom"]:
                ax1.spines[spine].set_visible(False)
            ax1.tick_params(axis='x', length=0)
            ax1.tick_params(axis='y', length=0)

            ax1.grid(True, axis="both", color="white", linewidth=1.0)
            ax1.set_axisbelow(True)
      
            title = f'../figs/twoOfUs/{target}/{data}_twoOfUs_{model_names}_{ret_dict[target]}({target}).pdf'
            os.makedirs(f'../figs/twoOfUs/{target}', exist_ok=True)
            fig1.savefig(title, bbox_inches='tight', pad_inches=0.001)

            print(f'{data} : {model_names}; {label} | {name_m0 if mean_m0 > mean_m1 else name_m1}')

    def write_tables(self, mets: List[List[PROBE]], alphas, betas, gammas=None, mode='a'):
        """Update a dataset's results and rebuild a complete LaTeX table.

        Modes a/b write table4/table5.tex under ../figs/table4/table5.
        A JSON sidecar keeps results across dataset runs; delete it to start over.
        The LaTeX document needs booktabs, multirow, xcolor (table), the
        rank1, rank2, ... colors, and the \\method macro used in the caption.
        """
        if mode not in ('a', 'b'):
            raise ValueError(f'Not supported mode name: {mode}')
        if not mets or any(not runs for runs in mets):
            raise ValueError('Each model must have at least one metric run.')

        data = mets[0][0].data

        match mode:
            case 'a':
                piv_ls, other = betas, alphas
            case 'b':
                piv_ls, other = alphas, betas

        if not len(piv_ls) or not len(other):
            raise ValueError('Table parameter lists must not be empty.')

        latex = defaultdict(str)

        for piv in piv_ls:
            results = [[] for _ in range(len(other))]

            all_mean_y = []

            for _mets in mets:
                calculated_metrics = []

                for met in _mets:
                    if mode == 'a':
                        values = [
                            met.calculate_final_metric(alpha, piv)
                            for alpha in alphas
                        ]
                    elif mode == 'b':
                        values = [
                            met.calculate_final_metric(piv, beta)
                            for beta in betas
                        ]

                    calculated_metrics.append(values)

                calculated_metrics = np.array(calculated_metrics)

                mean_y = np.mean(calculated_metrics, axis=0)
                std_y = np.std(calculated_metrics, axis=0)

                all_mean_y.append((mean_y, std_y))

            for idx, (mean_y, std_y) in enumerate(all_mean_y):
                for i in range(len(other)):
                    results[i].append(
                        (
                            mets[idx][0].model,
                            round(mean_y[i], 4),
                            round(std_y[i], 4)
                        )
                    )

            for i, ls in enumerate(results):
                ls = sorted(ls, key=lambda x: x[1], reverse=True)

                for rank, (model, score, std) in enumerate(ls):
                    rank += 1

                    color_style = f'\\cellcolor{{rank{rank}}}'

                    latex[model] += (
                        f' & {color_style}'
                        f'{{{score * 100:.2f}{{\\tiny\color{{gray}}$\\pm${std * 100:.2f}}}}}'
                    )

        folder_name = 'table4' if mode == 'a' else 'table5'
        folder = f'../figs/{folder_name}'
        os.makedirs(folder, exist_ok=True)
        cache_path = f'{folder}/{folder_name}.json'
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                tables = json.load(f)
        else:
            tables = {}

        parameter = 'alpha' if mode == 'a' else 'beta'
        pivot_parameter = 'beta' if mode == 'a' else 'alpha'
        columns = []
        for piv in piv_ls:
            for value in other:
                label = fr'$\{parameter}$={value:.1f}'
                if len(piv_ls) > 1:
                    label += fr', $\{pivot_parameter}$={piv:.1f}'
                columns.append(label)
        tables[data] = {
            'columns': columns, 'rows': dict(latex),
            'epsilon_reference': 'train',
            'epsilon_entity': mets[0][0].epsilon_entity,
            'epsilon_relation': mets[0][0].epsilon_relation,
        }
        with open(cache_path, 'w') as f:
            json.dump(tables, f, indent=2)

        dataset_labels = {
            'FB15k237': 'FB15k-237', 'wn18rr': 'WN18RR',
            'YAGO3-10': 'YAGO3-10', 'family': 'Family',
            'umls': 'UMLS', 'kinship': 'Kinship',
        }
        datasets = [name for name in dataset_labels if name in tables]
        datasets.extend(name for name in tables if name not in dataset_labels)
        # Use the established model order, then any additional models.
        table_models = [name for name in self.mods
                        if any(name in table['rows'] for table in tables.values())]
        for table in tables.values():
            for model in table['rows']:
                if model not in table_models:
                    table_models.append(model)

        def escape_label(value):
            replacements = {
                '\\': r'\textbackslash{}', '&': r'\&', '%': r'\%',
                '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{',
                '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}',
            }
            return ''.join(replacements.get(char, char) for char in value)

        aspect = 'rank sensitivity' if mode == 'a' else 'popularity-bias robustness'
        lines = [
            r'\begin{table}[ht]', r'\centering', r'\footnotesize',
            fr'\caption{{The accuracy of KGC Models evaluated by {{\method}} '
            fr'with varying levels of {aspect} $\{parameter}$. '
            r'The \textcolor[HTML]{F8A602}{gold}, '
            r'\textcolor[HTML]{A5A5B1}{silver}, and '
            r'\textcolor[HTML]{D36A49}{bronze} indicate the best, '
            r'the second, and the third best results, respectively.}',
            r'\vspace{-3mm}', r'\setlength{\tabcolsep}{0.3pt}',
        ]
        for start in range(0, len(datasets), 3):
            group = datasets[start:start + 3]
            widths = [len(tables[name]['columns']) for name in group]
            spec = 'l ' + ' | '.join('c' * width for width in widths)
            lines.append('\\resizebox{\\textwidth}{!}{')
            lines.append(fr'\begin{{tabular}}{{{spec}}}')
            if start == 0:
                lines.extend([r'\toprule', r'\toprule'])
            headers = [r'\multirow{2}{*}{Models}']
            headers.extend(
                fr'\multicolumn{{{width}}}{{c}}{{{escape_label(dataset_labels.get(name, name))}}}'
                for name, width in zip(group, widths)
            )
            lines.append(' & '.join(headers) + r'\\')
            lines.append(fr'\cmidrule(lr){{2-{1 + sum(widths)}}}')
            columns = [column for name in group for column in tables[name]['columns']]
            lines.append(' & ' + ' & '.join(columns) + r'\\')
            lines.append(r'\midrule')
            for model in table_models:
                lines.append(escape_label(model))
                for name, width in zip(group, widths):
                    row = tables[name]['rows'].get(model)
                    if row is None:
                        # The batch runner omits this model after running out of memory.
                        missing = 'OOM' if model == 'RNNLogic' and name == 'YAGO3-10' else '--'
                        row = (' & ' + missing) * width
                    lines.append(row)
                lines.append(r'\\')
            lines.append(r'\bottomrule')
            if start + 3 >= len(datasets):
                lines.append(r'\bottomrule')
            lines.append(r'\end{tabular}')
            lines.append('}')
            lines.append('')
        lines.extend([fr'\label{{table:{folder_name}}}', r'\end{table}'])
        tex_path = f'{folder}/{folder_name}.tex'
        with open(tex_path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        return tex_path

    def draw_rank_cluster_hist(
            self,
            mets: List[List["PROBE"]],
            cuts: Sequence[int] = (1,2,6,21,51,100),
            density: bool = False,
            both: bool = True,
            cut_last: bool = True,
            diff: bool = True,
            format: str = 'pdf',
            xlabel: bool = True
    ):

        models = [row[0].model for row in mets]
        color_ls = self.get_model_colors(mets)
        color_ls.append('grey')
        H, W = len(mets), len(mets[0])
        data = mets[0][0].data

        all_ranks = [[] for _ in range(H)]
        for i, _mets in enumerate(mets):
            for j, met in enumerate(_mets):
                all_ranks[i].append(np.asarray(met.rank, dtype=float))

        cuts = list(cuts)
        if len(cuts) == 0:
            raise ValueError("`cuts` must contain at least one integer.")
        if any(c <= 0 for c in cuts):
            raise ValueError("`cuts` should be positive integers.")
        if any(cuts[i] >= cuts[i + 1] for i in range(len(cuts) - 1)):
            raise ValueError("`cuts` must be strictly increasing.")

        slices = []
        for a, b in zip(cuts[:-1], cuts[1:]):
            slices.append((a, b - 1))
        if not cut_last: slices.append((cuts[-1], np.inf))

        def _lab(lo, hi):
            if lo < hi:
                return f"{int(lo)}~{int(hi)}" if np.isfinite(hi) else f"Others"
            elif lo == hi:
                return f"{int(lo)}"

        x_labels = [_lab(lo, hi) for (lo, hi) in slices]
        nbins = len(slices)

        fig, ax = plt.subplots(figsize=(5,2))
        total_width = 0.7
        bar_width = total_width / H
        centers = np.arange(nbins, dtype=float)  
        offsets = (np.arange(H) - (H - 1) / 2.0) * bar_width
        means = []
        stds = []
        for i in range(H):
            counts_list = []
            target_ranks = all_ranks[i]
            for r in target_ranks:
                counts = np.zeros(nbins, dtype=float)
                if r.size > 0:
                    for k, (lo, hi) in enumerate(slices):
                        if np.isfinite(hi):
                            counts[k] = np.sum((r >= lo) & (r <= hi))
                        else:
                            counts[k] = np.sum(r >= lo)

                if density:
                    s = counts.sum()
                    if s > 0:
                        counts = counts / s
                counts_list.append(counts)

            arr = np.stack(counts_list, axis=0)  
            means.append(arr.mean(axis=0)) 
            stds.append(arr.std(axis=0, ddof=1) if W > 1 else np.zeros_like(means[i]))

            bar_color = self.mods.get(models[i], None)

            if both:
                bar_color = self.mods.get(models[i], None)
                xbar = centers + offsets[i]
                ax.bar(
                    xbar,
                    means[i],
                    width=bar_width,
                    label=models[i],
                    capsize=3,
                    color=bar_color, 
                    linewidth=0.6,
                    zorder=2,
                    error_kw = {
                        "elinewidth": 1.8,  
                        "capthick": 1.8 
                    }
                )

        axis_font = 13
        highest_points = np.max(np.array(means) + np.array(stds), axis=0)
        lowest_points = np.min(np.array(means) - np.array(stds), axis=0)
        if diff:
            try:
                diff = highest_points - lowest_points
                win_sign = []
                for _diff in means[0] - means[1]:
                    _diff = int(_diff)
                    if _diff > 0: win_sign.append(0)
                    elif _diff < 0: win_sign.append(1)
                    else: win_sign.append(2)

                for i in range(len(highest_points)):
                    ax.text(
                        centers[i],
                        highest_points[i] + 100,
                        s=f'+{int(diff[i])}',
                        color=color_ls[win_sign[i]],
                        fontsize=axis_font,
                        fontweight='bold',
                        ha='center'
                    )
            except: pass

        if not cut_last:
            x_pos = sum(centers[len(centers)-2:]) / 2
            plt.axvline(x_pos, ymin=0, ymax=1, color='grey', linestyle='--')

        if not both:
            final_mean = means[0] - means[1]
            bar_color = []
            for i in range(len(means[0])):
                if final_mean[i] > 0: bar_color.append(self.mods.get(models[0], None))
                elif final_mean[i] < 0: bar_color.append(self.mods.get(models[1], None))
                else: bar_color.append('grey')
                final_mean[i] = abs(final_mean[i])
            xbar = centers
            ax.bar(
                xbar,
                final_mean,
                width=bar_width,
                capsize=3,
                color=bar_color,
                edgecolor="None",
                linewidth=0.6,
                zorder=2
            )
   
            for i in range(H):
                bar_color = self.mods.get(models[i], None)
                ax.scatter(
                    centers, means[i],
                    s=18,
                    marker=self.markers[models[i]],
                    color=bar_color,
                    zorder=4,
                    label=models[i],
                )

                ax.plot(
                    centers, means[i],
                    linewidth=1.0,
                    linestyle='-',
                    color=bar_color,
                    zorder=3,
                )

        plt.yscale('log')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis="x", length=0)
        ax.set_xticks(centers)

        if self.data == 'FB15k237': plt.yticks([2000,4000,8000],['2k','4k','8k'])
        else: plt.yticks([1000,2000,4000],['1k','2k','4k'])
        ax.tick_params(axis='y', which='both', left=False, labelleft=True, labelsize=axis_font)
        xtick_labels = []
        for i, cut in enumerate(cuts):
            if i == len(cuts) - 1: break
            if cuts[i + 1] - cut == 1: xtick_labels.append(f'{cut}')
            else:
                if cuts[-1] == cuts[i + 1]: xtick_labels.append(f'{cut}-{cuts[i + 1]}')
                else: xtick_labels.append(f'{cut}-{cuts[i + 1] - 1}')
        ax.set_xticklabels(xtick_labels,
                           fontsize=axis_font)
        if xlabel: ax.set_xlabel('Rank', fontsize=axis_font, labelpad=0)
        ax.yaxis.grid(True, color="grey", linestyle="-", linewidth=0.5, alpha=0.5, zorder=0)
        ax.set_ylabel("Normalized frequency" if density else "# prediction", fontsize=axis_font, labelpad=0)
        ax.legend(ncol=2, fontsize=axis_font - 1,loc='upper right', borderaxespad=0.0)
        ax.minorticks_off()
        ax.set_ylim(max(0, min(lowest_points) - 200), max(highest_points) + 500)
        plt.tight_layout()
        model_names = '_'.join(models)
        os.makedirs('../figs/fig2_obs1_fig9/', exist_ok=True)
        plt.savefig(f'../figs/fig2_obs1_fig9/{data}_rank_hist_{model_names}.{format}', bbox_inches='tight', pad_inches=0.01)

    @staticmethod
    def draw_vector_custom(vs: Sequence[Vector], a, b, E: int=1000, eps: float=1e-5, split=4):
        def probe(vec: Vector, a, b, E, eps):
            N_coeff = (1 / E) ** a
            transformed_ranks = ((1 / vec.v) ** a - N_coeff) / (1 - N_coeff)
            print(f'{vec.color}: a:{a}, b:{b}, tr:{transformed_ranks}')
            sep_transformed_ranks = np.array([transformed_ranks[:split].mean(), transformed_ranks[split:].mean()])

            EW = 1 / (eps + vec.epop) ** b
            EW = EW / np.mean(EW)
            norm_EW = EW / np.sum(EW)

            RW = 1 / (eps + vec.rpop) ** b
            RW = RW / np.mean(RW)
            norm_RW = RW / np.sum(RW)

            W = norm_EW * norm_RW
            norm_W = W / np.sum(W)
            print(f'{norm_W}')
            sep_norm_W = np.array([norm_W[:split].sum(), norm_W[split:].sum()])
            return sep_transformed_ranks, sep_norm_W, np.dot(transformed_ranks, norm_W)

        dim_alpha = 0.3
        if a == 1 and b == 0:
            bt_alpha = 1.0
            bw_alpha = 1.0
        elif b == 0:
            bt_alpha = dim_alpha
            bw_alpha = 1.0
        elif a == 1:
            bt_alpha = 1.0
            bw_alpha = dim_alpha
        else:
            bt_alpha = dim_alpha
            bw_alpha = dim_alpha

        fig, ax = plt.subplots(figsize=(2.3, 1.9))

        axis_font = 9
        hl = 3.0
        hal = hl * 0.8
        width=0.028

        if 1 or bt_alpha < 1: 
            for v in vs:
                tr, w, metric = probe(v, a, b, E, eps)
                ax.quiver(0, 0, tr[0], tr[1], angles='xy', scale_units='xy',
                          scale=1, alpha=1, facecolor=v.color, width=width, edgecolor='black',
                          headlength=hl, headaxislength=hal, zorder=2)
        if 1 or bw_alpha < 1: 
            tr, w, metric = probe(vs[0], a, b, E, eps)
            ax.quiver(0, 0, w[0], w[1], angles='xy', scale_units='xy', scale=1,
                      alpha=1, width=width, color='white', label='w-vector',
                      headlength=hl, headaxislength=hal, edgecolor='black', linewidth=1.2, zorder=1
                      ,linestyles='dashed')

        results = [(probe(v, a, b, E, eps)[-1], v.color) for v in vs]
        metrics = np.array([m for m, _ in results])
        rank_idx = np.argsort(metrics)[::-1] 
        rank_map = {idx: rank for rank, idx in enumerate(rank_idx)}

        plt.xlim(0, 0.9)
        plt.ylim(0.0, 0.83)
        ax.set_aspect('auto')
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])

        legend_elements = []

        for i, (metric, color) in enumerate(results):
            rank = rank_map[i]
            label = f'{metric:.3f}'
            legend_elements.append(Patch(facecolor=color, label=label))

        leg = ax.legend(handles=legend_elements, fontsize=axis_font, frameon=False,
                        labelspacing=0.1,
                        loc='upper right',
                        bbox_to_anchor=(1.05, 1.00),
                        borderaxespad=0.0
                        )
        for i, text in enumerate(leg.get_texts()):
            if rank_map[i] == 0:
                text.set_fontweight('bold')
            if rank_map[i] == 2:
                text.set_color('gray')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        ax.tick_params(axis='both',  
                       which='major', 
                       length=2,  
                       width=0.5,
                       labelsize=axis_font,)
        ax.grid(linewidth=0.5, alpha=0.7, zorder=0)
        os.makedirs(f'../figs/fig7/svg', exist_ok=True)
        title_s = f'../figs/fig7/svg/toy_{a}(a)_{b}(b)'
        os.makedirs(f'../figs/fig7/pdf', exist_ok=True)
        title_p = f'../figs/fig7/pdf/toy_{a}(a)_{b}(b)'
        plt.savefig(f'{title_s}.svg', bbox_inches='tight', dpi=400, pad_inches=0.01)
        plt.savefig(f'{title_p}.pdf', bbox_inches='tight', dpi=400, pad_inches=0.01)

    @staticmethod
    def draw_vectors():
        we = [0.5, 0.5, 0.1]
        wr = [0.8, 0.8, 0.25]
        A = Vector(np.array([1, 2, 300]), np.array(we), np.array(wr), '#DE0005')
        B = Vector(np.array([2, 3, 10]), np.array(we), np.array(wr), '#3F991F')
        C = Vector(np.array([4, 4, 5]), np.array(we), np.array(wr),'#002ADE')

        vs = [A, B, C]

        for a, b in [[1,0],[0.25,0],[1,0.8]]:
            VisTool.draw_vector_custom(vs, a, b, 1000, 1e-5, split=2)

    @staticmethod
    def draw_owa(data_dir='../data/owa', output_dir='../figs/fig12', metrics=None):

        def draw_rank_change_box(
                ax,
                top_indices,
                bottom_indices,
                model_order,
                model_wise_colors,
                top_label="f/ test",
                bottom_label="s/ test",

                standalone=True,
                frame_lw=0.8,
                frame_ec="grey",
                frame_alpha=0.2,

                content_left=0.26,
                content_right=0.96,

                top_row_y=0.72,
                bottom_row_y=0.28,

                label_x=0.04,
                label_top_y=None,
                label_bottom_y=None,

                connector_lw=1.2,
                fontsize=18,

                square_size_px=22,
                square_gap_px=2,
                square_align="center",

                show_numbers=True,
                number_start=1,
                number_fontsize=12,
                number_color="white",
                number_fontweight="bold",
        ):
            if standalone:
                ax.set_axis_off()
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)

            n = len(top_indices)
            if len(bottom_indices) != n:
                raise ValueError("top_indices and bottom_indices must have same length")

            if label_top_y is None:
                label_top_y = top_row_y
            if label_bottom_y is None:
                label_bottom_y = bottom_row_y

            frame = Rectangle(
                (0, 0), 1, 1,
                transform=ax.transAxes,
                fill=False,
                ec=frame_ec,
                lw=frame_lw,
                alpha=frame_alpha,
                clip_on=False,
                zorder=10
            )
            ax.add_patch(frame)

            row_dist = abs(top_row_y - bottom_row_y)

            ax.figure.canvas.draw()
            bbox = ax.get_window_extent()

            sq_w = square_size_px / bbox.width
            sq_h = square_size_px / bbox.height

            # Convert horizontal gap from pixels to axes coordinates
            gap_w = square_gap_px / bbox.width

            total_square_w = n * sq_w + (n - 1) * gap_w
            content_w = content_right - content_left

            if total_square_w > content_w:
                raise ValueError(
                    f"Squares do not fit: required width={total_square_w:.3f}, "
                    f"available width={content_w:.3f}. "
                    "Reduce square_size_px, square_gap_px, or increase content_right-content_left."
                )

            if square_align == "left":
                start_x = content_left + sq_w / 2

            elif square_align == "right":
                start_x = content_right - total_square_w + sq_w / 2

            elif square_align == "center":
                start_x = content_left + (content_w - total_square_w) / 2 + sq_w / 2

            else:
                raise ValueError("square_align must be one of: 'left', 'center', 'right'")

            xs = [
                start_x + i * (sq_w + gap_w)
                for i in range(n)
            ]

            ax.text(
                label_x, label_top_y,
                top_label,
                transform=ax.transAxes,
                ha="left", va="center",
                fontsize=fontsize,
                zorder=12,
                fontstyle="italic"
            )

            ax.text(
                label_x, label_bottom_y,
                bottom_label,
                transform=ax.transAxes,
                ha="left", va="center",
                fontsize=fontsize,
                zorder=12,
                fontstyle="italic"
            )

            top_pos = {}
            bottom_pos = {}

            for rank, idx in enumerate(top_indices):
                model = model_order[idx]
                color = model_wise_colors[model]
                xc = xs[rank]
                top_pos[model] = xc

                ax.add_patch(Rectangle(
                    (xc - sq_w / 2, top_row_y - sq_h / 2),
                    sq_w, sq_h,
                    transform=ax.transAxes,
                    facecolor=color,
                    edgecolor=color,
                    lw=0.8,
                    clip_on=False,
                    zorder=13
                ))

                if show_numbers:
                    ax.text(
                        xc, top_row_y - 0.03,
                        str(number_start + rank),
                        transform=ax.transAxes,
                        ha="center", va="center",
                        fontsize=number_fontsize,
                        color=number_color,
                        fontweight=number_fontweight,
                        zorder=14
                    )

            for rank, idx in enumerate(bottom_indices):
                model = model_order[idx]
                color = model_wise_colors[model]
                xc = xs[rank]
                bottom_pos[model] = xc

                ax.add_patch(Rectangle(
                    (xc - sq_w / 2, bottom_row_y - sq_h / 2),
                    sq_w, sq_h,
                    transform=ax.transAxes,
                    facecolor=color,
                    edgecolor=color,
                    lw=0.8,
                    clip_on=False,
                    zorder=13
                ))

                if show_numbers:
                    ax.text(
                        xc, bottom_row_y - 0.03,
                        str(number_start + rank),
                        transform=ax.transAxes,
                        ha="center", va="center",
                        fontsize=number_fontsize,
                        color=number_color,
                        fontweight=number_fontweight,
                        zorder=14
                    )

            for model, x_top in top_pos.items():
                x_bottom = bottom_pos[model]

                ax.add_line(Line2D(
                    [x_top, x_bottom],
                    [top_row_y - sq_h / 2, bottom_row_y + sq_h / 2],
                    transform=ax.transAxes,
                    color=model_wise_colors[model],
                    lw=connector_lw,
                    alpha=0.8,
                    zorder=11,
                    clip_on=False
                ))
                
        histories, metrics = load_owa_histories(data_dir, metrics)
        model_order = list(OWA_MODEL_COLORS)
        fontsize = 8
        tick_fontsize = 6
        axis_label_fontsize = 7
        box_fontsize = 7
        folder_plot = os.path.join(output_dir, 'plot')
        folder_box = os.path.join(output_dir, 'box')
        os.makedirs(folder_plot, exist_ok=True)
        os.makedirs(folder_box, exist_ok=True)
        summary = {}
        for metric in metrics:
            selected_full, selected_sparse, selections = [], [], {}
            x_values, y_values = [], []
            fig, ax = plt.subplots(figsize=(2, 1.5))
            fig_box, ax_box = plt.subplots(figsize=(2, 0.6))
            for model in model_order:
                pairs = histories[model][metric]
                order = np.argsort(pairs[:, 0], kind='mergesort')
                ax.plot(pairs[order, 0], pairs[order, 1], color=OWA_MODEL_COLORS[model], linewidth=0.5)
                # Select the same evaluation by sparse score for both split rankings.
                best = int(np.argmax(pairs[:, 1]))
                full, sparse = map(float, pairs[best])
                selected_full.append(full)
                selected_sparse.append(sparse)
                steps, _ = owa_training_steps(data_dir, model, len(pairs))
                selections[model] = {'evaluation_index': best, 'step': int(steps[best]),
                                     'full': full, 'sparse': sparse}
                x_values.extend(pairs[:, 0])
                y_values.extend(pairs[:, 1])
            full_sort_idx = np.argsort(-np.asarray(selected_full), kind='mergesort')
            sparse_sort_idx = np.argsort(-np.asarray(selected_sparse), kind='mergesort')
            # Average ranks retain score ties instead of breaking them using the other split.
            full_ranks = rankdata(-np.asarray(selected_full), method='average')
            sparse_ranks = rankdata(-np.asarray(selected_sparse), method='average')
            tau, tau_p = kendalltau(sparse_ranks, full_ranks)
            rho, rho_p = spearmanr(sparse_ranks, full_ranks)
            print("%s: Kendall's coefficient=%.4f, Spearman's coefficient=%.4f" % (metric, tau, rho))
            summary[metric] = {
                'selection': 'maximum sparse score; first evaluation wins exact ties',
                'selected_models': selections,
                'full_order': [model_order[i] for i in full_sort_idx],
                'sparse_order': [model_order[i] for i in sparse_sort_idx],
                'kendall_tau': float(tau) if np.isfinite(tau) else None,
                'spearman_rho': float(rho) if np.isfinite(rho) else None,
                'kendall_p': float(tau_p) if np.isfinite(tau_p) else None,
                'spearman_p': float(rho_p) if np.isfinite(rho_p) else None,
            }
            draw_rank_change_box(
                ax=ax_box, top_indices=sparse_sort_idx, bottom_indices=full_sort_idx,
                model_order=model_order, model_wise_colors=OWA_MODEL_COLORS,
                top_label='sparse', bottom_label='full  ',
                content_left=0.30, content_right=0.96, top_row_y=0.72, bottom_row_y=0.28,
                label_x=0.04, square_size_px=10, square_gap_px=0.2, show_numbers=True,
                number_start=1, number_fontsize=box_fontsize, number_color='white',
                number_fontweight='normal', fontsize=box_fontsize*0.9)
            fig_box.savefig(os.path.join(folder_box, metric + '_box.pdf'), dpi=300,
                            bbox_inches='tight', pad_inches=0.01)
            xmin, xmax = float(min(x_values)), float(max(x_values))
            ymin, ymax = float(min(y_values)), float(max(y_values))
            if metric in METRIC_NAMES:
                xpad = max((xmax - xmin)*0.04, abs(xmax)*0.01, 1e-8)
                ypad = max((ymax - ymin)*0.04, abs(ymax)*0.01, 1e-8)
                ax.set_xlim(xmin - xpad, xmax + xpad)
                if metric in ('AMRI', 'WMR_BAR'):
                    ax.set_xlim(right=1.05)
                ax.set_ylim(ymin - ypad, ymax + ypad)
                ax.xaxis.set_major_locator(
                    MultipleLocator(0.2) if metric in ('AMRI', 'WMR_BAR') else MaxNLocator(nbins=4))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
                for axis in (ax.xaxis, ax.yaxis):
                    formatter = ScalarFormatter(useMathText=True)
                    formatter.set_powerlimits((-2, 3))
                    axis.set_major_formatter(formatter)
                    axis.get_offset_text().set_fontsize(tick_fontsize)
            else:
                ax.set_yticks([round(i*0.2, 1) for i in range(6)] if ymax > 0.4 else [0, 0.1, 0.2, 0.3, 0.4])
                ax.set_xlim(0.0 if xmin < 0.05 else xmin, 1.04)
                ax.set_ylim(0.0 if ymin < 0.05 else ymin, 1.02 if ymax >= 0.94 else ymax + 0.01)
            low, high = min(xmin, ymin), max(xmax, ymax)
            ax.plot([low, high], [low, high], color='grey', ls='--', lw=2, alpha=0.5, label='ideal')
            if metric == 'MRR':
                ax.legend(loc='lower right')
            label = owa_metric_label(metric)
            ax.tick_params(axis='both', labelsize=tick_fontsize)
            ax.set_xlabel('%s (full test set)' % label, fontsize=axis_label_fontsize)
            ax.set_ylabel('%s (sparse test set)' % label, fontsize=axis_label_fontsize)
            ax.set_title(label, fontsize=fontsize)
            ax.grid(True, which='both', linestyle='-', linewidth=0.3, color='grey', alpha=0.5)
            if metric in METRIC_NAMES:
                fig.canvas.draw()
                xoffset = ax.xaxis.get_offset_text().get_text()
                yoffset = ax.yaxis.get_offset_text().get_text()
                if xoffset and xoffset == yoffset:
                    # Keep scientific scale notes inside the panel so its frame matches other metrics.
                    ax.xaxis.get_offset_text().set_visible(False)
                    ax.yaxis.get_offset_text().set_visible(False)
                    ax.text(0.96, 0.04, '%s (both axes)' % xoffset,
                            transform=ax.transAxes, ha='right', va='bottom', fontsize=fontsize - 1,
                            bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.9, 'pad': 1})
                elif xoffset:
                    ax.xaxis.get_offset_text().set_visible(False)
                    ax.set_xlabel('%s (full test set)\n%s' % (label, xoffset), fontsize=axis_label_fontsize)
            fig.savefig(os.path.join(folder_plot, metric + '.pdf'), dpi=300,
                        bbox_inches='tight', pad_inches=0.01)
            plt.close(fig)
            plt.close(fig_box)
        from owa_history import write_json
        write_json(os.path.join(output_dir, 'summary.json'), summary)
        return summary

    @staticmethod
    def reb_m8(output_dir=None, from_json=None):
        """Write the six-dataset epsilon-scale correlation table (raw Spearman rho)."""
        from reb_m8 import create_table
        return create_table(output_dir=output_dir, from_json=from_json)

    rebm8 = reb_m8

    @staticmethod
    def reb_m14(data_dir='../data/owa', output_dir='../figs/m14', metric='PROBE(a=-0.5)', yscale='auto'):
        """Plot full-test histories at their actual training steps, allowing signed metrics."""
        histories, _ = load_owa_histories(data_dir, [metric])
        fig, ax = plt.subplots(figsize=(6, 4))
        summary = {}
        full_values = []
        xlabel = 'Evaluation step'
        for model, color in OWA_MODEL_COLORS.items():
            pairs = histories[model][metric]
            full_test, sparse_test = pairs[:, 0], pairs[:, 1]
            steps, xlabel = owa_training_steps(data_dir, model, len(pairs))
            sparse_min_idx, sparse_max_idx = int(np.argmin(sparse_test)), int(np.argmax(sparse_test))
            summary[model] = {
                'sparse_min': float(sparse_test[sparse_min_idx]),
                'sparse_max': float(sparse_test[sparse_max_idx]),
                'full_at_sparse_min': float(full_test[sparse_min_idx]),
                'full_at_sparse_max': float(full_test[sparse_max_idx]),
                'step_at_sparse_min': int(steps[sparse_min_idx]),
                'step_at_sparse_max': int(steps[sparse_max_idx]),
                'final_full': float(full_test[-1]), 'final_sparse': float(sparse_test[-1]),
            }
            print('%s %s: sparse min=%.6g, max=%.6g; corresponding full=%.6g, %.6g' %
                  (metric, model, summary[model]['sparse_min'], summary[model]['sparse_max'],
                   summary[model]['full_at_sparse_min'], summary[model]['full_at_sparse_max']))
            ax.plot(steps, full_test, color=color, linewidth=1.5, label=model)
            full_values.extend(full_test)
        if yscale == 'auto':
            yscale = 'log' if min(full_values) > 0 else 'symlog'
        if yscale == 'log' and min(full_values) <= 0:
            plt.close(fig)
            raise ValueError('Non-positive values cannot be shown on a log scale')
        if yscale == 'symlog':
            threshold = 1.0 if metric == 'ZMRR' else 0.01
            ax.set_yscale(yscale, linthresh=threshold)
        else:
            ax.set_yscale(yscale)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(owa_metric_label(metric))
        ax.grid(True, which='both', linewidth=0.4, color='grey', alpha=0.5)
        ax.legend(ncol=2, fontsize=8, frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, metric + '.pdf')
        fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.01)
        plt.close(fig)
        from owa_history import write_json
        write_json(os.path.join(output_dir, metric + '.summary.json'),
                   {'metric': metric, 'yscale': yscale, 'models': summary})
        return output_path

    def draw_reb_m15c(self, mets: List[List[PROBE]], scales=None,
                     output_dir='../figs/m15c', file_format='pdf'):
        """Plot eight metrics against candidate-size scale with unchanged ranks.

        Subtract each run's scale-1 score before computing model means and
        sample standard deviation bands. Include scale 1 in custom sweeps.
        Use a fixed valid cohort within each run for all scales; ranks
        exceeding the smallest scaled set and singleton sets are excluded once.
        Counts include the target: N_i = met.nentity - met.filtered[i].
        Plot PROBE at alpha=1, 0.5, 0; 2/3-MRR and 1/3-MRR; ZMRR, WMR_BAR, AMRI.
        Return the eight figure paths.
        """
        if not mets or any(not runs for runs in mets):
            raise ValueError('`mets` must contain at least one run for every model')
        scales = np.asarray(np.linspace(0.4, 1.6, 7) if scales is None else scales,
                            dtype=float)
        if (scales.ndim != 1 or not scales.size or not np.isfinite(scales).all()
                or np.any(scales <= 0) or np.any(np.diff(scales) <= 0)):
            raise ValueError('Scales must be finite, positive, and strictly increasing')
        pivot_matches = np.isclose(scales, 1.0, rtol=1e-10, atol=1e-12)
        scales = scales.copy()
        scales[pivot_matches] = 1.0
        scales = np.unique(np.append(scales, 1.0))
        model_names = [runs[0].model for runs in mets]
        if len(set(model_names)) != len(model_names):
            raise ValueError('Each entry in `mets` must represent a different model')
        summary = {'dataset': self.data, 'scales': scales.tolist(),
                   'pivot_scale': 1.0, 'plotted_quantity': 'score_difference',
                   'probe_alphas': [1.0, 0.5, 0.0], 'pmrr_powers': [2 / 3, 1 / 3],
                   'beta': 0.0,
                   'aggregation': 'answer', 'rounding': 'floor(scale * N + 0.5)',
                   'cohort': 'Fixed per run; N >= 2 and rank <= N at every scale',
                   'models': {}}
        for model, runs in zip(model_names, mets):
            run_results = []
            for index, met in enumerate(runs):
                if met.model != model or met.data != self.data:
                    raise ValueError('All runs must match their model group and dataset')
                result = candidate_size_sweep(met.rank, met.nentity - met.filtered, scales)
                run_results.append(result)
                print(f'M15c {self.data}/{model} run {index + 1}: '
                      f"retained {result['retained_count']}/{result['record_count']} "
                      f"records; excluded {result['excluded_count']}")
            summary['models'][model] = {'runs': run_results}
        return self._save_reb_m15c(summary, output_dir, file_format)

    def _save_reb_m15c(self, summary, output_dir, file_format='pdf'):
        """Center and render saved sweep scores without recalculating rank metrics."""
        scales = np.asarray(summary['scales'], dtype=float)
        pivot_matches = np.flatnonzero(np.isclose(scales, 1.0, rtol=1e-10, atol=1e-12))
        if len(pivot_matches) != 1:
            raise ValueError('The saved sweep must contain exactly one scale-1 pivot')
        fs = 12 
        pivot_index = int(pivot_matches[0])
        scales[pivot_index] = 1.0
        summary['scales'] = scales.tolist()
        summary['pivot_scale'] = 1.0
        summary['plotted_quantity'] = 'score_difference'
        model_names = list(summary['models'])
        metric_names = CANDIDATE_SWEEP_METRICS
        for model_summary in summary['models'].values():
            runs = model_summary['runs']
            model_summary['metrics'] = {}
            for run in runs:
                run['baseline_scores'] = {
                    metric: run['scores'][metric][pivot_index] for metric in metric_names}
                run['differences'] = {
                    metric: (np.asarray(run['scores'][metric])
                             - run['baseline_scores'][metric]).tolist()
                    for metric in metric_names}
            for metric in metric_names:
                values = np.asarray([run['differences'][metric] for run in runs])
                mean = values.mean(axis=0)
                std = values.std(axis=0, ddof=1) if len(runs) > 1 else np.zeros_like(mean)
                model_summary['metrics'][metric] = {'mean': mean.tolist(), 'std': std.tolist()}

        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []
        for metric in metric_names:
            fig, ax = plt.subplots(figsize=(4, 3))
            for index, model in enumerate(model_names):
                stats = summary['models'][model]['metrics'][metric]
                mean, std = np.asarray(stats['mean']), np.asarray(stats['std'])
                color = self.mods.get(model, OWA_MODEL_COLORS.get(model, plt.get_cmap('tab10')(index % 10)))
                ax.plot(scales, mean, color=color, marker=self.markers.get(model, 'o'),
                        markersize=4, linewidth=1.5, label=model)
                ax.fill_between(scales, mean - std, mean + std, color=color, alpha=0.15)
            ax.axvline(1.0, color='grey', linestyle='--', linewidth=0.8, alpha=0.7)
            ax.axhline(0.0, color='grey', linestyle='--', linewidth=0.8, alpha=0.7)
            ax.set_xticks(scales)
            ax.set_xticklabels([rf'{scale:.3g}$\times$' for scale in scales], fontsize=fs*0.9)
            ax.set_xlabel('Candidate size scale (fixed ranks)', fontsize=fs)
            ax.set_ylabel(r'Difference from $1\times$', fontsize=fs)
            ax.set_title(f'{summary["dataset"]}({owa_metric_label(metric)})', fontsize=fs)
            formatter = ScalarFormatter(useOffset=False, useMathText=False)
            formatter.set_powerlimits((0, 0))
            ax.yaxis.set_major_formatter(formatter)
            ax.grid(axis='y', color='grey', alpha=0.7, linewidth=0.4)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend(ncol=2, fontsize=fs*0.8, frameon=False)
            fig.tight_layout()
            path = os.path.join(output_dir, f"{summary['dataset']}_{metric}.{file_format}")
            try:
                fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.01)
            finally:
                plt.close(fig)
            saved_paths.append(path)
        return saved_paths

    def fbdecode(self, id2e):
        assert self.data == 'FB15k237', f'This function can not be used in {self.data}'
        _id2e = {}
        with open(f'../data/{self.data}/FB_decoded.json') as f:
            decoded_entities = json.load(f)

        for id, ename in id2e.items():
            try:
                _id2e[id] = decoded_entities[ename]
            except:
                _id2e[id] = ename
        return _id2e

    def draw_e_and_r(self, eid, rid, id2e, id2r, wls, lls, wname, lname, threshold):
        fig, ax3 = plt.subplots(
            figsize=(4, 4),
        )
        fig.tight_layout()

        if self.data == 'FB15k237':
            id2e = self.fbdecode(id2e)

        @lru_cache(maxsize=200000)
        def decode_synset_any_pos(offset_like):
            try:
                offset = int(offset_like)
            except (TypeError, ValueError):
                return None

            for pos in ("n", "v", "a", "r"):
                try:
                    syn = wn.synset_from_pos_and_offset(pos, offset)
                except (KeyError, ValueError):
                    continue

                if syn is None:
                    continue

                names = syn.lemma_names()
                if not names:
                    continue

                return names[0]

            return None

        edge_ls = list(self.edges[eid])

        G = nx.MultiDiGraph()

        rel_dict = defaultdict(int)

        def rel_last_token(rel: str) -> str:
            rel = rel.split(".")[-1]  
            return rel.rsplit("/", 1)[-1]  

        for i, (r, e, mode) in enumerate(edge_ls):
            rel_dict[id2r[r]] += 1
            if mode == 'h':
                h, r_name, t = eid, r, e
            else:
                h, r_name, t = e, r, eid

            G.add_node(h)
            G.add_node(t)
            G.add_edge(h, t, label=r_name)

        center = eid
        others = [n for n in G.nodes() if n != center]

        pos = nx.shell_layout(G, nlist=[[center], others])

        node_default_col = '#E9E9F1'
        node_target_col = '#BABAD4'
        rel_default_col = '#EDDFDF'
        rel_target_col = '#FF5757'

        nodes = list(G.nodes()) 
        node_sizes = []
        node_colors = []
        for n in nodes:
            if n == eid: 
                node_sizes.append(500)
                node_colors.append(node_target_col)
            else:
                node_sizes.append(200)
                node_colors.append(node_default_col)

        edges = list(G.edges(data=True)) 
        edge_colors = []
        edge_widths = []

        locality = 0
        pie_dict = defaultdict(int)
        for u, v, data in edges:
            r_name = data["label"]
            if r_name == rid:
                edge_colors.append(rel_target_col)
                edge_widths.append(2)
                locality += 1
            else:
                edge_colors.append(rel_default_col)
                edge_widths.append(1)

            pie_dict[r_name] += 1

        try:
            locality = locality / len(edges)
        except:
            print('Div by zero')

        entity_font_size = 5
        relation_font_size = 4

        if self.data == "wn18rr":
            node_labels = {}
            for n in nodes:
                offset_str = id2e[n]
                decoded = decode_synset_any_pos(offset_str)
                node_labels[n] = decoded if decoded is not None else offset_str
        else:
            node_labels = {n: id2e[n] for n in nodes}

        edge_labels = {(u, v): rel_last_token(id2r[data["label"]]) for (u, v, data) in edges}

        y = 0.95  
        line_gap = 0.06 

        cnt = 0
        queries = []
        for w, l in zip(wls, lls):
            q, m, rw, tpre = w
            _, _, rl, _ = l
            h, r, t = q
            h, r, t = id2e[h], id2r[r], id2e[t]

            if rw < rl: cnt += 1
            else: cnt -= 1

            if self.data == 'wn18rr':
                h = decode_synset_any_pos(int(h))
                t = decode_synset_any_pos(int(t))
            if h is None or t is None:
                plt.close()
                return

            text = f'{h}, {r}, {t} | {int(rw)}, {int(rl)}'

            queries.append(text)
        victory = 'w' if cnt > 0 else 'l'
        model = wname if cnt > 0 else lname
        if tpre > threshold:
            if model == lname:
                eval = 'good'
            else:
                eval = 'bad'
        else:
            if model == wname:
                eval = 'good'
            else:
                eval = 'bad'

        y -= line_gap

        def make_autopct(pie_keys, target_k):
            idx = {"i": 0}

            def autopct(pct):
                _rid = pie_keys[idx["i"]]
                idx["i"] += 1
                return ""

            return autopct

        highlight_color = "red"
        default_color = "lightgray"

        pie_keys = list(pie_dict.keys())
        pie_sizes = list(pie_dict.values())
        if not pie_sizes:
            print('pie_sizes is empty')
            return
        pie_labels = ['' for k in pie_keys]

        colors = []
        cmap = plt.cm.get_cmap('binary')
        norm = mcolors.Normalize(vmin=-1, vmax=max(pie_sizes) * 1.2)

        place = -1
        for idx, k in enumerate(pie_keys):
            if rid == k:
                place = idx
                colors.append(highlight_color)
            else:
                colors.append(cmap(norm(pie_dict[k])))

        if place != -1:
            assert pie_keys[place] == rid, "What..."

            key = pie_keys.pop(place)
            pie_keys.insert(0, key)

            key = pie_sizes.pop(place)
            pie_sizes.insert(0, key)

            c = colors.pop(place)
            colors.insert(0, c)

        ax3.pie(
            pie_sizes,
            labels=pie_labels,
            colors=colors,
            autopct=make_autopct(pie_keys, rid),
            startangle=90,
            textprops={'fontsize': 25}
        )
        ax3.axis("equal")
        print(f'{victory} | Degree = {len(edge_ls)}')
        os.makedirs(f'../figs/fig10_bottom/{self.data}', exist_ok=True)
        ename = id2e[eid].replace('/','_')
        fig.savefig(f'../figs/fig10_bottom/{self.data}/{self.data}_{model}_{victory}_p({rid}l{ename})_({len(queries)})_({tpre}).svg',
                    bbox_inches='tight', pad_inches=0.01)
        plt.close(fig)
        return queries, tpre, id2e

    def get_dict_for_pre_case(self, met: PROBE):
        record_dict = defaultdict(list)
        for q, mode, rank, pre in zip(met.query, met.mode, met.rank, met.t_p_r_e):
            h, r, t = q
            _q = (h, r) if mode == 'h' else (t, r)
            record_dict[_q].append([q, mode, rank, pre])
        return record_dict

    def decide_for_pre_case(self, wls, lls, threshold):
        '''
        reject if
        1. number of quries are too small
        2. lose model won most of the time
        '''
        if len(wls) < 1: return False
        cnt = 0
        pivot = wls[0][-1]
        for w, l in zip(wls, lls):
            q, m, rw, tpre = w
            assert pivot == tpre, 'tpre does not match'
            assert lls[0][-1] == tpre, 'tpre does not match'
            _, _, rl, _ = l
            if tpre > threshold: # l should win
                if rw > rl: cnt += 1
                else: cnt -= 1
            elif tpre <= threshold: # w should win
                if rw < rl: cnt += 1
                else: cnt -= 1

        if cnt > 0:
            return True
        else: return False

    def draw_pre_cases(self, mets: List[List[PROBE]], id2e, id2r):
        win_model = mets[0][0]
        lose_model = mets[1][0]

        w_dict, l_dict = self.get_dict_for_pre_case(win_model), self.get_dict_for_pre_case(lose_model)
        assert len(w_dict) == len(l_dict), f'Dictionary length not equal : W({len(w_dict)}) vs L({len(l_dict)})'

        os.makedirs(f'../figs/fig10_bottom', exist_ok=True)
        with open(f'../figs/fig10_bottom/{self.data}_queries.txt', 'w') as f:
            pass

        all_dict = defaultdict(list)
        threshold = 0.3
        for e_r in w_dict.keys():
            e, r = e_r
            w_ls, l_ls = sorted(w_dict[e_r]), sorted(l_dict[e_r])
            assert len(w_ls) == len(l_ls), f'Internal list length not equal : W({len(w_ls)}) vs L({len(l_ls)})'
            if self.trn_count_info[e] > 100: continue
            draw = self.decide_for_pre_case(w_ls, l_ls, threshold)
            if draw:
                queries, tpre, id2e = self.draw_e_and_r(e, r, id2e, id2r, w_ls, l_ls, win_model.model, lose_model.model, threshold)
                with open(f'../figs/fig10_bottom/{self.data}_queries.txt', 'a', encoding='utf-8') as f:
                    f.write(f'{r}|{id2e[e]} ({round(tpre,4)})\n')
                    if queries is None: continue
                    for q in queries:
                        f.write(f'{q}\n')
                    f.write('\n')

    class Info:
        def __init__(self, ls):
            self.query = ls[0]
            self.target = ls[1]
            self.pop = float(ls[2])
            self.rank = int(float(ls[3][5:]))
            self.scores = np.array([float(str_score) for str_score in ls[4:]])

    def draw_e_cases(self, args, e2id, r2id, threshold):
        who_win = args.models[0]
        model1, model2 = args.models[0], args.models[1]
        model1_path, model2_path = f'../case_info/{args.data}_{model1}_case.json', \
            f'../case_info/{self.data}_{model2}_case.json'
        ls1, ls2 = get_ship(model1_path), get_ship(model2_path)
        os.makedirs(f'../figs/fig10_top/{self.data}/{model1}', exist_ok=True)
        os.makedirs(f'../figs/fig10_top/{self.data}/{model2}', exist_ok=True)

        if self.data == 'FB15k237':
            decoded = get_decoded(f'../data/FB15k237/FB_decoded.json')
            new_e2id = {}
            for ename, eid in e2id.items():
                try:
                    new_e2id[decoded[ename]] = eid
                except:
                    new_e2id[ename] = eid
            e2id = new_e2id

        def decoder_check(query):
            for el in query:
                if 'm/' in el: return False
            return True

        def win_check(rank1, rank2):
            if who_win == model1:
                return rank1 < rank2
            else:
                return rank1 > rank2

        def norm_win_check(norm1, rank1, norm2, rank2):
            if who_win == model1:
                return norm1[100 - rank1] > norm2[100 - rank2]
            else:
                return norm1[100 - rank1] < norm2[100 - rank2]

        total_pairs = min(len(ls1), len(ls2)) 

        for l1, l2 in tqdm(zip(ls1, ls2),
                           total=total_pairs,
                           desc="Rendering case figs",
                           unit="pair",
                           dynamic_ncols=True):
            mod1_info, mod2_info = self.Info(l1), self.Info(l2)
            assert mod1_info.query == mod2_info.query
            assert mod1_info.target == mod2_info.target
            assert mod1_info.pop == mod2_info.pop

            def normalize(x):
                mn, mx = np.min(x), np.max(x)
                return np.zeros_like(x) if mx == mn else (x - mn) / (mx - mn)

            list1_norm = normalize(mod1_info.scores)[::-1]
            list2_norm = normalize(mod2_info.scores)[::-1]

            who_win = model1 if mod1_info.pop > threshold else model2

            if mod1_info.rank <= 100 and \
                mod2_info.rank <= 100 and \
                decoder_check(mod1_info.query) and \
                win_check(mod1_info.rank, mod2_info.rank) and \
                norm_win_check(list1_norm, mod1_info.rank, list2_norm, mod2_info.rank):

                def sanitize_filename(s):
                    return s.replace('/', '_').replace('%', '_').replace(':', '_').replace(' ', '_')

                fig, ax = plt.subplots(figsize=(2, 2))

                wo_ans_list1_norm, wo_ans_list2_norm = np.delete(list1_norm, 100 - mod1_info.rank), np.delete(
                    list2_norm, 100 - mod2_info.rank)
                vp = ax.violinplot(
                    [wo_ans_list1_norm, wo_ans_list2_norm],
                    positions=[-0.5, 0.5],
                    widths=0.35,
                    vert=True,
                    showmeans=False,
                    showmedians=False,
                    showextrema=False
                )

                for body, color in zip(vp['bodies'], [self.mods[model1], self.mods[model2]]):
                    body.set_facecolor(color)
                    body.set_alpha(0.3)
                    body.set_edgecolor('black')  
                    body.set_linewidth(0.8) 

                ax.scatter([-0.5] * len(wo_ans_list1_norm), wo_ans_list1_norm, s=30, marker='x', color=self.mods[model1],
                           label=model1, alpha=0.5)
                ax.scatter([0.5] * len(wo_ans_list2_norm), wo_ans_list2_norm, s=30, marker='x', color=self.mods[model2],
                           label=model2, alpha=0.5)

                idx1 = 100 - mod1_info.rank
                idx2 = 100 - mod2_info.rank
                ax.scatter(-0.5, list1_norm[idx1], s=150, marker='*', color=self.mods[model1], edgecolor='black',
                           linewidths=0.6, zorder=3)
                ax.scatter(0.5, list2_norm[idx2], s=150, marker='*', color=self.mods[model2], edgecolor='black',
                           linewidths=0.6, zorder=3)

                who_bold = lambda x: 'bold' if x == who_win else 'normal'
                ax.text(-0.3, list1_norm[idx1], f'{mod1_info.rank}', va='center', fontsize=10,
                        fontweight=who_bold(model1))
                ax.text(0.7, list2_norm[idx2], f'{mod2_info.rank}', va='center', fontsize=10,
                        fontweight=who_bold(model2))

                for i in range(6):
                    ax.axhline(i * 0.2, color='grey', linestyle='--', alpha=0.5, linewidth=0.5)

                ax.axvline(0, color='black', linewidth=0.5)

                ax.set_ylim(-0.05, 1.05)

                axis_fontsize = 12
                ax.set_xlim(-1, 1)
                ax.set_xticks([-0.5, 0.5])
                ax.set_xticklabels([f'{model1}', f'{model2}'], fontsize=axis_fontsize)
                ax.set_yticks([])

                ax.grid(True, axis='y', linestyle='--', alpha=0.35)
                new_path = f'../figs/fig10_top/{args.data}/{who_win}'

                plt.title(f'Top {round(mod1_info.pop, 2)}%')
                rank_info = f'{mod1_info.rank} vs {mod2_info.rank}'
                q2ids = f'{e2id[mod1_info.query[0]]}.{r2id[mod1_info.query[1]]}.{e2id[mod1_info.query[2]]}'
                filename = f"{self.data}_top{round(mod1_info.pop, 4)}_{q2ids}_{rank_info}.pdf"
                filename = sanitize_filename(filename)

                fig.savefig(os.path.join(new_path, filename), dpi=300, bbox_inches='tight', pad_inches=0.01)
                plt.close(fig)

    def reb_m9(self, mets: List[List[PROBE]], output_dir='../figs/m9/'):
        """Plot the same beta-sweep diagnostics for base, clip, shrink, and log weights."""
        met = mets[0][0]

        betas = [0.0, 0.2, 0.4, 0.6, 0.8]

        def create_figure():
            # Reserve the same space for labels and end ticks in every panel.
            fig, ax = plt.subplots(figsize=(4, 3), layout='none')
            fig.subplots_adjust(left=0.25, right=0.95, bottom=0.22, top=0.92)
            return fig, ax

        pending_figures = []

        def save_figure(fig, filename):
            # Measure all panels before choosing their shared tight crop.
            pending_figures.append((fig, filename))

        def l1_normalize(weights):
            weights = np.asarray(weights, dtype=float)

            total = weights.sum()

            if total <= 0:
                raise ValueError("Weights must have a positive sum.")

            return weights / total

        def gini_coefficient(weights):
            """
            Gini coefficient of non-negative weights.

            0 -> perfectly uniform
            1 -> maximally concentrated
            """
            weights = np.asarray(weights, dtype=float)

            if len(weights) == 0:
                return np.nan

            if np.any(weights < 0):
                raise ValueError(
                    "Gini coefficient requires non-negative weights."
                )

            total = weights.sum()

            if total == 0:
                return 0.0

            x = np.sort(weights)
            n = len(x)

            index = np.arange(1, n + 1)

            gini = (
                2.0 * np.sum(index * x) / (n * total)
                - (n + 1.0) / n
            )

            return gini

        def kish_ess(weights):
            """
            Kish effective sample size:

                ESS = (sum w)^2 / sum(w^2)

            For L1-normalized weights:

                ESS = 1 / sum(w^2)
            """
            weights = np.asarray(weights, dtype=float)

            numerator = weights.sum() ** 2
            denominator = np.sum(weights ** 2)

            if denominator == 0:
                return 0.0

            return numerator / denominator

        def top_mass(weights, fraction):
            """
            Fraction of total weight mass assigned to the
            highest-weight `fraction` of queries.
            """
            weights = np.asarray(weights, dtype=float)

            if len(weights) == 0:
                return np.nan

            sorted_weights = np.sort(weights)[::-1]

            k = max(
                1,
                int(np.ceil(len(sorted_weights) * fraction))
            )

            total = sorted_weights.sum()

            if total == 0:
                return 0.0

            return sorted_weights[:k].sum() / total

        def lorenz_curve(weights):
            """
            x:
                cumulative fraction of queries

            y:
                cumulative fraction of weight mass
            """
            weights = np.asarray(weights, dtype=float)

            if np.any(weights < 0):
                raise ValueError(
                    "Lorenz curve requires non-negative weights."
                )

            sorted_weights = np.sort(weights)

            total = sorted_weights.sum()

            if total == 0:
                cumulative_mass = np.zeros(len(sorted_weights))
            else:
                cumulative_mass = (
                    np.cumsum(sorted_weights) / total
                )

            y = np.concatenate(
                [[0.0], cumulative_mass]
            )

            x = np.linspace(
                0.0,
                1.0,
                len(y),
            )

            return x, y

        def clipping_weights(weights, percentile=99.0):
            """
            Hard clipping / winsorization.

            The largest weights are capped at the specified
            percentile, after which weights are L1-normalized.

                w'_i = min(w_i, tau)

            where tau is the percentile threshold.
            """
            weights = l1_normalize(weights)

            threshold = np.percentile(
                weights,
                percentile,
            )

            modified = np.minimum(
                weights,
                threshold,
            )

            modified = l1_normalize(modified)

            return modified, threshold

        def shrinkage_weights(weights, lam=0.1):
            """
            Shrink popularity-aware weights toward uniform weights.

                w'_i =
                    (1-lambda) * w_i
                    + lambda / N

            lambda = 0:
                original PROBE weights

            lambda = 1:
                completely uniform weights
            """
            weights = l1_normalize(weights)

            n = len(weights)

            uniform = np.full(
                n,
                1.0 / n,
                dtype=float,
            )

            modified = (
                (1.0 - lam) * weights
                + lam * uniform
            )

            return l1_normalize(modified)

        def logarithmic_weights(weights):
            """
            Logarithmic compression.

            Since normalized weights themselves are very small,
            applying log(1 + w) directly would barely change them.

            Therefore express each weight relative to the
            uniform-query weight:

                relative_i = w_i / (1/N)
                        = N * w_i

            and compress:

                w'_i ∝ log(1 + N*w_i)

            This is dimensionless and independent of the
            absolute scale of the raw weights.
            """
            weights = l1_normalize(weights)

            n = len(weights)

            relative_weights = n * weights

            modified = np.log1p(
                relative_weights
            )

            return l1_normalize(modified)

        def temperature_weights(weights, temperature=2.0):
            """
            Temperature / power flattening.

                w'_i ∝ w_i^(1/T)

            T = 1:
                original weights

            T > 1:
                flatter weights

            For PROBE's inverse-power weighting this is
            approximately equivalent to reducing beta from
            beta to beta/T.
            """
            if temperature <= 0:
                raise ValueError(
                    "Temperature must be > 0."
                )

            weights = l1_normalize(weights)

            modified = np.zeros_like(
                weights,
                dtype=float,
            )

            positive = weights > 0

            # Divide by max first for numerical stability.
            max_weight = weights[positive].max()

            modified[positive] = (
                weights[positive] / max_weight
            ) ** (1.0 / temperature)

            return l1_normalize(modified)

        norm_weights = []

        for b in betas:
            # Concentration depends on the weight functions, not on rank scores.
            met.set_entity_weight(b)
            met.set_relation_weight(b)
            weights = l1_normalize(
                met.entity_raw_weights * met.relation_raw_weights
            )
            met.norm_W = weights.copy()

            norm_weights.append(weights)

        new_path = output_dir
        os.makedirs(new_path, exist_ok=True)

        def concentration_frame(weights_by_beta):
            concentration_results = []

            for b, weights in zip(
                betas,
                weights_by_beta,
            ):
                n = len(weights)

                result = {
                    'beta': b,
                    'n_queries': n,

                    'gini':
                        gini_coefficient(weights),

                    'kish_ess':
                        kish_ess(weights),

                    'ess_ratio':
                        kish_ess(weights) / n,

                    'top_1pct_mass':
                        top_mass(weights, 0.01),

                    'top_5pct_mass':
                        top_mass(weights, 0.05),

                    'top_10pct_mass':
                        top_mass(weights, 0.10),

                    'max_weight':
                        np.max(weights),

                    'median_weight':
                        np.median(weights),

                    'mean_weight':
                        np.mean(weights),

                    'min_weight':
                        np.min(weights),

                    'total_weight':
                        np.sum(weights),
                }

                concentration_results.append(
                    result
                )

            return pd.DataFrame(
                concentration_results
            )

        concentration_df = concentration_frame(norm_weights)

        print('\n' + '=' * 80)
        print(
            f'M9 Weight Concentration: {self.data}'
        )
        print('=' * 80)

        print(
            concentration_df[
                [
                    'beta',
                    'gini',
                    'kish_ess',
                    'ess_ratio',
                    'top_1pct_mass',
                    'top_5pct_mass',
                    'top_10pct_mass',
                    'max_weight',
                    'total_weight',
                ]
            ].to_string(
                index=False,
                float_format=lambda x: f'{x:.6f}'
            )
        )

        print('=' * 80 + '\n')

        weight_variants = {
            'base': norm_weights,
            'clip_99': [clipping_weights(weights, percentile=99.0)[0]
                        for weights in norm_weights],
            'shrink_0.1': [shrinkage_weights(weights, lam=0.1)
                           for weights in norm_weights],
            'log': [logarithmic_weights(weights) for weights in norm_weights],
        }

        def plot_weight_diagnostics(norm_weights, concentration_df, suffix=''):

            fig, ax = create_figure()

            for i, (b, weights) in enumerate(
                zip(betas, norm_weights)
            ):
                sorted_weights = np.sort(
                    weights
                )[::-1]

                ax.plot(
                    np.arange(
                        1,
                        len(sorted_weights) + 1
                    ),
                    sorted_weights,
                    color=f'C{i}',
                    linewidth=1.5,
                    label=fr'$\beta$={b}',
                )

            ax.set_xlabel('Query index')
            ax.set_ylabel('Normalized weight')

            ax.set_yscale('log')

            ax.legend()

            ax.set_axisbelow(True)
            ax.grid(axis='y', which='major', color='grey', alpha=0.7, linewidth=0.5)

            save_figure(
                fig, os.path.join(
                    new_path,
                    f'{self.data}_unit_weight{suffix}.pdf'
                ),
            )

            plt.close(fig)

            fig, ax = create_figure()

            all_weights = np.concatenate(
                weight_variants['base']
            )

            positive_weights = (
                all_weights[all_weights > 0]
            )

            if len(positive_weights) == 0:
                raise ValueError(
                    'All normalized weights are zero.'
                )

            weight_min = positive_weights.min()
            weight_max = positive_weights.max()

            if np.isclose(
                weight_min,
                weight_max,
            ):
                lower = weight_min / 1.1
                upper = weight_max * 1.1

                bin_edges = np.array(
                    [lower, upper]
                )

            else:
                bin_edges = np.logspace(
                    np.log10(weight_min),
                    np.log10(weight_max),
                    31,
                )

            for i, (b, weights) in enumerate(
                zip(betas, norm_weights)
            ):
                positive = weights[
                    weights > 0
                ]

                ax.hist(
                    positive,
                    bins=bin_edges,

                    # Sum of actual normalized weight mass
                    # within each weight interval.
                    weights=positive,

                    histtype='step',
                    linewidth=1.5,
                    color=f'C{i}',
                    label=fr'$\beta$={b}',
                )

            ax.set_xscale('log')
            ax.set_yscale('log')

            ax.set_xlabel(
                'Normalized weight'
            )
            ax.set_ylabel(
                'Aggregated weight mass'
            )

            ax.set_ylim(bottom=0)

            ax.legend()

            ax.set_axisbelow(True)
            ax.grid(axis='y', which='major', color='grey', alpha=0.7, linewidth=0.5)

            save_figure(
                fig, os.path.join(
                    new_path,
                    f'{self.data}_clus_weight{suffix}.pdf'
                ),
            )

            plt.close(fig)

            fig, ax = create_figure()

            for i, (b, weights) in enumerate(
                zip(betas, norm_weights)
            ):
                x, y = lorenz_curve(weights)

                gini = gini_coefficient(
                    weights
                )

                ax.plot(
                    x,
                    y,
                    color=f'C{i}',
                    linewidth=1.5,
                    label=(
                        fr'$\beta$={b} '
                        fr'($G$={gini:.2f})'
                    ),
                )

            ax.set_xlabel(
                'Cumulative fraction of queries'
            )
            ax.set_ylabel(
                'Cumulative weight mass'
            )

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

            ax.legend(
                fontsize=8
            )

            ax.set_axisbelow(True)
            ax.grid(axis='y', which='major', color='grey', alpha=0.7, linewidth=0.5)

            save_figure(
                fig, os.path.join(
                    new_path,
                    f'{self.data}_lorenz_weight{suffix}.pdf'
                ),
            )

            plt.close(fig)

            fig, ax = create_figure()

            ax.plot(
                betas,
                concentration_df[
                    'ess_ratio'
                ].values,
                marker='o',
                linewidth=1.5,
                color='#8c564b',
            )

            ax.set_xlabel(r'$\beta$')
            ax.set_ylabel(
                'Effective sample ratio'
            )

            ax.set_ylim(0, 1.05)

            ax.set_axisbelow(True)
            ax.grid(axis='y', which='major', color='grey', alpha=0.7, linewidth=0.5)

            save_figure(
                fig, os.path.join(
                    new_path,
                    f'{self.data}_ess_weight{suffix}.pdf'
                ),
            )

            plt.close(fig)

            fig, ax = create_figure()

            ax.plot(
                betas,
                concentration_df[
                    'top_1pct_mass'
                ].values,
                marker='o',
                linewidth=1.5,
                color='#888888',
                label='Top 1%',
            )

            ax.plot(
                betas,
                concentration_df[
                    'top_5pct_mass'
                ].values,
                marker='s',
                linewidth=1.5,
                color='#555555',
                label='Top 5%',
            )

            ax.plot(
                betas,
                concentration_df[
                    'top_10pct_mass'
                ].values,
                marker='^',
                linewidth=1.5,
                color='#222222',
                label='Top 10%',
            )

            ax.set_xlabel(r'$\beta$')
            ax.set_ylabel(
                'Fraction of total weight mass'
            )

            ax.set_ylim(0, 1)

            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[::-1], labels[::-1])

            ax.set_axisbelow(True)
            ax.grid(axis='y', which='major', color='grey', alpha=0.7, linewidth=0.5)

            save_figure(
                fig, os.path.join(
                    new_path,
                    f'{self.data}_top_mass_weight{suffix}.pdf'
                ),
            )

            plt.close(fig)

        for variant, weights_by_beta in weight_variants.items():
            suffix = '' if variant == 'base' else f'_{variant}'
            diagnostics = (concentration_df if variant == 'base'
                           else concentration_frame(weights_by_beta))
            plot_weight_diagnostics(weights_by_beta, diagnostics, suffix)

        target_beta = 0.8

        target_idx = betas.index(
            target_beta
        )

        original = norm_weights[
            target_idx
        ].copy()

        n = len(original)

        clipped, clip_threshold = (
            clipping_weights(
                original,
                percentile=99.0,
            )
        )

        #

        shrink_lambda = 0.10

        shrunk = shrinkage_weights(
            original,
            lam=shrink_lambda,
        )

        logged = logarithmic_weights(
            original
        )

        temperature = 2.0

        tempered = temperature_weights(
            original,
            temperature=temperature,
        )

        mitigation_weights = {
            'Original': original,

            'Clip-99%': clipped,

            fr'Shrink ($\lambda$={shrink_lambda})':
                shrunk,

            'Log':
                logged,

        }

        mitigation_results = []

        for name, weights in (
            mitigation_weights.items()
        ):
            result = {
                'method': name,

                'gini':
                    gini_coefficient(weights),

                'kish_ess':
                    kish_ess(weights),

                'ess_ratio':
                    kish_ess(weights) / n,

                'top_1pct_mass':
                    top_mass(weights, 0.01),

                'top_5pct_mass':
                    top_mass(weights, 0.05),

                'top_10pct_mass':
                    top_mass(weights, 0.10),

                'max_weight':
                    weights.max(),

                'median_weight':
                    np.median(weights),

                'min_weight':
                    weights.min(),

                'total_weight':
                    weights.sum(),
            }

            mitigation_results.append(
                result
            )

        mitigation_df = pd.DataFrame(
            mitigation_results
        )

        print('\n' + '=' * 100)
        print(
            f'M9 Mitigation Analysis: '
            f'{self.data}, beta={target_beta}'
        )
        print('=' * 100)

        print(
            f'99th percentile clipping threshold: '
            f'{clip_threshold:.8e}'
        )

        print(
            mitigation_df.to_string(
                index=False,
                float_format=lambda x: f'{x:.6f}'
            )
        )

        print('=' * 100 + '\n')

        # Keep the original query order across all mitigation curves.

        order = np.argsort(
            original
        )[::-1]

        fig, ax = create_figure()

        for name, weights in (
            mitigation_weights.items()
        ):
            ax.plot(
                np.arange(
                    1,
                    n + 1
                ),
                weights[order],
                linewidth=1.5,
                label=name,
            )

        ax.set_xlabel(
            'Query index (ordered by original weight)'
        )

        ax.set_ylabel(
            'Normalized weight'
        )

        ax.set_yscale('log')

        ax.legend(
            fontsize=7
        )

        ax.set_axisbelow(True)
        ax.grid(axis='y', which='major', color='grey', alpha=0.7, linewidth=0.5)

        save_figure(
            fig, os.path.join(
                new_path,
                f'{self.data}_mitigation_weights.pdf'
            ),
        )

        plt.close(fig)

        fig, ax = create_figure()

        for name, weights in (
            mitigation_weights.items()
        ):
            x, y = lorenz_curve(
                weights
            )

            gini = gini_coefficient(
                weights
            )

            ax.plot(
                x,
                y,
                linewidth=1.5,
                label=(
                    f'{name} '
                    f'($G$={gini:.2f})'
                ),
            )

        ax.set_xlabel(
            'Cumulative fraction of queries'
        )

        ax.set_ylabel(
            'Cumulative weight mass'
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.legend(
            fontsize=7
        )

        ax.set_axisbelow(True)
        ax.grid(axis='y', which='major', color='grey', alpha=0.7, linewidth=0.5)

        save_figure(
            fig, os.path.join(
                new_path,
                f'{self.data}_mitigation_lorenz.pdf'
            ),
        )

        plt.close(fig)

        fig, ax = create_figure()

        positive_original = (
            original > 0
        )

        for name, weights in (
            mitigation_weights.items()
        ):

            mask = (
                positive_original
                & (weights > 0)
            )

            # Rasterize dense points in the PDF; keep axes and text as vectors.
            ax.scatter(
                original[mask],
                weights[mask],
                s=5,
                alpha=0.35,
                label=name,
                rasterized=True,
            )

        combined_positive = np.concatenate([
            original[original > 0],
            clipped[clipped > 0],
            shrunk[shrunk > 0],
            logged[logged > 0],
            tempered[tempered > 0],
        ])

        diagonal_min = (
            combined_positive.min()
        )

        diagonal_max = (
            combined_positive.max()
        )

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_xlabel(
            'Original normalized weight'
        )

        ax.set_ylabel(
            'Modified normalized weight'
        )

        ax.legend(
            fontsize=7
        )

        ax.set_axisbelow(True)
        ax.grid(axis='y', which='major', color='grey', alpha=0.7, linewidth=0.5)

        save_figure(
            fig, os.path.join(
                new_path,
                f'{self.data}_mitigation_mapping.pdf'
            ),
        )

        plt.close(fig)

        # A common content boundary keeps tight exports the same size without
        # changing the axes to accommodate longer tick labels in one panel.
        from matplotlib.transforms import Bbox
        content_bounds = []
        for fig, _ in pending_figures:
            fig.canvas.draw()
            content_bounds.append(fig.get_tightbbox(fig.canvas.get_renderer()))
        # Allow for small font-metric differences between screen and PDF renderers.
        crop = Bbox.union(content_bounds).padded(0.02)
        for fig, filename in pending_figures:
            crop_frame = Rectangle(
                (crop.x0, crop.y0), crop.width, crop.height,
                transform=fig.dpi_scale_trans, facecolor='none', edgecolor='none',
                clip_on=False)
            fig.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.04,
                        bbox_extra_artists=[crop_frame])

        return concentration_df, mitigation_df

    def reb_m16(
            self,
            mets: List[List[PROBE]],
            alphas: Sequence[float] = (1.0, 0.5, 0.0, -0.5, -1.0),
            betas: Sequence[float] = (0.0, 0.2, 0.4, 0.6, 0.8),
            output_dir: str = '../figs/m16',
            file_format: str = 'pdf',
            dispersion: str = 'variance',
    ):
        """Plot model performance and adjacent-model significance for each (alpha, beta).

        Bar heights and error bars are the mean and sample standard deviation
        over each model's runs. Models are sorted by increasing mean PROBE score
        in every plot. Significance labels use a two-sided Welch t-test between
        adjacent bars.

        Also plot sample variance (or standard deviation with dispersion='std')
        divided by that model's mean at the same beta, as a percentage, at
        alpha=1.0, with one line per model in its signature color.
        Positive means are required.
        This extra figure is produced even if `alphas` does not include 1.0.
        Both figures share beta across both weight factors and use finite scores only; a single finite run
        has zero dispersion, matching the bar plot's error bars.

        Returns
        -------
        list[str]
            Bar figure paths in alpha-major order, followed by the dispersion figure.
        """
        if not mets or any(not model_runs for model_runs in mets):
            raise ValueError("`mets` must contain at least one run for every model.")
        if len(alphas) == 0 or len(betas) == 0:
            raise ValueError("`alphas` and `betas` must be non-empty.")

        if dispersion not in ('variance', 'std'):
            raise ValueError("`dispersion` must be 'variance' or 'std'.")

        model_names = [model_runs[0].model for model_runs in mets]
        if len(set(model_names)) != len(model_names):
            raise ValueError("Each entry in `mets` must represent a different model.")

        missing_colors = [name for name in model_names if name not in self.mods]
        if missing_colors:
            raise ValueError(f'Missing signature colors for models: {missing_colors}')

        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []

        def significance_label(scores_a, scores_b):
            scores_a = np.asarray(scores_a, dtype=float)
            scores_b = np.asarray(scores_b, dtype=float)
            scores_a = scores_a[np.isfinite(scores_a)]
            scores_b = scores_b[np.isfinite(scores_b)]

            if scores_a.size < 2 or scores_b.size < 2:
                return 'n.s.'

            _, p_value = ttest_ind(
                scores_a,
                scores_b,
                equal_var=False,
                nan_policy='omit',
            )
            return self._stars(p_value) if np.isfinite(p_value) else 'n.s.'

        def model_results(alpha, beta):
            results = []
            for model_index, (model_name, model_runs) in enumerate(zip(model_names, mets)):
                scores = np.asarray([
                    met.calculate_final_metric(alpha, beta)
                    for met in model_runs
                ], dtype=float)
                finite_scores = scores[np.isfinite(scores)]
                if finite_scores.size == 0:
                    raise ValueError(
                        f'No finite PROBE scores for {model_name} at '
                        f'alpha={alpha}, beta={beta}.'
                    )
                results.append({
                    'model_index': model_index,
                    'model_name': model_name,
                    'scores': finite_scores,
                    'mean': float(np.mean(finite_scores)),
                    'std': (
                        float(np.std(finite_scores, ddof=1))
                        if finite_scores.size > 1 else 0.0
                    ),
                })
            return results

        dispersion_results = {}
        for alpha in alphas:
            for beta in betas:
                results = model_results(alpha, beta)
                if alpha == 1.0:
                    dispersion_results[beta] = list(results)

                # Retain input order for equal means so sorting is deterministic.
                results.sort(key=lambda result: (result['mean'], result['model_index']))

                x = np.arange(1, len(results) + 1, dtype=float)
                means = np.asarray([result['mean'] for result in results])
                stds = np.asarray([result['std'] for result in results])
                bar_colors = [self.mods[result['model_name']] for result in results]

                fig, ax = plt.subplots(figsize=(3.2, 2.0))
                ax.bar(
                    x,
                    means,
                    yerr=stds,
                    width=0.65,
                    color=bar_colors,
                    edgecolor='black',
                    linewidth=0.8,
                    capsize=4,
                    error_kw={'elinewidth': 1.2, 'capthick': 1.2},
                    zorder=2,
                )

                lower = float(np.min(means - stds))
                upper = float(np.max(means + stds))
                value_range = upper - lower
                if value_range <= 0:
                    value_range = max(abs(upper), 1.0) * 0.05

                bracket_step = 0.16 * value_range
                bracket_height = 0.035 * value_range
                bracket_base = upper + 0.01 * value_range

                for pair_index in range(len(results) - 1):
                    left_x, right_x = x[pair_index], x[pair_index + 1]
                    bracket_y = bracket_base + pair_index * bracket_step
                    ax.plot(
                        [left_x, left_x, right_x, right_x],
                        [bracket_y, bracket_y + bracket_height,
                         bracket_y + bracket_height, bracket_y],
                        color='black',
                        linewidth=1.0,
                        clip_on=False,
                        zorder=3,
                    )
                    ax.text(
                        (left_x + right_x) / 2.0,
                        bracket_y + bracket_height + 0.01 * value_range,
                        significance_label(
                            results[pair_index]['scores'],
                            results[pair_index + 1]['scores'],
                        ),
                        ha='center',
                        va='bottom',
                        fontsize=10,
                        fontweight='bold',
                    )

                top = bracket_base
                if len(results) > 1:
                    top += (len(results) - 2) * bracket_step + bracket_height
                top += 0.14 * value_range

                ax.set_xlim(0.4, len(results) + 0.6)
                ax.set_ylim(lower - 0.10 * value_range, top)
                ax.set_xticks([])
                ax.yaxis.set_major_locator(
                    MaxNLocator(nbins=5, steps=[1, 2, 5, 10], min_n_ticks=3)
                )
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                ax.set_title(
                    rf'PROBE($\alpha$={float(alpha):g}, $\beta$={float(beta):g})'
                )
                ax.yaxis.grid(
                    True, color='grey', linestyle='-', linewidth=0.5,
                    alpha=0.4, zorder=0,
                )
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                fig.tight_layout()

                filename = (
                    f'{self.data}_alpha_{alpha}_'
                    f'beta_{beta}.{file_format}'
                )
                save_path = os.path.join(output_dir, filename)
                fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.01)
                plt.close(fig)
                saved_paths.append(save_path)

        sorted_betas = sorted(betas)
        for beta in sorted_betas:
            if beta not in dispersion_results:
                dispersion_results[beta] = model_results(1.0, beta)
        values = np.asarray([
            [result['std'] for result in dispersion_results[beta]]
            for beta in sorted_betas
        ])
        means = np.asarray([
            [result['mean'] for result in dispersion_results[beta]]
            for beta in sorted_betas
        ])
        if np.any(means <= 0.0):
            beta_index, model_index = np.argwhere(means <= 0.0)[0]
            raise ValueError(
                f'Relative dispersion requires a positive mean for '
                f'{model_names[model_index]} at alpha=1.0, '
                f'beta={sorted_betas[beta_index]}.'
            )
        if dispersion == 'variance':
            values = values ** 2
        # Every model/beta pair uses its own run mean as the denominator.
        values = values / means

        # Match the M9 diagnostic canvas and reserve the same label margins.
        fig, ax = plt.subplots(figsize=(4, 3), layout='none')
        fig.subplots_adjust(left=0.25, right=0.95, bottom=0.22, top=0.92)
        for model_index, model_name in enumerate(model_names):
            ax.plot(
                sorted_betas, values[:, model_index],
                color=self.mods[model_name], marker='o', linewidth=1.5, label=model_name,
            )
        ax.set_xlabel(r'$\beta$', fontsize=14)
        ax.set_ylabel(
            'Variance / mean (%)' if dispersion == 'variance' else 'Std / mean (%)',
            fontsize=14,
        )
        ax.tick_params(axis='both', labelsize=12)
        ax.yaxis.set_major_formatter(
            PercentFormatter(xmax=1.0, decimals=1 if dispersion == 'variance' else 0)
        )
        ax.set_xticks(sorted_betas)
        if dispersion == 'variance':
            ax.set_ylim(0.0, 0.005)
        else:
            ax.set_ylim(bottom=0.0)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color='grey', linewidth=0.5, alpha=0.7)
        ax.legend(
            fontsize=11, loc='upper center',
            ncol=min(2, len(model_names)),
            handlelength=1.3, handletextpad=0.4, columnspacing=0.8,
            borderpad=0.35,
        )

        save_path = os.path.join(
            output_dir, f'{self.data}_alpha_1.0_beta_{dispersion}.{file_format}'
        )
        fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.01)
        plt.close(fig)
        saved_paths.append(save_path)

        return saved_paths

    def reb_m6(self, Es: List, ls: List):
        def delta(x, E, l):
            diff = E - x
            return (1-l)*(np.log(diff) / diff)
        output_dir = '../figs/m6'
        os.makedirs(output_dir, exist_ok=True)
        for l in ls:
            fig, ax = plt.subplots(figsize=(4, 3))
            for i, E in enumerate(Es):
                x = np.arange(E)
                y = delta(x, E, l)
                ax.plot(x, y, color=f'C{i}', label=f'{int(E/1000)}K')
                ax.axvline(0.1*E, linestyle='--', color=f'C{i}', linewidth=1, alpha=0.7)
            
            ax.set_yscale('log')
            ax.legend()
            
            filename = (
                f'{self.data}_l_{l}.pdf'
            )
            save_path = os.path.join(output_dir, filename)
            fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.01)
            plt.close(fig)
                