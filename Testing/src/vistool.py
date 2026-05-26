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
from matplotlib.ticker import NullLocator, NullFormatter
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
from scipy.stats import ttest_ind
from matplotlib.patches import Patch
from dataclasses import dataclass
from nltk.corpus import wordnet as wn
import warnings
from functools import lru_cache

warnings.filterwarnings(
    "ignore",
    message="No WordNet synset found for pos="
)


@dataclass
class Vector:
    v: np.ndarray
    epop: np.ndarray
    rpop: np.ndarray
    color: str

class VisTool():
    def __init__(self, trn_count_info, tst_count_info, trains, tests, nentity, nrelation, total_degree, rel_prob, data):
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        self.trn_count_info = trn_count_info
        self.tst_count_info = tst_count_info
        self.markers = {'RotatE': 'o', 'ComplEx': 's', 'HousE':'^', 'TuckER':'v', 'pLogicNet':'D',
                        'RNNLogic':'p', 'CompGCN':'*'}
        self.mods = {'RotatE':'#073b4c', 'ComplEx':'#118ab2','HousE':'#06d6a0', 'TuckER':'#FFB715',
                      'pLogicNet':'#F78C6B', 'RNNLogic':'#ef476f','CompGCN':'#80B563'}

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

    def draw__hyperPlane(self, mets: List[List[PROBE]], alphas: list, betas: list, gammas: List, mode='ab',
                        vis_axis=False, vis_cbar=False):
        if mode == 'ab':
            target1, target2 = alphas, betas
        elif mode == 'ag':
            target1, target2 = alphas, gammas
        elif mode == 'bg':
            target1, target2 = betas, gammas
        else:
            raise ValueError(f"Unknown mode: {mode}")

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
                        if mode == 'ab':
                            values.append(met.calculate_final_metric(alpha, beta, 0.0))
                        elif mode == 'ag':
                            values.append(met.calculate_final_metric(alpha, 0.0, beta))
                        elif mode == 'bg':
                            values.append(met.calculate_final_metric(1.0, alpha, beta))

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
        levels = np.linspace(zmin, zmax, 256)  # adjust 12

        os.makedirs(f'../figs/2D_hyper', exist_ok=True)
        os.makedirs(f'../figs/2D_hyper/contour', exist_ok=True)
        os.makedirs(f'../figs/2D_hyper/hyper', exist_ok=True)
        contour_pdf_path = f'../figs/2D_hyper/contour/{mets[0][0].data}_contour_{mode}.pdf'

        def _is_YAGO():
            return self.data == 'YAGO3-10'

        def _is_FB():
            return self.data == 'FB15k237'

        nrow, ncol, mag = 1, 6, 1.3
        fig_row, axes = plt.subplots(
            nrow, ncol,
            figsize=(ncol * mag, nrow * mag),
            constrained_layout=True
        )
        axes = np.array(axes).reshape(-1)

        if num_models == 1:
            axes = [axes]

        last_csf = None
        cmap_scheme = 'RdYlBu_r'
        for m_idx, ax2 in enumerate(axes):
            fig, ax2 = plt.subplots(figsize=(mag, mag), constrained_layout=True)
            if _is_YAGO() and m_idx + 1 == len(axes):
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

            if not (m_idx == 5 and _is_YAGO()):
                ax2.axvline(x_mid, color='grey', lw=1, ls='--')
                ax2.axhline(y_mid, color='grey', lw=1, ls='--')

            if m_idx == 5 and _is_YAGO():
                ax2.text(
                    0.5, 0.5,
                    "OOM",
                    ha='center',
                    va='center',
                    fontsize=15,
                    transform=plt.gca().transAxes
                )

            if m_idx == 0:
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
                    xy=(0.9, 0.3),  # up
                    xytext=origin,
                    arrowprops=dict(arrowstyle='-|>', lw=1, fc='black'),
                    xycoords='axes fraction'
                )

                # labels
                ax2.text(0.69, 0.1, r'$\alpha$', transform=ax2.transAxes, ha='right', va='center')
                ax2.text(0.9, 0.31, r'$\beta$', transform=ax2.transAxes, ha='center', va='bottom')

            ax2.tick_params(axis='both', which='both', length=0)
            ax2.set_xticks(np.arange(n1))
            ax2.set_yticks(np.arange(n2))
 
            if vis_axis:
                if m_idx >= 0:
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
            if (_is_YAGO() and m_idx + 1 != len(axes)) or (not _is_YAGO()): last_csf = csf
            if _is_YAGO() and m_idx == 5: save_path = f'../figs/fig11/{self.data}_RNNLogic_contour_ab.pdf'
            else: save_path = f'../figs/fig11/{self.data}_{model_name_ls[m_idx]}_contour_ab.pdf'
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


        def in_bin(cnt, bin_idx):  # [,)
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
                score = met.calculate_final_metric(alpha, beta, 0.0)
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


        # compute some helpers first
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
                if is_bin: # hatch=hatch_dict[models[i]],
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
                        ecolor='black',  # match line color
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

    def draw_for_table(self, mets: List[List[PROBE]], alpha, beta, gamma, target='a', figw=2, figh=1.5):
        data = mets[0][0].data


        ret_dict = {'a':alpha, 'b':beta, 'g':gamma}
        name_dict = {'a': 'alpha', 'b': 'beta', 'g': 'gamma'}

        colors = {_mets[0].model: self.mods[_mets[0].model] for _mets in mets}

        model_results = {}
        for _mets in mets:
            model_name = _mets[0].model
            scores = [met.calculate_final_metric(alpha, beta, gamma) for met in _mets]
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
 


    def write_tables(self, mets: List[List[PROBE]], alphas, betas, gammas, mode='a'):
        models = [met[0].model for met in mets]
        data = mets[0][0].data
        match mode:
            case 'a': piv_ls, other = betas, alphas
            case 'b': piv_ls, other = alphas, betas
            case 'g': piv_ls, other = alphas, gammas

        latex = defaultdict(str)

        for p_idx, piv in enumerate(piv_ls):
            results = [[] for _ in range(len(other))]
            colors = [self.mods[_mets[0].model] for _mets in mets]

            all_mean_y = []

            raw_model_scores = []  
            for idx, _mets in enumerate(mets):
                calculated_metrics = []
                for met in _mets:
                    if mode == 'a':
                        values = [met.calculate_final_metric(alpha, piv, 0.0) for alpha in alphas]
                    elif mode == 'b':
                        values = [met.calculate_final_metric(piv, beta, 0.0) for beta in betas]
                    elif mode == 'g':
                        values = [met.calculate_final_metric(piv, 0.0, gamma) for gamma in gammas]
                    calculated_metrics.append(values)

                calculated_metrics = np.array(calculated_metrics)
                mean_y = np.mean(calculated_metrics, axis=0)
                std_y = np.std(calculated_metrics, axis=0)
                all_mean_y.append((mean_y, std_y))
                raw_model_scores.append(mean_y)

            for idx, (mean_y, std_y) in enumerate(all_mean_y):
                norm_y = raw_model_scores[idx]
                for i in range(len(other)):
                    results[i].append((mets[idx][0].model, round(norm_y[i], 4)))

            for i, ls in enumerate(results):
                ls = sorted(ls, key=lambda x: x[1], reverse=True)

                for rank, (model, score) in enumerate(ls):
                    rank = rank + 1
                    color_style = f'\\cellcolor{{rank{rank}}}'

                    latex[model] += f' & {color_style}{{{score:.4f}}}'

        if mode == 'a':
            folder_name = 'table4'
        elif mode == 'b':
            folder_name = 'table5'
        else:
            print(f'Not supported mode name: {mode}')
            return
        
        os.makedirs(f'../figs/{folder_name}', exist_ok=True)
        for model in models:
            txt_name = f'{model}_tables_{mode}.txt'
            if not os.path.exists(f'../figs/{folder_name}/{txt_name}'):
                with open(f'../figs/{folder_name}/{txt_name}', 'w') as f:
                    f.write(model)
            with open(f'../figs/{folder_name}/{txt_name}', 'a') as f:
                f.write(''.join(latex[model]) + '\n')


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
        #ax.set_xticklabels(x_labels, rotation=45, fontsize=10)
        # if cut_last: ax.set_xticklabels(['1st','2nd~5th','6th~20th','21st~50th','51st~100th','101st~200th'], fontsize=15, rotation=24)
        # else: ax.set_xticklabels(['1st','2nd~5th','6th~20th','21st~50th','51st~100th','101st~200th','Others'], fontsize=15)
        xtick_labels = []
        for i, cut in enumerate(cuts):
            if i == len(cuts) - 1: break
            if cuts[i + 1] - cut == 1: xtick_labels.append(f'{cut}')
            else:
                if cuts[-1] == cuts[i + 1]: xtick_labels.append(f'{cut}-{cuts[i + 1]}')
                else: xtick_labels.append(f'{cut}-{cuts[i + 1] - 1}')
        # if cut_last and len(cuts) == 6:
        #     #ax.set_xticklabels(['1st','2nd-5th','6th-20th','21st-50th','51st-100th'], fontsize=axis_font)
        #     ax.set_xticklabels(['1', '2-5', '6-20', '21-50', '51-100'], fontsize=axis_font)
        # elif cut_last and len(cuts) == 7: ax.set_xticklabels(['1', '2-5', '6-20', '21-50', '51-100', '101-200'], fontsize=axis_font)
        # else:
        #     #ax.set_xticklabels(['1st','2nd~5th','6th~20th','21st~50th','51st~100th','101st~200th','Others'], fontsize=axis_font)
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
            sep_transformed_ranks = np.array([transformed_ranks[:split].mean(), transformed_ranks[split:].mean()])

            EW = 1 / (eps + vec.epop) ** b
            EW = EW / np.mean(EW)
            norm_EW = EW / np.sum(EW)

            RW = 1 / (eps + vec.rpop) ** b
            RW = RW / np.mean(RW)
            norm_RW = RW / np.sum(RW)

            W = norm_EW * norm_RW
            norm_W = W / np.sum(W)
            sep_norm_W = np.array([norm_W[:split].sum(), norm_W[split:].sum()])
            return sep_transformed_ranks, sep_norm_W, np.dot(transformed_ranks, norm_W)


        dim_alpha = 0.3
        if a == 1 and b == 0: # base case
            bt_alpha = 1.0
            bw_alpha = 1.0
        elif b == 0: # less pred sharp
            bt_alpha = dim_alpha
            bw_alpha = 1.0
        elif a == 1: # pop robust
            bt_alpha = 1.0
            bw_alpha = dim_alpha
        else: # both
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
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        we = [0.5, 0.5, 0.1]
        wr = [0.8, 0.8, 0.25]
        A = Vector(np.array([1, 2, 300]), np.array(we), np.array(wr), '#DE0005') # red
        B = Vector(np.array([2, 3, 10]), np.array(we), np.array(wr), '#3F991F') # blue
        C = Vector(np.array([4, 4, 5]), np.array(we), np.array(wr),'#002ADE') # green

        vs = [A, B, C]

        for a, b in [[1,0],[0.25,0],[1,0.8]]:
            VisTool.draw_vector_custom(vs, a, b, 1000, 1e-5, split=2)

    @staticmethod
    def draw_owa():
        def draw_rank_change_box(
                ax,
                top_indices,
                bottom_indices,
                model_order,
                model_wise_colors,
                top_label="f/ test",
                bottom_label="s/ test",
                anchor=(0.98, 0.98), 
                box_width=0.6,
                box_height=0.5,
                label_frac=0.23, 
                square_size_frac=0.4, 
                connector_lw=1.2,
                frame_lw=0.8,
                fontsize=18,
                row_gap=0.28
        ):
            x_right, y_top = anchor
            x0 = x_right - box_width
            y0 = y_top - box_height

            frame = Rectangle(
                (x0, y0), box_width, box_height,
                transform=ax.transAxes,
                fill=False,
                ec="grey",
                lw=frame_lw,
                alpha=0.2,
                clip_on=False,
                zorder=10
            )
            ax.add_patch(frame)

            label_w = box_width * label_frac
            content_x0 = x0 + label_w
            content_x1 = x0 + box_width - 0.02 * box_width
            content_w = content_x1 - content_x0

            y_top_row = y0 + box_height * (row_gap + 0.5)
            y_bottom_row = y0 + box_height * (0.5 - row_gap)

            n = len(top_indices)
            if len(bottom_indices) != n:
                raise ValueError("top_indices and bottom_indices must have same length")

            xs = [
                content_x0 + content_w * (i + 0.5) / n
                for i in range(n)
            ]

            row_gap = abs(y_top_row - y_bottom_row)
            sq = row_gap * square_size_frac

            ax.text(
                x0 + 0.04 * box_width, y_top_row,
                top_label,
                transform=ax.transAxes,
                ha="left", va="center",
                fontsize=fontsize,
                zorder=12,fontstyle='italic'
            )
            ax.text(
                x0 + 0.04 * box_width, y_bottom_row,
                bottom_label,
                transform=ax.transAxes,
                ha="left", va="center",
                fontsize=fontsize,
                zorder=12,fontstyle='italic'
            )

            top_pos = {}
            bottom_pos = {}

            for rank, idx in enumerate(top_indices):
                model = model_order[idx]
                color = model_wise_colors[model]
                xc = xs[rank]
                top_pos[model] = xc

                ax.add_patch(Rectangle(
                    (xc - sq / 2, y_top_row - sq / 2),
                    sq, sq,
                    transform=ax.transAxes,
                    facecolor=color,
                    edgecolor=color,
                    lw=0.8,
                    clip_on=False,
                    zorder=13
                ))

            for rank, idx in enumerate(bottom_indices):
                model = model_order[idx]
                color = model_wise_colors[model]
                xc = xs[rank]
                bottom_pos[model] = xc

                ax.add_patch(Rectangle(
                    (xc - sq / 2, y_bottom_row - sq / 2),
                    sq, sq,
                    transform=ax.transAxes,
                    facecolor=color,
                    edgecolor=color,
                    lw=0.8,
                    clip_on=False,
                    zorder=13
                ))

            for model, x_top in top_pos.items():
                x_bottom = bottom_pos[model]
                ax.add_line(Line2D(
                    [x_top, x_bottom],
                    [y_top_row - sq / 2, y_bottom_row + sq / 2],
                    transform=ax.transAxes,
                    color=model_wise_colors[model],
                    lw=connector_lw,
                    alpha=0.8,
                    zorder=11,
                    clip_on=False
                ))

        def change_metric_name(name):
            if 'PROBE' in name: return metric.replace('a=', r"$\alpha$=")
            if 'HITS' in name: return name
            if name == 'sqrt': return '0.5-MRR'
            if name == 'MRR': return '1.0-MRR'
            if name == 'log': return 'log-MRR'
            a, b = int(name[1]), int(name[-1])
            p = round(a/b, 2)
            return f'{p}-MRR'

        model_wise_colors = {'RotatE1':'#073B4C',
                           'RotatE2':'#009999',
                           'RotatE3':'#6062B8',
                           'RotatE4':'#0099FF',
                           'RotatE5':'#2149C9',
                           'ComplEx1':'#996633',
                           'ComplEx2':'#808000',
                           'ComplEx3':'#FF9900',
                           'ComplEx4':'#A50021',
                           'ComplEx5':'#CCCC00',}

        met_wise_colors = {'MRR':'#000000',
                            'log':'#5F5F5F',
                            'sqrt':'#9E9E9E',
                            'P1_4':'#18186F',
                            'P1_3':'#0000CC',
                            'P2_3':'#6394EC',
                            'P3_4':'#00BFBF',
                            'HITS1':'#9F512C',
                            'HITS3':'#7F7F00',
                            'HITS10':'#BFBF00',
                            'PROBE(a=-1.0)':'#660033',
                            'PROBE(a=-0.5)':'#CC0066',
                            'PROBE(a=0.0)':'#FF0066',
                            'PROBE(a=0.5)':'#FF6600',
                            'PROBE(a=1.0)':'#FF9933',
                            'PROBE(a=2.0)':'#FFCC00',}

        fontsize = 11

        files = os.listdir(f'../data/owa')
        owa_model_ls = defaultdict(list)
        owa_metric_ls = dict()
        model_order = []
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        best = None
        txt_file = f'../figs/fig12/latex'
        os.makedirs(txt_file, exist_ok=True)

        for file in files:

            model = file.split('.')[0]
            model_order.append(model)

            with open(os.path.join('../data/owa', file), 'r') as f:
                temp = json.load(f)

            for metric, ls in temp.items():
                owa_model_ls[metric].append(ls)

            if file == 'RotatE1.json':
                for metric, ls in temp.items():
                    owa_metric_ls[metric] = ls

        # metric wise
        for metric, ls in owa_model_ls.items():
            with open(f'{txt_file}/{metric}.txt', 'w') as f:
                pass
            model_full_test = list()
            model_sparse_test = list()

            x_max = 0
            x_min = 1
            y_max = 0
            y_min = 1
            fig, ax = plt.subplots(figsize=(3,1.5))

            for model, _ls in zip(model_order, ls):
                x, y = [], []
                _ls = sorted(_ls, key=lambda x:x[0])
                for xy in _ls:
                    x.append(xy[0])
                    y.append(xy[1])
                ax.plot(x, y, label=metric, color=model_wise_colors[model], linewidth=0.5)
                max_full, max_sparse = max(_ls, key=lambda x:x[1])
                model_full_test.append(round(max_full, 6))
                model_sparse_test.append(round(max_sparse, 6))
                _max = max(x) if max(x) > x_max else x_max
                x_min = min(x) if min(x) < x_min else x_min
                y_max = max(y) if max(y) > y_max else y_max
                y_min = min(y) if min(y) < y_min else y_min

            sort_kind = 'mergesort'
            full_sort_idx = np.argsort(-np.array(model_full_test), kind=sort_kind)
            sparse_sort_idx = np.argsort(-np.array(model_sparse_test), kind=sort_kind)
           
            cx = Counter(model_full_test)
            cx = dict(sorted(cx.items(), reverse=True))
            new = []
            for f_val, cnt in cx.items():
                target_idxs = set([i for i, s in enumerate(model_full_test) if s == f_val])
                assert cnt == len(target_idxs)
                target_sparse_order = [i for i in sparse_sort_idx if (i in target_idxs)]
                new.extend(target_sparse_order)
            full_sort_idx = new
            
            with open(f'{txt_file}/{metric}.txt', 'w') as f:
                f.write('full')
                for fi in full_sort_idx:
                    model = model_order[fi]
                    f.write(f' & \\tikz \\fill[fill={model.lower()}] (0,0) rectangle (\\boxsize,\\boxsize);')
                f.write('\\\\')
                f.write('sparse')
                for si in sparse_sort_idx:
                    model = model_order[si]
                    f.write(f' & \\tikz \\fill[fill={model.lower()}] (0,0) rectangle (\\boxsize,\\boxsize);')
                f.write('\\\\')

            if max_sparse > 0.4: ax.set_yticks([round(i*0.2, 1) for i in range(6)])
            else: ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
            plt.xlim(0.0 if x_min < 0.05 else x_min, 1.04)
            plt.ylim(0.0 if y_min < 0.05 else y_min, 1.02 if y_max >= 0.94 else y_max + 0.01)
            ax.tick_params(axis='both', labelsize=fontsize-2)
            replace_name = change_metric_name(metric)
            plt.title(f'{replace_name}', fontsize=fontsize)

            fig.savefig(f'../figs/fig12/{metric}.pdf', dpi=300, bbox_inches='tight', pad_inches=0.01)
         
        linestyle_ls = {'MRR':'-.',
                        'log':'-.',
                        'sqrt':'-.',
                        'P1_4':':',
                        'P1_3':':',
                        'P2_3':':',
                        'P3_4':':',
                        'HITS1':'--',
                        'HITS3':'--',
                        'HITS10':'--',
                        'PROBE(a=-1.0)':'-',
                        'PROBE(a=-0.5)':'-',
                        'PROBE(a=0.0)':'-',
                        'PROBE(a=0.5)':'-',
                        'PROBE(a=1.0)':'-',
                        'PROBE(a=2.0)':'-',}
        fig, ax = plt.subplots(figsize=(5,4))
        for idx, (metric, ls) in enumerate(owa_metric_ls.items()):
            x, y = [], []
            ls = sorted(ls, key=lambda x: x[0])
            for xy in ls:
                x.append(xy[0])
                y.append(xy[1])
            ax.plot(x, y, label=metric, color=met_wise_colors[metric], linewidth=1.5, linestyle=linestyle_ls[metric], alpha=1.0 if 'PROBE' in metric else 1)
        ax.legend(fontsize=6.5)
        plt.xlim(0, 1.02)
        plt.ylim(0, 1.02)
        plt.xlabel('(full test set)', fontsize=fontsize)
        plt.ylabel('(sparse test set)', fontsize=fontsize)
        fig.savefig(f'../figs/fig12/metric_wise.pdf', bbox_inches='tight', pad_inches=0.01)

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

                if syn is None:  # <-- important
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
            # store relation as edge label
            G.add_edge(h, t, label=r_name)

        # 2. Layout
        # after you finish building G
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
                node_sizes.append(500)  # larger
                node_colors.append(node_target_col)
            else:
                node_sizes.append(200)  # default
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
            idx = {"i": 0}  # mutable counter

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