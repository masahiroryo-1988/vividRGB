"""Figure 7: RGB-edge F1 versus saved extraction duration and board efficiency.

Runs from the frozen 50-image result JSON alone, without model inference or
source photographs. Missing full-stage DINOv3 energy is never imputed from a
cached encoder-stage measurement.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DOMAINS = ['moin', 'fungal_network', 'neon', 'coralscapes', 'pmid']
NAMES = ['Grassland', 'Fungal networks', 'Airborne forest', 'Coral reefs', 'Phytoplankton']
COLORS = ['#217e70', '#997b35', '#467aaa', '#a25d91', '#cc6b46']
METHODS = ['whole', 'sam3', 's05', 's10']
LABELS = ['DINOv3', 'SAM3 automatic', 'FFA 5%', 'FFA 10%']
MARKERS = ['o', '^', 's', 'D']


def summary(values):
    values = np.asarray(values, dtype=float)
    return {'n': len(values), 'mean': float(values.mean()),
            'sd': float(values.std(ddof=1))}


def build_cost_tradeoff(rows, out, source_sha256=None):
    out = Path(out)
    (out / 'figures').mkdir(parents=True, exist_ok=True)
    (out / 'data').mkdir(parents=True, exist_ok=True)
    assert len(rows) == 50
    assert all(sum(r['dataset'] == d for r in rows) == 10 for d in DOMAINS)
    points = []
    for r in rows:
        for m in METHODS:
            result = r['methods'][m]
            score = (result['metrics']['quality'] if m == 'sam3'
                     else result['k']['6'])['rgb_edge']['f1']
            ex = result['extraction']
            # A full-stage reciprocal cannot be formed for the cached baseline.
            joules = None if m == 'whole' else ex['gpu_board_joules']
            assert ex['seconds'] > 0 and 0 <= score <= 1
            assert joules is None or joules > 0
            points.append({'image': r['id'], 'dataset': r['dataset'], 'method': m,
                           'rgb_edge_f1': score, 'extraction_seconds': ex['seconds'],
                           'gpu_board_joules': joules,
                           'gpu_board_efficiency_inverse_joules':
                           None if joules is None else 1 / joules})
    fields = list(points[0])
    with (out / 'data/cost_tradeoff_points.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(points)
    measures = ['rgb_edge_f1', 'extraction_seconds', 'gpu_board_joules',
                'gpu_board_efficiency_inverse_joules']

    def describe(selected):
        return {key: summary([p[key] for p in selected if p[key] is not None])
                if any(p[key] is not None for p in selected) else None
                for key in measures}

    evidence = {
        'source_sha256': source_sha256,
        'endpoint': 'Strong-RGB-edge boundary F1; appearance proxy, not biological accuracy',
        'cost_scope': 'Saved extraction-stage timings and GPU-board estimates; stages differ by implementation',
        'mean_efficiency': 'Mean of per-image 1/J, not 1/mean(J)',
        'missing_energy': {'method': 'whole', 'images': 50,
                           'reason': 'Full extraction-stage energy unavailable; cached encoding energy is not substituted'},
        'overall': {m: describe([p for p in points if p['method'] == m]) for m in METHODS},
        'domains': {d: {m: describe([p for p in points if p['dataset'] == d and p['method'] == m])
                        for m in METHODS} for d in DOMAINS},
        'plot_counts': {'duration_observations': 200, 'efficiency_observations': 150,
                        'duration_domain_means': 20, 'efficiency_domain_means': 15},
    }
    (out / 'data/cost_tradeoff_summary.json').write_text(
        json.dumps(evidence, indent=2), encoding='utf-8')

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 13.5,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42})
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 6.2), sharey=True)
    keys = ['extraction_seconds', 'gpu_board_efficiency_inverse_joules']
    for ax, key in zip(axs, keys):
        for d, color in zip(DOMAINS, COLORS):
            for m, marker in zip(METHODS, MARKERS):
                selected = [p for p in points if p['dataset'] == d and p['method'] == m
                            and p[key] is not None]
                if not selected:
                    continue
                x = np.asarray([p[key] for p in selected])
                y = np.asarray([p['rgb_edge_f1'] for p in selected])
                ax.scatter(x, y, marker=marker, color=color, s=17, alpha=.18,
                           linewidths=0, zorder=1)
                ax.scatter(x.mean(), y.mean(), marker=marker, color=color,
                           s=65, edgecolors='white', linewidths=.6, zorder=3)
        for m, label, marker in zip(METHODS, LABELS, MARKERS):
            s = evidence['overall'][m]
            if s[key] is None:
                continue
            x, y = s[key]['mean'], s['rgb_edge_f1']['mean']
            ax.scatter(x, y, marker=marker, s=115, color='#202c35',
                       edgecolors='white', linewidths=.9, zorder=4)
            # Annotate the pooled means; avoid the dense domain-mean clouds.
            offsets = ({'whole': (0, 30, 'right'), 'sam3': (8, 40, 'left'),
                        's05': (-8, 58, 'right'), 's10': (-16, 45, 'right')}
                       if key == 'extraction_seconds' else
                       {'sam3': (-8, 40, 'right'), 's05': (0, 58, 'left'),
                        's10': (-14, 42, 'right')})
            dx, dy, align = offsets[m]
            ax.annotate(label, (x, y), xytext=(dx, dy),
                        textcoords='offset points', ha=align,
                        va='bottom', fontsize=12, color='#202c35',
                        arrowprops={'arrowstyle': '-', 'color': '#64727b', 'lw': .7},
                        zorder=5)
        ax.set_xscale('log')
        ax.set_ylim(-.025, .88)
        ax.set_yticks(np.arange(0, .9, .2))
        ax.grid(axis='y', alpha=.16)
        ax.tick_params(axis='both', labelsize=12)
    axs[0].set_xlim(.08, 650)
    axs[1].set_xlim(3.8e-5, 5.6e-3)
    axs[0].set_title('A   Performance vs duration', loc='left', fontsize=14, weight='bold', pad=13)
    axs[1].set_title('B   Performance vs GPU efficiency', loc='left', fontsize=14, weight='bold', pad=13)
    axs[0].set_ylabel('RGB-edge boundary F1')
    axs[0].set_xlabel('Extraction duration (s; log scale)')
    axs[1].set_xlabel(r'GPU board efficiency (1/J; log scale)')
    axs[0].text(.03, .97, 'Better: higher F1, shorter duration', transform=axs[0].transAxes,
                va='top', fontsize=10.5, color='#48565f')
    axs[1].text(.03, .97, 'Better: higher F1, greater efficiency', transform=axs[1].transAxes,
                va='top', fontsize=10.5, color='#48565f')
    axs[1].text(.98, .46, 'DINOv3: full-stage\nenergy unavailable',
                transform=axs[1].transAxes, ha='right', va='top', fontsize=11,
                color='#48565f', bbox={'facecolor': 'white', 'edgecolor': '#d7dfe1',
                                      'boxstyle': 'round,pad=.4', 'alpha': .95})
    domain_handles = [Line2D([0], [0], color=c, marker='o', linestyle='None',
                            markersize=7, label=n) for c, n in zip(COLORS, NAMES)]
    method_handles = [Line2D([0], [0], color='#202c35', marker=mk, linestyle='None',
                            markersize=7, label=label) for mk, label in zip(MARKERS, LABELS)]
    fig.legend(handles=domain_handles, loc='lower center', bbox_to_anchor=(.5, .025),
               ncol=3, frameon=False, fontsize=11, columnspacing=1.6, handletextpad=.4)
    fig.legend(handles=method_handles, loc='lower center', bbox_to_anchor=(.5, .12),
               ncol=4, frameon=False, fontsize=11, columnspacing=1.5, handletextpad=.4)
    fig.text(.5, .22, 'Faint: individual images   |   Colored: domain means (n=10)   |   Black: all-image means (n=50)',
             ha='center', fontsize=11, color='#48565f')
    fig.subplots_adjust(left=.085, right=.985, bottom=.34, top=.91, wspace=.12)
    for ext in ['pdf', 'png']:
        fig.savefig(out / 'figures' / f'cost_tradeoff.{ext}', dpi=220, bbox_inches='tight')
    plt.close(fig)
    return evidence


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    evidence = build_cost_tradeoff(json.loads(args.source.read_text())['images'], args.out,
                                   hashlib.sha256(args.source.read_bytes()).hexdigest())
    print(json.dumps({'plot_counts': evidence['plot_counts'],
                      'overall': evidence['overall']}, indent=2))
