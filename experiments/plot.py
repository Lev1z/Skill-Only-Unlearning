"""Standalone exportable figures; all numbers read from saved analyses."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiments.run_bemula import OUT


def main():
    m = json.loads((OUT / 'analysis.json').read_text())['official_metrics']
    cs = json.loads((OUT / 'cs2_analysis.json').read_text())
    arms = ['request_only', 'append', 'priority', 'constrained', 'delete_scope', 'rewrite_scope', 'clean']
    names = ['Request only', 'Append', 'Priority', '+ Constraints', 'Delete / filter', 'Rewrite', 'Graph-clean / replay']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
    xs = np.arange(len(arms))
    axes[0].bar(xs - .18, [100*m[a]['FE'] for a in arms], .36, label='FE', color='#5279ad')
    axes[0].bar(xs + .18, [100*m[a]['AR']['average'] for a in arms], .36, label='AR', color='#d68c3b')
    axes[0].set_title('Bemula: 50 profiles, 541 selected probes')
    axes[0].set_xticks(xs, names, rotation=35, ha='right')
    cs_arms = ['append', 'priority', 'constrained', 'clean', 'clean_seed', 'clean_constrained']
    cs_names = ['Append', 'Priority', '+ Constraints', 'Clean', 'Clean + priority', 'Clean + constraints']
    xs = np.arange(len(cs_arms))
    safety = [100*(1-cs[a]['historical_extraction']['leaks']/cs[a]['historical_extraction']['n']) for a in cs_arms]
    utility = [100*cs[a]['normal_topic_utility']['correct']/cs[a]['normal_topic_utility']['n'] for a in cs_arms]
    axes[1].bar(xs-.18, safety, .36, label='No old relation confirmed', color='#3d927b')
    axes[1].bar(xs+.18, utility, .36, label='Normal CS2 utility', color='#98a6b8')
    coverage = json.loads((OUT / 'cs2_coverage.json').read_text())
    axes[1].set_title(f"CS2: {len(coverage['complete_paired_profiles'])} complete template variants, judge v2")
    axes[1].set_xticks(xs, cs_names, rotation=35, ha='right')
    for ax in axes:
        ax.set_ylim(0, 108)
        ax.set_ylabel('Percent (higher is better)')
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(loc='upper left', fontsize=8)
    fig.suptitle('Added policy memory helps behavior; priority alone is insufficient', fontsize=14)
    for ext in ['svg', 'png', 'pdf']:
        fig.savefig(OUT / ('comparison.' + ext), dpi=180)


if __name__ == '__main__':
    main()
