import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import numpy as np
import pandas as pd

from kit.maths import calc_entropy
from kit.bioinf import AA1_STD, AA_CHARACTERISTICS_2, AA_CHARACTERISTICS_2a
from kit.bioinf.utils import calc_seq_aa_cnts
from kit.plot import spider_plot, plot_text, A4_width, A4_height

def plot_aa_dist(df_aa_probs, title=None, area=None, plot_background=True):
    if area is None:
        fig = plt.figure(figsize=(A4_width, 10 * A4_height / 32))
        ax_1 = fig.add_subplot(1, 3, 1)
        ax_2 = fig.add_subplot(1, 3, 2, projection='polar')
        ax_3 = fig.add_subplot(1, 3, 3, projection='polar')
    else:
        fig = plt.gcf()
        gs = area.subgridspec(1, 3, height_ratios=[1], width_ratios=[0.2, 1, 1], wspace=0.3, hspace=0.3)
        ax_1 = fig.add_subplot(gs[0, 0])
        ax_2 = fig.add_subplot(gs[0, 1], projection='polar')
        ax_3 = fig.add_subplot(gs[0, 2], projection='polar')

    assert np.allclose([df_aa_probs.iloc[:, i].sum() for i in range(len(df_aa_probs.columns))], 1., rtol=0.02)
    if plot_background:
        df_aa_probs['bg'] = 1 / len(AA1_STD)

    df_cat_probs = {}
    for c, aa in AA_CHARACTERISTICS_2.items():
        df_cat_probs[c] = list(df_aa_probs.loc[[_aa for _aa in aa]].sum(axis=0))
    df_cat_probs = pd.DataFrame(df_cat_probs).transpose()

    ylim_1 = np.sum(np.arange(0, 10) * 0.1 < df_aa_probs.max().max()) * 0.1
    spider_plot(df_aa_probs, ax=ax_2, ylim=(0, ylim_1), ysections=2)
    spider_plot(df_cat_probs, ax=ax_3, ylim=(0, 1.), ysections=2)

    c = df_aa_probs.columns[0]
    entropy = calc_entropy(list(df_aa_probs.loc[:, c]))
    polar = np.sum([df_aa_probs.loc[aa, c] for aa in AA_CHARACTERISTICS_2a['polar']])
    nonpolar = np.sum([df_aa_probs.loc[aa, c] for aa in AA_CHARACTERISTICS_2a['non-polar']])

    assert np.isclose(polar + nonpolar, 1., rtol=0.02)

    plot_text(f"{c}\nEntropy: {entropy:.2f}\nPolar: {polar:.2f}", ax_1, rotation=90, y_pos=0.5, x_pos=0.1,
              font_scale=0.65)


def plot_pwm_dist(df_pwm, positions, title=None):
    n_cols, n_rows = 1, len(positions)
    fig = plt.figure(figsize=(A4_width * 0.8, n_rows * A4_height / 4))
    if title is not None:
        fig.suptitle(title)
    gs = mpl.gridspec.GridSpec(n_rows, n_cols, height_ratios=[1] * n_rows, width_ratios=[1] * n_cols, wspace=0.3,
                               hspace=0.3)

    for r, position in enumerate(positions):
        plot_aa_dist(df_pwm.loc[:, [position]], area=gs[r, :])


def plot_spider_seqs_aa_dist(seqs):
    if isinstance(seqs, str):
        seqs = {'': seqs}

    aa_cnts = calc_seq_aa_cnts(list(seqs.values()), standardize=True, aggregate=False)
    aa_cnts = {a: aa_cnts[a] for a in AA1_STD}
    _df = pd.DataFrame(aa_cnts, index=list(seqs.keys())).transpose()

    n_cols = 1
    n_rows = 3
    fig = plt.figure(figsize=(A4_width, n_rows * A4_height / 4))
    gs = mpl.gridspec.GridSpec(
        n_rows,
        n_cols,
        height_ratios=[1] * n_rows,
        width_ratios=[1] * n_cols,
        wspace=0.3, hspace=0.3)

    for i, c in enumerate(_df.columns):
        plot_aa_dist(_df[[c]], area=gs[i, :])

