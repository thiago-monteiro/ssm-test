from __future__ import annotations

import numpy as np


def bar_metric(
    ax, frame, metric, groups, colors, *, group="mode", title, ylabel, ylim=None
):
    if metric in frame.columns:
        values = [frame[frame[group] == value][metric] for value in groups]
        ax.bar(
            groups,
            [v.mean() for v in values],
            yerr=[v.sem() for v in values],
            color=[colors.get(value, "#333") for value in groups],
            capsize=4,
        )
    ax.set(title=title, ylabel=ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.tick_params(axis="x", rotation=45)


def errorbar_metric(
    ax,
    frame,
    metric,
    groups,
    colors,
    labels,
    *,
    group="normalized",
    x="quant_label",
    marker="o",
    linestyle="-",
    label_suffix="",
):
    for value in groups:
        grouped = frame[frame[group] == value].groupby(x, observed=True)[metric]
        means, sems = grouped.mean(), grouped.sem()
        ax.errorbar(
            np.arange(len(means)),
            means.values,
            yerr=sems.values,
            marker=marker,
            label=labels[value] + label_suffix,
            color=colors[value],
            capsize=3,
            linestyle=linestyle,
        )


def finish_figure(fig, path, *, dpi=150):
    import matplotlib.pyplot as plt

    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
