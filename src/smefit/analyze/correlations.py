# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_correlations(
    posterior_df,
    latex_names,
    fig_name,
    thr_show=None,
    hide_dofs=None,
    title=None,
    figsize=(10, 10),
):
    """
    Computes and displays the correlation coefficients
    between parameters in a heat map

    Parameters
    ----------
    posterior_df : pd.DataFrame
        fit results
    latex_names :  pd.DataFrame
        coefficnet latex name table
    fig_name : str
        path to save the plot
    thr_show: float, None
        if given shows only off diagonal entries higher than the threshold
    hide_dofs: list, None
        coefficients to hide
    title: str, None
        plot title
    figsize: tuple, (10, 10)
        Figure size
    """
    if hide_dofs is not None:
        posterior_df = posterior_df.drop(hide_dofs, axis=1)

    # get correlation of free parameters
    correlations = posterior_df[latex_names.index].corr()

    # Show only the values higher than a threshold
    if thr_show is not None:
        diag_corr = pd.DataFrame(
            np.eye(correlations.shape[0]),
            index=correlations.index,
            columns=correlations.columns,
        )
        correlations = correlations[np.abs(correlations) - diag_corr >= thr_show]
        correlations = correlations[correlations.sum() != 0]
        coeff_to_keep = correlations.index
        correlations = correlations.loc[:, coeff_to_keep]
        correlations = correlations.replace(np.nan, 0) + np.eye(correlations.shape[0])
    else:
        coeff_to_keep = correlations.index

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    cmap = plt.get_cmap("coolwarm")
    norm = colors.BoundaryNorm(np.arange(-1, 1.1, step=0.1), cmap.N)

    divider = make_axes_locatable(ax)
    cax = ax.matshow(correlations.values, cmap=cmap, norm=norm)
    fig.colorbar(cax, cax=divider.append_axes("right", size="5%", pad=0.1))

    for i, row in enumerate(correlations.values):
        for j, cij in enumerate(row):
            if thr_show is None or np.abs(cij) >= thr_show:
                ax.text(
                    i,
                    j,
                    f"{cij:.1f}",
                    va="center",
                    ha="center",
                    fontsize=10,
                )
            else:
                continue
    import pdb; pdb.set_trace()
    labels = latex_names[coeff_to_keep].values
    ticks = np.arange(labels.shape[0])
    ax.set_yticks(ticks, labels=labels, fontsize=15)
    ax.set_xticks(ticks, labels=labels, rotation=90, fontsize=15)
    ax.tick_params(which="major", top=False, bottom=False, left=False)

    # minor grid
    ax.set_xticks(ticks - 0.5, minor=True)
    ax.set_yticks(ticks - 0.5, minor=True)
    ax.tick_params(which="minor", bottom=False)
    ax.grid(visible=True, which="minor", alpha=0.2)

    if title is not None:
        ax.set_title(f"\\rm Correlation:\\ {title}", fontsize=25, y=-0.06)

    plt.tight_layout()
    plt.savefig(f"{fig_name}.pdf")
    plt.savefig(f"{fig_name}.png")
