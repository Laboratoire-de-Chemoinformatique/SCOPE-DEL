from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

def plot_cdp(
    metrics: pd.DataFrame,
    size_by: str = "n_unique_mols",
    colors: Optional[Union[List[str], Dict[str, str]]] = None,
    label_ids: Optional[List[str]] = None,
    label_top_k: Optional[int] = None,
    title: str = "Consensus Diversity Plot (DELs)",
    figsize=(7, 6),
    savepath: Optional[str] = None,
    # style controls
    show_title: bool = False,
    title_size: int = 14,
    label_size: int = 12,
    tick_size: int = 10,
    # new parameters
    label_color: str = "red",
    axis_linewidth: float = 1.5,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    label_names: Optional[Dict[str, str]] = None,
    # numerical color mapping
    color_by: Optional[Union[str, List[float], np.ndarray]] = None,
    colormap: str = "viridis",
    show_colorbar: bool = True,
    colorbar_label: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    # NEW:
    ax=None,
    show: bool = True,
):
    """
    Simple customizable CDP scatter plot.

    If `ax` is provided, draw into that axes and DO NOT create a new figure.
    Only call plt.show() if show=True.
    Returns (ax, colorbar) where colorbar may be None.
    """
    import matplotlib.pyplot as plt

    # --- data prep (unchanged) ---
    x = metrics["median_intra_sim"].values
    y = metrics["f50"].values
    sizes = metrics[size_by].astype(float).values
    if np.nanmax(sizes) > 0:
        sizes = 40 + 160 * (sizes / np.nanmax(sizes))
    else:
        sizes = np.full_like(x, 60, dtype=float)

    labeled_mask = np.zeros(len(metrics), dtype=bool)
    if label_ids is not None:
        labeled_mask = metrics["identifier"].isin(label_ids).values
    elif label_top_k:
        top_indices = (
            metrics.sort_values(size_by, ascending=False).head(label_top_k).index
        )
        labeled_mask = metrics.index.isin(top_indices).values

    # --- figure/axes handling (CHANGED) ---
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True

    cbar = None

    # --- plotting (unchanged logic) ---
    if color_by is not None:
        if isinstance(color_by, str):
            if color_by not in metrics.columns:
                raise ValueError(f"Column '{color_by}' not found in metrics DataFrame")
            color_values = metrics[color_by].values
            default_colorbar_label = color_by
        else:
            color_values = np.array(color_by)
            if len(color_values) != len(metrics):
                raise ValueError(
                    f"Length of color_by values ({len(color_values)}) must match number of data points ({len(metrics)})"
                )
            default_colorbar_label = "Values"

        scatter_for_colorbar = None

        if np.any(~labeled_mask):
            scatter_unlabeled = ax.scatter(
                x[~labeled_mask],
                y[~labeled_mask],
                s=sizes[~labeled_mask],
                c=color_values[~labeled_mask],
                cmap=colormap,
                alpha=0.7,
                edgecolors="k",
                linewidths=0.5,
                vmin=vmin,
                vmax=vmax,
            )
            scatter_for_colorbar = scatter_unlabeled

        if np.any(labeled_mask):
            scatter_labeled = ax.scatter(
                x[labeled_mask],
                y[labeled_mask],
                s=sizes[labeled_mask],
                c=color_values[labeled_mask],
                cmap=colormap,
                alpha=0.7,
                edgecolors=label_color,
                linewidths=2.0,
                vmin=vmin,
                vmax=vmax,
            )
            # Prefer labeled for colorbar if it exists
            scatter_for_colorbar = scatter_labeled

        if show_colorbar and scatter_for_colorbar is not None:
            cbar = plt.colorbar(scatter_for_colorbar, ax=ax)
            cbar.set_label(
                (
                    colorbar_label
                    if colorbar_label is not None
                    else default_colorbar_label
                ),
                fontsize=label_size,
            )
            cbar.ax.tick_params(labelsize=tick_size)
    else:
        if np.any(~labeled_mask):
            unlabeled_colors = "C0"
            if colors is not None:
                if isinstance(colors, dict):
                    unlabeled_colors = [
                        colors.get(i, "C0")
                        for i in metrics.loc[~labeled_mask, "identifier"]
                    ]
                elif isinstance(colors, list) and len(colors) == len(metrics):
                    unlabeled_colors = [
                        colors[i] for i in range(len(colors)) if not labeled_mask[i]
                    ]
                else:
                    unlabeled_colors = colors
            ax.scatter(
                x[~labeled_mask],
                y[~labeled_mask],
                s=sizes[~labeled_mask],
                c=unlabeled_colors,
                alpha=0.7,
                edgecolors="k",
                linewidths=0.5,
            )
        if np.any(labeled_mask):
            ax.scatter(
                x[labeled_mask],
                y[labeled_mask],
                s=sizes[labeled_mask],
                c=label_color,
                alpha=0.7,
                edgecolors="k",
                linewidths=0.5,
            )

    ax.set_xlabel(
        (
            xlabel
            if xlabel is not None
            else "Median intra-library Tanimoto (Morgan count, r=2)"
        ),
        fontsize=label_size,
    )
    ax.set_ylabel(
        (
            ylabel
            if ylabel is not None
            else "F50 (fraction of scaffolds to reach 50% coverage)"
        ),
        fontsize=label_size,
    )
    if show_title:
        ax.set_title(title, fontsize=title_size)
    ax.tick_params(labelsize=tick_size)
    ax.grid(True, alpha=0.3)
    for spine in ax.spines.values():
        spine.set_linewidth(axis_linewidth)

    def get_display_name(identifier: str) -> str:
        if label_names is not None and identifier in label_names:
            return label_names[identifier]
        else:
            return str(identifier).replace("_1", "")

    if label_ids is not None:
        sub = metrics[metrics["identifier"].isin(label_ids)]
        for _, row in sub.iterrows():
            ax.annotate(
                get_display_name(row["identifier"]),
                (row["median_intra_sim"], row["f50"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=tick_size,
            )

    if label_top_k:
        top = metrics.sort_values(size_by, ascending=False).head(label_top_k)
        for _, row in top.iterrows():
            ax.annotate(
                get_display_name(row["identifier"]),
                (row["median_intra_sim"], row["f50"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=tick_size,
            )

    if len(metrics) > 1:
        xm = np.nanmedian(x)
        ym = np.nanmedian(y)
        ax.axvline(xm, linestyle="--", alpha=0.25, color="black")
        ax.axhline(ym, linestyle="--", alpha=0.25, color="black")

    if savepath and created_fig:
        plt.tight_layout()
        plt.savefig(savepath)

    if show and created_fig:
        plt.tight_layout()
        plt.show()

    return ax, cbar
