# This code was written by cursor under my direction. I used the following prompts:
#Hi. Can you create a function which plots calibration? Specifically, the function should take as an input the number of bins and some data in the form of (p_i, y_i) where p_i is the predicted probability of y_i=1. As an output the function should return the actual probability of y given each bin. The funciton should be robust to certain bins being empty. Afterwards, can you create another funciton to plot this in matplotlib?
#Good first pass. I want the return value of calibration_bin_statistics to be more simple. Specifically, just return an n_bins-dimentional array with a number indicating the actual probability of y given the predicted probability is in the coorisponding bin. Intelegently handle empty bin cases. Also for plot calibration, I don't want anything to be returned. I just want to see a plot.
#I think you are misunderstanding the bins. Each data point (p_i, y_i) should be in ONE bin, specifically bin (AKA interval) coorisponding to the value of p_i.
#No I do not like where this is going. Start over. Start by simply going from taking inputs (a list of (p_i, y_i)s and an integer indicating the number of bins ) and converting it to a list of length n_bins where each entry in the array is the array of datapoints with coorisponding entries in that bin.
#Okay good job. Now can you make a function that takes this data as input and returns an array of the average values of p_i and y_i in each bin. And then another function that takes the list of average values of p_i and y_i per bin and plots them?
#can you write an example so I can see it?
#No do just like a toy example with like 10 specific numbers
# Okay I rewrote the start of this function so the specific bin is specified. The cannonical bin is given by get_bins(n_bins). But for a given input bin, could you redo this method?
#No. Just rewrite the one funciton I told you to.
"""Group samples by predicted probability into histogram bins on ``[0, 1]``."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

Pair = Tuple[float, float]
DataInput = Union[Sequence[Pair], np.ndarray]


def get_bins(n_bins: int) -> List[Tuple[float, float]]:
    rv = []
    for i in range(n_bins):
        rv.append((float(i/n_bins), float((i+1)/n_bins)))
    return rv

def points_per_bin(data: DataInput, bins: List[Tuple[float, float]]) -> List[np.ndarray]:
    """
    Each sample ``(p_i, y_i)`` goes into exactly one bin ``bins[k] = (lo, hi)``.
    Interior bins use ``[lo, hi)``; the last bin uses ``[lo, hi]`` so ``p == hi``
    is included. ``p`` is clipped to ``[bins[0][0], bins[-1][1]]``.
    """
    n_bins = len(bins)
    if n_bins < 1:
        raise ValueError("bins must be non-empty")

    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("data must be (n, 2): columns are p and y for each row")
    p = arr[:, 0]
    y = arr[:, 1]

    lo0, hi_last = bins[0][0], bins[-1][1]
    p_c = np.clip(p, lo0, hi_last)

    idx = np.empty(len(p_c), dtype=int)
    for j in range(len(p_c)):
        pv = float(p_c[j])
        placed = False
        for k in range(n_bins):
            lo, hi = bins[k]
            if k < n_bins - 1:
                if lo <= pv < hi:
                    idx[j] = k
                    placed = True
                    break
            else:
                if lo <= pv <= hi:
                    idx[j] = k
                    placed = True
                    break
        if not placed:
            raise ValueError(f"p={pv} does not fall in any bin; check bins cover [{lo0}, {hi_last}].")

    stacked = np.column_stack((p_c, y))
    out: List[np.ndarray] = []
    for k in range(n_bins):
        out.append(stacked[idx == k])
    return out


def per_bin_means(binned: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function takes as input a list of all the data partitioned into its cooresponding bin, 
    and for each bin, it outputs the average value of p_i, the average value of y_i, and the number of samples in that bin.
    """
    n_bins = len(binned)
    mean_p = np.full(n_bins, np.nan, dtype=float)
    mean_y = np.full(n_bins, np.nan, dtype=float)
    counts_per_bin = np.full(n_bins, 0, dtype=int)
    for k, pts in enumerate(binned):
        counts_per_bin[k] = pts.size
        if pts.size == 0:
            continue
        mean_p[k] = float(np.mean(pts[:, 0]))
        mean_y[k] = float(np.mean(pts[:, 1]))
    return mean_p, mean_y, counts_per_bin


def plot_per_bin_means(
    mean_p: np.ndarray,
    mean_y: np.ndarray,
    *,
    ax: Optional[plt.Axes] = None,
    draw_diagonal: bool = True,
    **plot_kwargs: Any,
) -> None:
    """
    Plot mean predicted probability vs mean ``y`` per bin (reliability diagram).

    Points with non-finite ``mean_p`` or ``mean_y`` (e.g. empty bins) are skipped.

    Parameters
    ----------
    mean_p, mean_y :
        Same length, typically from :func:`per_bin_means`.
    ax :
        Optional axes. If ``None``, a new figure is created and :func:`plt.show`
        is called at the end.
    draw_diagonal :
        If True, draw the ``y = x`` reference line.
    **plot_kwargs :
        Passed to ``ax.plot`` for the calibration curve.
    """
    mean_p = np.asarray(mean_p, dtype=float).ravel()
    mean_y = np.asarray(mean_y, dtype=float).ravel()
    if mean_p.shape != mean_y.shape:
        raise ValueError("mean_p and mean_y must have the same length")

    valid = np.isfinite(mean_p) & np.isfinite(mean_y)
    created = ax is None
    if created:
        _, ax = plt.subplots()

    plot_kwargs.setdefault("marker", "o")
    plot_kwargs.setdefault("linestyle", "-")
    if np.any(valid):
        ax.plot(mean_p[valid], mean_y[valid], **plot_kwargs)
    if draw_diagonal:
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Ideal")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Mean predicted p (bin)")
    ax.set_ylabel("Mean y (bin)")
    ax.grid(True, alpha=0.3)
    if draw_diagonal and ax.get_legend_handles_labels()[0]:
        ax.legend(loc="lower right")

    if created:
        plt.show()

def find_ECE(confidence, accuracy, counts_per_bin) -> float:
    " The average value of y_i in each bin is the accuracy, and the confidence is the average value of p_i"
    total = 0
    error = 0
    for i in range(len(counts_per_bin)):
        conf = confidence[i]
        acc = accuracy[i]
        count = counts_per_bin[i]
        if count > 0:
            error += count*abs(acc-conf)
            total += count
    return error/total

# # Toy data: 10 points
# data = np.array(
#     [
#         [0.05, 0.0],
#         [0.10, 1.0],
#         [0.15, 0.0],
#         [0.25, 1.0],
#         [0.35, 0.0],
#         [0.50, 1.0],
#         [0.55, 1.0],
#         [0.59, 0.0],
#         [0.90, 0.0],
#         [0.95, 0.0],
#     ]
# )
# n_bins = 5
# # bins = get_bins(n_bins=5)
# bins = [(0,0.2), (0.2,0.5), (0.5, 0.6), (0.6, 0.96)]
# binned = points_per_bin(data, bins=bins)
# mean_p, mean_y, counts = per_bin_means(binned)
# print("mean_p:", mean_p)
# print("mean_y:", mean_y)
# print(f"the ECE is {find_ECE(mean_p, mean_y, counts)}")
# plot_per_bin_means(mean_p, mean_y)

