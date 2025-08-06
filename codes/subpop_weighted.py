#!/usr/bin/env python3

"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Plots of deviation of a subpop. from the full pop., with weighted sampling

*
This implementation considers responses r that can take arbitrary values,
not necesssarily restricted to taking values 0 or 1.
*

Functions
---------
cumulative
    Cumulative difference between observations from a subpop. & the full pop.
equiscores
    Reliability diagram with roughly equispaced average scores over bins
equierrs
    Reliability diagram with similar ratio L2-norm / L1-norm of weights by bin
exactplot
    Reliability diagram with exact values plotted

This source code is licensed under the MIT license found in the LICENSE file in
the root directory of this source tree.
"""

import math
import os
import subprocess
import numpy as np
from numpy.random import default_rng
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter


def cumulative(r, s, inds, majorticks, minorticks, bernoulli=True,
               filename='cumulative.pdf',
               title='subpop. deviation is the slope as a function of $A_k$',
               fraction=1, weights=None):
    """
    Cumulative difference between observations from a subpop. & the full pop.

    Saves a plot of the difference between the normalized cumulative weighted
    sums of r for the subpopulation indices inds and the normalized cumulative
    weighted sums of r from the full population interpolated to the subpop.
    indices, with majorticks major ticks and minorticks minor ticks on the
    lower axis, labeling the major ticks with the corresponding values from s.

    Parameters
    ----------
    r : array_like
        random outcomes
    s : array_like
        scores (must be in non-decreasing order)
    inds : array_like
        indices of the subset within s that defines the subpopulation
        (must be unique and in strictly increasing order)
    majorticks : int
        number of major ticks on each of the horizontal axes
    minorticks : int
        number of minor ticks on the lower axis
    bernoulli : bool, optional
        set to True (the default) for Bernoulli variates; set to False
        to use empirical estimates of the variance rather than the formula
        p(1-p) for a Bernoulli variate whose mean is p
    filename : string, optional
        name of the file in which to save the plot
    title : string, optional
        title of the plot
    fraction : float, optional
        proportion of the full horizontal axis to display
    weights : array_like, optional
        weights of the observations
        (the default None results in equal weighting)

    Returns
    -------
    float
        Kuiper statistic
    float
        Kolmogorov-Smirnov statistic
    float
        quarter of the full height of the isosceles triangle
        at the origin in the plot
    float
        ordinate (vertical coordinate) at the greatest abscissa (horizontal
        coordinate)
    """

    def histcounts(nbins, a):
        # Counts the number of entries of a
        # falling into each of nbins equispaced bins.
        j = 0
        nbin = np.zeros(nbins, dtype=np.int64)
        for k in range(len(a)):
            if a[k] > a[-1] * (j + 1) / nbins:
                j += 1
            if j == nbins:
                break
            nbin[j] += 1
        return nbin

    def aggregate(r, s, ss, w):
        # Determines the weighted mean and variance of the entries of r
        # in a bin around each entry of s corresponding to the subset ss of s.
        # The bin ranges from halfway to the nearest entry of s in ss
        # on the left to halfway to the nearest entry of s in ss on the right.
        q = np.insert(np.append(ss, [1e20]), 0, [-1e20])
        t = np.asarray([(q[k] + q[k + 1]) / 2 for k in range(len(q) - 1)])
        rc = np.zeros((len(ss)))
        rc2 = np.zeros((len(ss)))
        sc = np.zeros((len(ss)))
        sc2 = np.zeros((len(ss)))
        j = 0
        for k in range(len(s)):
            if s[k] > t[j + 1]:
                j += 1
                if j == len(ss):
                    break
            if s[k] >= t[0]:
                sc[j] += w[k]
                sc2[j] += w[k]**2
                rc[j] += w[k] * r[k]
                rc2[j] += w[k] * r[k]**2
        means = rc / sc
        # Calculate an adjustment factor for the estimate of the variance
        # that will make the estimate unbiased.
        unbias = sc**2
        unbias[sc**2 == sc2] = 0
        unbias[sc**2 != sc2] /= (sc**2 - sc2)[sc**2 != sc2]
        return means, unbias * (rc2 / sc - means**2)

    assert all(s[k] <= s[k + 1] for k in range(len(s) - 1))
    assert all(inds[k] < inds[k + 1] for k in range(len(inds) - 1))
    # Determine the weighting scheme.
    if weights is None:
        w = np.ones((len(s)))
    else:
        w = weights.copy()
    assert np.all(w > 0)
    w /= w.sum()
    # Create the figure.
    plt.figure()
    ax = plt.axes()
    # Subsample s, r, and w.
    ss = s[inds]
    rs = r[inds]
    ws = w[inds]
    # Average the results and weights for repeated scores, while subsampling
    # the scores and indices for the subpopulation. Also calculate factors
    # for adjusting the variances of responses to account for the responses
    # being averages of other responses (when the scores need not be unique).
    sslist = []
    rslist = []
    wslist = []
    fslist = []
    rssum = 0
    wssum = 0
    wssos = 0
    for k in range(len(ss)):
        rssum += rs[k] * ws[k]
        wssum += ws[k]
        wssos += ws[k]**2
        if k == len(ss) - 1 or not math.isclose(
                ss[k], ss[k + 1], rel_tol=1e-14):
            sslist.append(ss[k])
            rslist.append(rssum / wssum)
            wslist.append(wssum)
            fslist.append(wssos / wssum**2)
            rssum = 0
            wssum = 0
            wssos = 0
    ss = np.asarray(sslist)
    rs = np.asarray(rslist)
    ws = np.asarray(wslist)
    fs = np.asarray(fslist)
    # Normalize the weights.
    ws /= ws[:int(len(ws) * fraction)].sum()
    # Aggregate r according to s, ss, and w.
    rt, rtvar = aggregate(r, s, ss, w)
    # Accumulate the weighted rs and rt, as well as ws.
    f = np.insert(np.cumsum(ws * rs), 0, [0])
    ft = np.insert(np.cumsum(ws * rt), 0, [0])
    x = np.insert(np.cumsum(ws), 0, [0])
    # Plot the difference.
    plt.plot(
        x[:int(len(x) * fraction)], (f - ft)[:int(len(f) * fraction)], 'k')
    # Make sure the plot includes the origin.
    plt.plot(0, 'k')
    # Add an indicator of the scale of 1/sqrt(n) to the vertical axis.
    rtsub = np.insert(rt, 0, [0])[:(int(len(rt) * fraction) + 1)]
    if bernoulli:
        lenscale = np.sqrt(np.sum(ws**2 * rtsub[1:] * (1 - rtsub[1:]) * fs))
    else:
        lenscale = np.sqrt(np.sum(ws**2 * rtvar * fs))
    plt.plot(2 * lenscale, 'k')
    plt.plot(-2 * lenscale, 'k')
    kwargs = {
        'head_length': 2 * lenscale, 'head_width': fraction / 20, 'width': 0,
        'linewidth': 0, 'length_includes_head': True, 'color': 'k'}
    plt.arrow(.1e-100, -2 * lenscale, 0, 4 * lenscale, shape='left', **kwargs)
    plt.arrow(.1e-100, 2 * lenscale, 0, -4 * lenscale, shape='right', **kwargs)
    plt.margins(x=0, y=.1)
    # Label the major ticks of the lower axis with the values of ss.
    lenxf = int(len(x) * fraction)
    sl = ['{:.2f}'.format(a) for a in
          np.insert(ss, 0, [0])[:lenxf:(lenxf // majorticks)].tolist()]
    plt.xticks(
        x[:lenxf:(lenxf // majorticks)], sl,
        bbox=dict(boxstyle='Round', fc='w'))
    if len(rtsub) >= 300 and minorticks >= 50:
        # Indicate the distribution of s via unlabeled minor ticks.
        plt.minorticks_on()
        ax.tick_params(which='minor', axis='x')
        ax.tick_params(which='minor', axis='y', left=False)
        ax.set_xticks(x[np.cumsum(histcounts(minorticks,
                      ss[:int((len(x) - 1) * fraction)]))], minor=True)
    # Label the axes.
    plt.xlabel('$S_k$', labelpad=6)
    plt.ylabel('$B_k$')
    ax2 = plt.twiny()
    plt.xlabel(
        '$k/n$ (together with minor ticks at equispaced values of $A_k$)',
        labelpad=8)
    ax2.tick_params(which='minor', axis='x', top=True, direction='in', pad=-17)
    ax2.set_xticks(np.arange(1 / majorticks, 1, 1 / majorticks), minor=True)
    ks = ['{:.2f}'.format(a) for a in
          np.arange(0, 1 + 1 / majorticks / 2, 1 / majorticks).tolist()]
    alist = (lenxf - 1) * np.arange(0, 1 + 1 / majorticks / 2, 1 / majorticks)
    alist = alist.tolist()
    # Jitter minor ticks that overlap with major ticks lest Pyplot omit them.
    alabs = []
    for a in alist:
        multiple = x[int(a)] * majorticks
        if abs(multiple - round(multiple)) > multiple * 1e-3 / 2:
            alabs.append(x[int(a)])
        else:
            alabs.append(x[int(a)] * (1 + 1e-3))
    plt.xticks(alabs, ks, bbox=dict(boxstyle='Round', fc='w'))
    ax2.xaxis.set_minor_formatter(FixedFormatter(
        [r'$A_k\!=\!{:.2f}$'.format(1 / majorticks)]
        + [r'${:.2f}$'.format(k / majorticks) for k in range(2, majorticks)]))
    # Title the plot.
    plt.title(title)
    # Clean up the whitespace in the plot.
    plt.tight_layout()
    # Save the plot.
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    # Calculate summary statistics.
    fft = (f - ft)[:int(len(f) * fraction)]
    kuiper = np.max(fft) - np.min(fft)
    kolmogorov_smirnov = np.max(np.abs(fft))
    maxint = int(len(f) * fraction) - 1
    return kuiper, kolmogorov_smirnov, lenscale, f[maxint] - ft[maxint]


def equiscores(r, s, inds, nbins, filename='equiscore.pdf', weights=None,
               top=None, left=None, right=None):
    """
    Reliability diagram with roughly equispaced average scores over bins

    Plots a reliability diagram with roughly equispaced average scores
    for the bins, for both the full population and the subpopulation specified
    by the indices inds.

    Parameters
    ----------
    r : array_like
        random outcomes
    s : array_like
        scores (must be in non-decreasing order)
    inds : array_like
        indices of the subset within s that defines the subpopulation
        (must be unique and in strictly increasing order)
    nbins : int
        number of bins
    filename : string, optional
        name of the file in which to save the plot
    weights : array_like, optional
        weights of all observations
        (the default None results in equal weighting)
    top : float, optional
        top of the range of the vertical axis (the default None is adaptive)
    left : float, optional
        leftmost value of the horizontal axis (the default None is adaptive)
    right : float, optional
        rightmost value of the horizontal axis (the default None is adaptive)

    Returns
    -------
    None
    """

    def bintwo(nbins, a, b, q, qmax, w):
        # Determines the total weight of entries of q falling into each
        # of nbins equispaced bins, and calculates the weighted average per bin
        # of the arrays a and b, returning np.nan as the "average"
        # for any bin that is empty.
        j = 0
        bina = np.zeros(nbins)
        binb = np.zeros(nbins)
        wbin = np.zeros(nbins)
        for k in range(len(q)):
            if q[k] > qmax * (j + 1) / nbins:
                j += 1
            if j == nbins:
                break
            bina[j] += w[k] * a[k]
            binb[j] += w[k] * b[k]
            wbin[j] += w[k]
        # Normalize the sum for each bin to compute the arithmetic average.
        bina = np.divide(bina, wbin, where=wbin != 0)
        bina[wbin == 0] = np.nan
        binb = np.divide(binb, wbin, where=wbin != 0)
        binb[wbin == 0] = np.nan
        return wbin, bina, binb

    assert all(s[k] <= s[k + 1] for k in range(len(s) - 1))
    assert all(inds[k] < inds[k + 1] for k in range(len(inds) - 1))
    # Determine the weighting scheme.
    if weights is None:
        w = np.ones((len(s)))
    else:
        w = weights.copy()
    assert np.all(w > 0)
    w /= w.sum()
    ws = w[inds]
    ws /= ws.sum()
    # Create the figure.
    plt.figure()
    _, binr, binst = bintwo(nbins, r, s, s, s[inds[-1]], w)
    _, binrs, binss = bintwo(nbins, r[inds], s[inds], s[inds], s[inds[-1]], ws)
    plt.plot(binst, binr, '*:', color='gray')
    plt.plot(binss, binrs, '*:', color='black')
    xmin = min(binst[0], binss[0]) if left is None else left
    xmax = max(binst[-1], binss[-1]) if right is None else right
    plt.xlim((xmin, xmax))
    plt.ylim(bottom=0)
    plt.ylim(top=top)
    plt.xlabel('weighted average of $S_k$ for $k$ in the bin')
    plt.ylabel('weighted average of $R_k$ for $k$ in the bin')
    plt.title('reliability diagram')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def equierrs(r, s, inds, nbins, rng, filename='equibins.pdf', weights=None,
             top=None, left=None, right=None):
    """
    Reliability diagram with similar ratio L2-norm / L1-norm of weights by bin

    Plots a reliability diagram with the ratio of the L2 norm of the weights
    to the L1 norm of the weights being roughly the same for every bin.
    The L2 norm is the square root of the sum of the squares, while the L1 norm
    is the sum of the absolute values. The plot includes a graph both for the
    full population and for the subpopulation specified by the indices inds.

    Parameters
    ----------
    r : array_like
        random outcomes
    s : array_like
        scores (must be in non-decreasing order)
    inds : array_like
        indices of the subset within s that defines the subpopulation
        (must be unique and in strictly increasing order)
    nbins : int
        rough number of bins to construct
    rng : Generator
        fully initialized random number generator from NumPy
    filename : string, optional
        name of the file in which to save the plot
    weights : array_like, optional
        weights of all observations
        (the default None results in equal weighting)
    top : float, optional
        top of the range of the vertical axis (the default None is adaptive)
    left : float, optional
        leftmost value of the horizontal axis (the default None is adaptive)
    right : float, optional
        rightmost value of the horizontal axis (the default None is adaptive)

    Returns
    -------
    int
        number of bins constructed for the subpopulation
    int
        number of bins constructed for the full population
    """

    def inbintwo(a, b, inbin, w):
        # Determines the total weight falling into the bins given by inbin,
        # and calculates the weighted average per bin of the arrays a and b,
        # returning np.nan as the "average" for any bin that is empty.
        wbin = [w[inbin[k]:inbin[k + 1]].sum() for k in range(len(inbin) - 1)]
        bina = [(w[inbin[k]:inbin[k + 1]] * a[inbin[k]:inbin[k + 1]]).sum()
                for k in range(len(inbin) - 1)]
        binb = [(w[inbin[k]:inbin[k + 1]] * b[inbin[k]:inbin[k + 1]]).sum()
                for k in range(len(inbin) - 1)]
        # Normalize the sum for each bin to compute the weighted average.
        bina = np.divide(bina, wbin, where=wbin != 0)
        bina[wbin == 0] = np.nan
        binb = np.divide(binb, wbin, where=wbin != 0)
        binb[wbin == 0] = np.nan
        return wbin, bina, binb

    def binbounds(nbins, w):
        # Partitions w into around nbins bins, each with roughly equal ratio
        # of the L2 norm of w in the bin to the L1 norm of w in the bin,
        # returning the indices defining the bins in the list inbin.
        proxy = len(w) // nbins
        v = w[np.sort(rng.permutation(len(w))[:proxy])]
        # t is a heuristic threshold.
        t = np.square(v).sum() / v.sum()**2
        inbin = []
        k = 0
        while k < len(w) - 1:
            inbin.append(k)
            k += 1
            s = w[k]
            ss = w[k]**2
            while ss / s**2 > t and k < len(w) - 1:
                k += 1
                s += w[k]
                ss += w[k]**2
        if len(w) - inbin[-1] < (inbin[-1] - inbin[-2]) / 2:
            inbin[-1] = len(w)
        else:
            inbin.append(len(w))
        return inbin

    assert all(s[k] <= s[k + 1] for k in range(len(s) - 1))
    assert all(inds[k] < inds[k + 1] for k in range(len(inds) - 1))
    # Determine the weighting scheme.
    if weights is None:
        w = np.ones((len(s)))
    else:
        w = weights.copy()
    assert np.all(w > 0)
    w /= w.sum()
    inbin = binbounds(nbins, w)
    ws = w[inds]
    ws /= ws.sum()
    inbins = binbounds(nbins, ws)
    # Create the figure.
    plt.figure()
    _, binr, binst = inbintwo(r, s, inbin, w)
    _, binrs, binss = inbintwo(r[inds], s[inds], inbins, ws)
    plt.plot(binst, binr, '*:', color='gray')
    plt.plot(binss, binrs, '*:', color='black')
    xmin = min(binst[0], binss[0]) if left is None else left
    xmax = max(binst[-1], binss[-1]) if right is None else right
    plt.xlim((xmin, xmax))
    plt.ylim(bottom=0)
    plt.ylim(top=top)
    plt.xlabel('weighted average of $S_k$ for $k$ in the bin')
    plt.ylabel('weighted average of $R_k$ for $k$ in the bin')
    title = r'reliability diagram'
    title += r' ($\Vert W \Vert_2 / \Vert W \Vert_1$ is similar for every bin)'
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

    return len(inbins) - 1, len(inbin) - 1


def exactplot(r, s, inds, filename='exact.pdf', title='exact expectations',
              top=None, left=None, right=None):
    """
    Reliability diagram with exact values plotted

    Plots a reliability diagram at full resolution with fractional numbers,
    for both the full population and the subpop. specified by indices inds.
    The entries of r should be the expected values of outcomes,
    even if the outcomes are integer-valued counts or just 0s and 1s.

    Parameters
    ----------
    r : array_like
        expected value of outcomes
    s : array_like
        scores (must be in non-decreasing order)
    inds : array_like
        indices of the subset within s that defines the subpopulation
        (must be unique and in strictly increasing order)
    filename : string, optional
        name of the file in which to save the plot
    title : string, optional
        title of the plot
    top : float, optional
        top of the range of the vertical axis (the default None is adaptive)
    left : float, optional
        leftmost value of the horizontal axis (the default None is adaptive)
    right : float, optional
        rightmost value of the horizontal axis (the default None is adaptive)

    Returns
    -------
    None
    """
    assert all(s[k] <= s[k + 1] for k in range(len(s) - 1))
    assert all(inds[k] < inds[k + 1] for k in range(len(inds) - 1))
    plt.figure()
    plt.plot(s, r, '*', color='gray')
    rs = r[inds]
    ss = s[inds]
    plt.plot(ss, rs, '*', color='black')
    plt.xlim((left, right))
    plt.ylim(bottom=0)
    plt.ylim(top=top)
    plt.xlabel('score $S_k$')
    plt.ylabel('expected value ($P_k$) of outcome $R_k$')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    #
    # Generate directories with plots as specified via the code below,
    # with each directory named m_len(inds)_nbins_iex-dithered or
    # m_len(inds)_nbins_iex-averaged (where m, inds, nbins, and iex
    # are defined in the code below, and "dithered" uses scores dithered
    # to become distinct, while "averaged" uses responses averaged together
    # at the same score).
    #
    # Set parameters.
    # minorticks is the number of minor ticks on the lower axis.
    minorticks = 100
    # majorticks is the number of major ticks on the lower axis.
    majorticks = 10
    # m is the number of members from the full population.
    m = 50000
    # n determines the number of observations for the subpopulation.
    n = 2500
    # Store processes for converting from pdf to jpeg in procs.
    procs = []
    # Consider 4 examples.
    for iex in range(4):
        for dither in [True, False]:
            # nbins is the number of bins for the reliability diagrams.
            for nbins in [10, 50]:
                # nbins must divide n evenly.
                assert n % nbins == 0

                if iex == 0:
                    # Define the indices of the subset for the subpopulation.
                    inds = np.arange(0, m, m // n) + m // n // 2
                    inds1 = np.arange(0, m // 4, m // n // 4) + m // 2 - m // 8
                    inds2 = np.arange(0, m // 2, m // n // 2) + m // 2 - m // 4
                    inds3 = np.arange(0, m, m // n)
                    inds = np.concatenate((inds1, inds1 + 1, inds2, inds3))
                    # Indices must be sorted and unique.
                    inds = np.unique(inds)
                    inds = inds[0:(m // (m // len(inds)))]

                    # Construct scores.
                    sl = np.arange(0, 1, 4 / m) + 2 / m
                    s = np.square(sl)
                    s = np.concatenate([s] * 4)
                    if dither:
                        ss = s.shape
                        rng = default_rng(seed=987654321)
                        s *= np.ones(ss) + rng.normal(size=ss) * 1e-8
                    # The scores must be in non-decreasing order.
                    s = np.sort(s)

                    # Construct perturbations to the scores for sampling rates.
                    d = .25
                    tl = -np.arange(-d, d, 2 * d / m) - d / m
                    t = d - 1.1 * np.square(np.square(tl)) / d**3
                    e = .7
                    ul = -np.arange(-e, e, 2 * e / m) - e / m
                    u = e - np.abs(ul)
                    ins = np.arange(m // 2 - m // 50, m // 2 + m // 50)
                    u[ins] = t[ins]
                    u2 = 2 * t - u
                    t += np.sin(np.arange((m))) * (u - t)

                    # Construct the exact sampling probabilities.
                    exact = s + t
                    exact[inds] = s[inds] + u[inds]
                    exact[inds + 1] = s[inds] + u2[inds]

                    # Construct weights.
                    weights = 4 - np.cos(9 * np.arange(m) / m)

                if iex == 1:
                    # Define the indices of the subset for the subpopulation.
                    rng = default_rng(seed=987654321)
                    inds = np.sort(rng.permutation((m))[:n])

                    # Construct scores.
                    s = np.arange(0, 1, 2 / m) + 1 / m
                    s = np.sqrt(s)
                    s = np.concatenate((s, s))
                    if dither:
                        ss = s.shape
                        s *= np.ones(ss) + rng.normal(size=ss) * 1e-8
                    # The scores must be in non-decreasing order.
                    s = np.sort(s)

                    # Construct perturbations to the scores for sampling rates.
                    d = math.sqrt(1 / 2)
                    tl = np.arange(-d, d, 2 * d / m) - d / m
                    t = (1 + np.sin(np.arange((m)))) / 2
                    t *= np.square(tl) - d**2
                    u = np.square(tl) - d**2
                    u *= .75 * np.round(1 + np.sin(10 * np.arange((m)) / m))
                    u /= 2

                    # Construct the exact sampling probabilities.
                    exact = s + t
                    exact[inds] = s[inds] + u[inds]

                    # Construct weights.
                    weights = 4 - np.cos(9 * np.arange(m) / m)

                if iex == 2:
                    # Define the indices of the subset for the subpopulation.
                    rng = default_rng(seed=987654321)
                    inds = np.arange(0, m ** (3 / 4), 1)
                    inds = np.unique(np.round(np.power(inds, 4 / 3)))
                    inds = inds.astype(int)
                    inds = inds[0:(50 * (len(inds) // 50))]

                    # Construct scores.
                    s = np.arange(0, 1, 10 / m) + 5 / m
                    s = np.concatenate([s] * 10)
                    if dither:
                        ss = s.shape
                        s *= np.ones(ss) + rng.normal(size=ss) * 1e-8
                    # The scores must be in non-decreasing order.
                    s = np.sort(s)

                    # Construct perturbations to the scores for sampling rates.
                    tl = np.arange(0, 1, 1 / m) + 1 / (2 * m)
                    t = np.power(tl, 1 / 4) - tl
                    t *= (1 + np.sin(np.arange((m)))) / 2
                    u = np.power(tl, 1 / 4) - tl
                    u *= .5 * (1 + np.sin(
                        50 * np.power(np.arange(0, m**4, m**3), 1 / 4) / m))

                    # Construct the exact sampling probabilities.
                    exact = s + t
                    exact[inds] = s[inds] + u[inds]

                    # Construct weights.
                    weights = 4 - np.cos(9 * np.arange(m) / m)

                if iex == 3:
                    # Define the indices of the subset for the subpopulation.
                    rng = default_rng(seed=987654321)
                    inds = np.sort(rng.permutation((m))[:n])

                    # Construct scores.
                    s = np.arange(0, 1, 4 / m) + 2 / m
                    s = np.concatenate([s] * 4)
                    if dither:
                        ss = s.shape
                        s *= np.ones(ss) + rng.normal(size=ss) * 1e-8
                    # The scores must be in non-decreasing order.
                    s = np.sort(s)

                    # Construct the exact sampling probabilities.
                    exact = np.sin(np.arange(m))
                    exact *= np.sin(50 * np.arange(-3 * m / 4, m / 4) / m)
                    exact = np.square(exact)
                    exact /= 5
                    exact[inds] = 0

                    # Construct weights.
                    weights = np.ones((m))
                    ind = 3 * m // 4 - 1
                    # Identify an index near the middle that belongs
                    # to the subpop. for which the two adjacent indices do not.
                    while (
                            np.any(inds == (ind - 1))
                            or not np.any(inds == ind)
                            or np.any(inds == (ind + 1))):
                        ind += 1
                    weights[ind] = n / 50
                    weights[ind - 1] = m / 500
                    weights[ind + 1] = m / 500

                    # Alter the exact sampling probabilities for the 3 indices
                    # selected in the preceding "while" loop.
                    exact[ind] = 1
                    exact[ind - 1] = 1
                    exact[ind + 1] = 0

                # Set a unique directory for each collection of experiments
                # (creating the directory if necessary).
                dir = 'weighted'
                try:
                    os.mkdir(dir)
                except FileExistsError:
                    pass
                dir = 'weighted/' + str(m) + '_' + str(len(inds))
                dir = dir + '_' + str(nbins)
                dir = dir + '_' + str(iex)
                dir += '-'
                if dither:
                    dir += 'dithered'
                else:
                    dir += 'averaged'
                try:
                    os.mkdir(dir)
                except FileExistsError:
                    pass
                dir = dir + '/'
                print(f'./{dir} is under construction....')

                # Generate a sample of classifications into two classes,
                # correct (class 1) and incorrect (class 0),
                # avoiding numpy's random number generators
                # that are based on random bits --
                # they yield strange results for many seeds.
                rng = default_rng(seed=987654321)
                uniform = np.asarray([rng.random() for _ in range(m)])
                r = (uniform <= exact).astype(float)

                # Generate five plots and a text file reporting metrics.
                filename = dir + 'cumulative.pdf'
                kuiper, kolmogorov_smirnov, lenscale, ordinate = cumulative(
                    r, s, inds, majorticks, minorticks, True, filename,
                    weights=weights)
                filename = dir + 'metrics.txt'
                with open(filename, 'w') as f:
                    f.write('n:\n')
                    f.write(f'{len(inds)}\n')
                    f.write('number of unique scores in the subset:\n')
                    f.write(f'{len(np.unique(s[inds]))}\n')
                    f.write('lenscale:\n')
                    f.write(f'{lenscale}\n')
                    f.write('Kuiper:\n')
                    f.write(f'{kuiper:.4}\n')
                    f.write('Kolmogorov-Smirnov:\n')
                    f.write(f'{kolmogorov_smirnov:.4}\n')
                    f.write('Kuiper / lenscale:\n')
                    f.write(f'{(kuiper / lenscale):.4}\n')
                    f.write('Kolmogorov-Smirnov / lenscale:\n')
                    f.write(f'{(kolmogorov_smirnov / lenscale):.4}\n')
                    f.write('ordinate:\n')
                    f.write(f'{ordinate:.4}\n')
                filename = dir + 'cumulative_exact.pdf'
                _, _, _, _ = cumulative(
                    exact, s, inds, majorticks, minorticks, True, filename,
                    title='exact expectations', weights=weights)
                filename = dir + 'equiscores.pdf'
                equiscores(r, s, inds, nbins, filename, weights, top=1, left=0,
                           right=1)
                filename = dir + 'equierrs.pdf'
                rng = default_rng(seed=987654321)
                equierrs(r, s, inds, nbins + 3, rng, filename, weights, top=1,
                         left=0, right=1)
                filepdf = dir + 'exact.pdf'
                filejpg = dir + 'exact.jpg'
                exactplot(exact, s, inds, filepdf, top=1, left=0, right=1)
                args = ['convert', '-density', '1200', filepdf, filejpg]
                procs.append(subprocess.Popen(args))
    print('waiting for conversion from pdf to jpg to finish....')
    for iproc, proc in enumerate(procs):
        proc.wait()
        print(f'{iproc + 1} of {len(procs)} conversions are done....')
