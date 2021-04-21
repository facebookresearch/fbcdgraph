#!/usr/bin/env python3

"""
Copyright (c) Facebook, Inc. and its affiliates.

Calibration plots, both cumulative and traditional, with weighted sampling

Functions
---------
cumulative
    Cumulative difference between observed and expected vals of Bernoulli vars
equiprob
    Reliability diagram with roughly equispaced average probabilities over bins
equierr
    Reliability diagram with similar ratio L2-norm / L1-norm of weights by bin
exactdist
    Reliability diagram with exact distributions plotted

This source code is licensed under the MIT license found in the LICENSE file in
the root directory of this source tree.
"""

import math
import os
import random
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter


def cumulative(r, s, majorticks, minorticks, filename='cumulative.pdf',
               title='miscalibration is the slope as a function of $A_k$',
               fraction=1, weights=None):
    """
    Cumulative difference between observed and expected vals of Bernoulli vars

    Saves a plot of the difference between the normalized cumulative
    weighted sums of r and the normalized cumulative weighted sums of s,
    with majorticks major ticks and minorticks minor ticks on the lower axis,
    labeling the major ticks with the corresponding values from s.

    Parameters
    ----------
    r : array_like
        class labels (0 for incorrect and 1 for correct classification)
    s : array_like
        success probabilities (must be unique and in strictly increasing order)
    majorticks : int
        number of major ticks on each of the horizontal axes
    minorticks : int
        number of minor ticks on the lower axis
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
    """

    def histcounts(nbins, a):
        # Counts the number of entries of a
        # falling into each of nbins equispaced bins.
        j = 0
        nbin = np.zeros(nbins, dtype=np.int64)
        for k in range(len(a)):
            if a[k] > (j + 1) / nbins:
                j += 1
            if j == nbins:
                break
            nbin[j] += 1
        return nbin

    assert all(s[k] < s[k + 1] for k in range(len(s) - 1))
    # Determine the weighting scheme.
    if weights is None:
        w = np.ones((len(s)))
    else:
        w = weights.copy()
    assert np.all(w > 0)
    w /= w[:int(len(w) * fraction)].sum()
    # Create the figure.
    plt.figure()
    ax = plt.axes()
    # Accumulate the weighted r and s, as well as w.
    f = np.insert(np.cumsum(w * r), 0, [0])
    ft = np.insert(np.cumsum(w * s), 0, [0])
    x = np.insert(np.cumsum(w), 0, [0])
    # Plot the difference.
    plt.plot(
        x[:int(len(x) * fraction)], (f - ft)[:int(len(f) * fraction)], 'k')
    # Make sure the plot includes the origin.
    plt.plot(0, 'k')
    # Add an indicator of the scale of 1/sqrt(n) to the vertical axis.
    ssub = np.insert(s, 0, [0])[:(int(len(s) * fraction) + 1)]
    lenscale = np.sqrt(np.sum(w**2 * ssub[1:] * (1 - ssub[1:])))
    plt.plot(lenscale, 'k')
    plt.plot(-lenscale, 'k')
    kwargs = {
        'head_length': lenscale, 'head_width': fraction / 20, 'width': 0,
        'linewidth': 0, 'length_includes_head': True, 'color': 'k'}
    plt.arrow(.1e-100, -lenscale, 0, 2 * lenscale, shape='left', **kwargs)
    plt.arrow(.1e-100, lenscale, 0, -2 * lenscale, shape='right', **kwargs)
    plt.margins(x=0, y=.1)
    # Label the major ticks of the lower axis with the values of s.
    ss = ['{:.2f}'.format(a) for a in
          ssub[::(len(ssub) // majorticks)].tolist()]
    lenxf = int(len(x) * fraction)
    plt.xticks(x[:lenxf:(lenxf // majorticks)], ss)
    if len(ssub) >= 300 and minorticks >= 50:
        # Indicate the distribution of s via unlabeled minor ticks.
        plt.minorticks_on()
        ax.tick_params(which='minor', axis='x')
        ax.tick_params(which='minor', axis='y', left=False)
        ax.set_xticks(x[np.cumsum(histcounts(minorticks, ssub[1:]))],
                      minor=True)
    # Label the axes.
    plt.xlabel('$S_k$')
    plt.ylabel('$F_k - \\tilde{F}_k$')
    ax2 = plt.twiny()
    plt.xlabel(
        '$k/n$ (together with minor ticks at equispaced values of $A_k$)')
    ax2.tick_params(which='minor', axis='x', top=True, direction='in', pad=-17)
    ax2.set_xticks(np.arange(0, 1 + 1 / majorticks, 1 / majorticks),
                   minor=True)
    ks = ['{:.2f}'.format(a) for a in
          np.arange(0, 1 + 1 / majorticks, 1 / majorticks).tolist()]
    alist = (lenxf - 1) * np.arange(0, 1 + 1 / majorticks, 1 / majorticks)
    alist = alist.tolist()
    plt.xticks([x[int(a)] for a in alist], ks)
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
    return kuiper, kolmogorov_smirnov, lenscale


def equiprob(r, s, nbins, filename='equiprob.pdf', n_resamp=0, weights=None):
    """
    Reliability diagram with roughly equispaced average probabilities over bins

    Plots a reliability diagram with roughly equispaced average probabilities
    for the bins.

    Parameters
    ----------
    r : array_like
        class labels (0 for incorrect and 1 for correct classification)
    s : array_like
        success probabilities (must be in non-decreasing order)
    nbins : int
        number of bins
    filename : string, optional
        name of the file in which to save the plot
    n_resamp : int, optional
        number of times to resample and plot an extra line for error bars
    weights : array_like, optional
        weights of the observations
        (the default None results in equal weighting)

    Returns
    -------
    None
    """

    def bintwo(nbins, a, b, q, w):
        # Determines the total weight of entries of q falling into each
        # of nbins equispaced bins, and calculates the weighted average per bin
        # of the arrays a and b, returning np.nan as the "average"
        # for any bin that is empty.
        j = 0
        bina = np.zeros(nbins)
        binb = np.zeros(nbins)
        wbin = np.zeros(nbins)
        for k in range(len(q)):
            if q[k] > (j + 1) / nbins:
                j += 1
            if j == nbins:
                break
            bina[j] += w[k] * a[k]
            binb[j] += w[k] * b[k]
            wbin[j] += w[k]
        # Normalize the sum for each bin to compute the arithmetic average.
        bina = np.divide(bina, wbin, where=wbin != 0)
        bina[np.where(wbin == 0)] = np.nan
        binb = np.divide(binb, wbin, where=wbin != 0)
        binb[np.where(wbin == 0)] = np.nan
        return wbin, bina, binb

    assert all(s[k] <= s[k + 1] for k in range(len(s) - 1))
    # Determine the weighting scheme.
    if weights is None:
        w = np.ones((len(s)))
    else:
        w = weights.copy()
    assert np.all(w > 0)
    w /= w.sum()
    # Create the figure.
    plt.figure()
    for _ in range(n_resamp):
        # Resample from s, preserving the pairing of s with r and w.
        srw = np.asarray([[s[k], r[k], w[k]] for k in
                          np.random.randint(0, len(s), (len(s)))])
        perm = np.argsort(srw[:, 0])
        ss = srw[perm, 0]
        rs = srw[perm, 1]
        ws = srw[perm, 2]
        _, binrs, binss = bintwo(nbins, rs, ss, ss, ws)
        # Use the light gray, "gainsboro".
        plt.plot(binss, binrs, 'gainsboro')
    _, binr, bins = bintwo(nbins, r, s, s, w)
    # Use the solid black, "k".
    plt.plot(bins, binr, 'k*:')
    zeroone = np.asarray((0, 1))
    plt.plot(zeroone, zeroone, 'k')
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xlabel('weighted average of $S_k$ for $k$ in the bin')
    plt.ylabel('weighted average of $R_k$ for $k$ in the bin')
    plt.title('reliability diagram')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def equierr(r, s, nbins, filename='equierr.pdf', n_resamp=0, weights=None):
    """
    Reliability diagram with similar ratio L2-norm / L1-norm of weights by bin

    Plots a reliability diagram with the ratio of the L2 norm of the weights
    to the L1 norm of the weights being roughly the same for every bin.
    The L2 norm is the square root of the sum of the squares, while the L1 norm
    is the sum of the absolute values.

    Parameters
    ----------
    r : array_like
        class labels (0 for incorrect and 1 for correct classification)
    s : array_like
        success probabilities (must be in non-decreasing order)
    nbins : int
        rough number of bins to construct
    filename : string, optional
        name of the file in which to save the plot
    n_resamp : int, optional
        number of times to resample and plot an extra line for error bars
    weights : array_like, optional
        weights of the observations
        (the default None results in equal weighting)

    Returns
    -------
    int
        number of bins constructed
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
        bina[np.where(wbin == 0)] = np.nan
        binb = np.divide(binb, wbin, where=wbin != 0)
        binb[np.where(wbin == 0)] = np.nan
        return wbin, bina, binb

    def binbounds(nbins, w):
        # Partitions w into around nbins bins, each with roughly equal ratio
        # of the L2 norm of w in the bin to the L1 norm of w in the bin,
        # returning the indices defining the bins in the list inbin.
        proxy = len(w) // nbins
        v = w[np.sort(np.random.permutation(len(w))[:proxy])]
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
    # Determine the weighting scheme.
    if weights is None:
        w = np.ones((len(s)))
    else:
        w = weights.copy()
    assert np.all(w > 0)
    w /= w.sum()
    inbin = binbounds(nbins, w)
    # Create the figure.
    plt.figure()
    for _ in range(n_resamp):
        # Resample from s, preserving the pairing of s with r and w.
        srw = np.asarray([[s[k], r[k], w[k]] for k in
                          np.random.randint(0, len(s), (len(s)))])
        perm = np.argsort(srw[:, 0])
        ss = srw[perm, 0]
        rs = srw[perm, 1]
        ws = srw[perm, 2]
        _, binrs, binss = inbintwo(rs, ss, inbin, ws)
        # Use the light gray, "gainsboro".
        plt.plot(binss, binrs, 'gainsboro')
    _, binr, bins = inbintwo(r, s, inbin, w)
    # Use the solid black, "k".
    plt.plot(bins, binr, 'k*:')
    zeroone = np.asarray((0, 1))
    plt.plot(zeroone, zeroone, 'k')
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xlabel('weighted average of $S_k$ for $k$ in the bin')
    plt.ylabel('weighted average of $R_k$ for $k$ in the bin')
    title = r'reliability diagram'
    title += r' ($\Vert W \Vert_2 / \Vert W \Vert_1$ is similar for every bin)'
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

    return len(inbin) - 1


def exactdist(r, s, filename='exact.pdf'):
    """
    Reliability diagram with exact distributions plotted

    Plots a reliability diagram at full resolution with fractional numbers.
    The entries of r should be the expected values of class labels,
    not necessarily just 0s and 1s.

    Parameters
    ----------
    r : array_like
        expected value of class labels
    s : array_like
        success probabilities (must be in non-decreasing order)
    filename : string, optional
        name of the file in which to save the plot

    Returns
    -------
    None
    """
    assert all(s[k] <= s[k + 1] for k in range(len(s) - 1))
    plt.figure()
    plt.plot(s, r, 'k*:')
    zeroone = np.asarray((0, 1))
    plt.plot(zeroone, zeroone, 'k')
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xlabel('score $S_k$')
    plt.ylabel('expected value ($P_k$) of outcome $R_k$')
    plt.title('exact expectations')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    #
    # Generate directories with plots as specified via the code below,
    # with each directory named n_nbins_inds_indt
    # (where n, nbins, inds, and indt are defined in the code below).
    #
    # Set parameters.
    # minorticks is the number of minor ticks on the lower axis.
    minorticks = 100
    # majorticks is the number of major ticks on the lower axis.
    majorticks = 10
    # n is the number of observations.
    for n in [100, 1000, 10000]:
        # Construct weights.
        weights = 4 - np.cos(9 * np.arange(n) / n)
        # nbins is the number of bins for the reliability diagrams.
        for nbins in [10, 40]:
            if n == 100 and nbins == 40:
                nbins = 4
            # nbins must divide n evenly.
            assert n % nbins == 0

            # Construct predicted success probabilities.
            sl = np.arange(0, 1, 1 / n) + 1 / (2 * n)
            # ss is a list of predicted probabilities for 3 kinds of examples.
            ss = [sl, np.square(sl), np.sqrt(sl)]
            for inds, s in enumerate(ss):
                # The probabilities must be in non-decreasing order.
                s = np.sort(s)

                # Construct true underlying probabilities for sampling.
                d = .4
                tl = -np.arange(-d, d, 2 * d / n) - d / n
                to = 1 - s - np.sqrt(s * (1 - s)) * (1 - np.sin(5 / (1.1 - s)))
                # ts is a list of true probabilities for 4 kinds of examples.
                ts = [tl, to, d - np.abs(tl), np.zeros(n)]
                ts[2][(n // 2 - n // 50):(n // 2 + n // 50)] = 0
                for indt, t in enumerate(ts):

                    # Limit consideration to only a third of the ss and ts.
                    if indt != 3 and (inds + indt) % 3 != 0:
                        continue
                    if indt == 3 and inds != 1:
                        continue

                    # Set a unique directory for each collection of experiments
                    # (creating the directory if necessary).
                    dir = 'weighted'
                    try:
                        os.mkdir(dir)
                    except FileExistsError:
                        pass
                    dir = 'weighted/' + str(n) + '_' + str(nbins)
                    dir = dir + '_' + str(inds)
                    dir = dir + '_' + str(indt)
                    try:
                        os.mkdir(dir)
                    except FileExistsError:
                        pass
                    dir = dir + '/'

                    # Generate a sample of classifications into two classes,
                    # correct (class 1) and incorrect (class 0),
                    # avoiding numpy's random number generators
                    # that are based on random bits --
                    # they yield strange results for many seeds.
                    random.seed(987654321)
                    uniform = np.asarray([random.random() for _ in range(n)])
                    r = (uniform <= s + t).astype(float)

                    print(f'./{dir} is under construction....')

                    # Generate five plots and a text file reporting metrics.
                    filename = dir + 'cumulative.pdf'
                    kuiper, kolmogorov_smirnov, lenscale = cumulative(
                        r, s, majorticks, minorticks, filename,
                        weights=weights)
                    filename = dir + 'metrics.txt'
                    with open(filename, 'w') as f:
                        f.write('n:\n')
                        f.write(f'{n}\n')
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
                    filename = dir + 'equiprob.pdf'
                    equiprob(r, s, nbins, filename, 20, weights)
                    filename = dir + 'equierr.pdf'
                    equierr(r, s, nbins, filename, 20, weights)
                    filename = dir + 'cumulative_exact.pdf'
                    _, _, _ = cumulative(
                        s + t, s, majorticks, minorticks, filename,
                        title='exact expectations', weights=weights)
                    filename = dir + 'exact.pdf'
                    exactdist(s + t, s, filename)
