#!/usr/bin/env python3

"""
Copyright (c) Facebook, Inc. and its affiliates.

Calibration plots, both cumulative and two kinds of reliability diagrams

Functions
---------
cumulative
    Cumulative difference between observed and expected vals of Bernoulli vars
equiprob
    Reliability diagram with roughly equispaced average probabilities over bins
equisamp
    Reliability diagram with an equal number of observations per bin

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


def cumulative(r, s, majorticks, minorticks, filename='cumulative.pdf',
               title='miscalibration is the slope as a function of $k/n$',
               fraction=1):
    """
    Cumulative difference between observed and expected vals of Bernoulli vars

    Saves a plot of the difference between the normalized cumulative sums of r
    and the normalized cumulative sums of s, with majorticks major ticks
    and minorticks minor ticks on the lower axis, labeling the major ticks
    with the corresponding values from s.

    Parameters
    ----------
    r : array_like
        class labels (0 for incorrect and 1 for correct classification)
    s : array_like
        success probabilities (must be unique and in strictly increasing order)
    majorticks : int
        number of major ticks on the lower axis
    minorticks : int
        number of minor ticks on the lower axis
    filename : string, optional
        name of the file in which to save the plot
    title : string, optional
        title of the plot
    fraction : float, optional
        proportion of the full horizontal axis to display

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

    def histcounts(nbins, x):
        # Counts the number of entries of x
        # falling into each of nbins equispaced bins.
        j = 0
        nbin = np.zeros(nbins)
        for k in range(len(x)):
            if x[k] > (j + 1) / nbins:
                j += 1
            if j == nbins:
                break
            nbin[j] += 1
        return nbin

    assert all(s[k] < s[k + 1] for k in range(len(s) - 1))
    assert len(r) == len(s)
    plt.figure()
    ax = plt.axes()
    # Accumulate and normalize r and s.
    f = np.cumsum(r) / int(len(r) * fraction)
    ft = np.cumsum(s) / int(len(s) * fraction)
    # Plot the difference.
    plt.plot(np.insert(f - ft, 0, [0])[:(int(len(f) * fraction) + 1)], 'k')
    # Make sure the plot includes the origin.
    plt.plot(0, 'k')
    # Add an indicator of the scale of 1/sqrt(n) to the vertical axis.
    ssub = s[:int(len(s) * fraction)]
    lenscale = np.sqrt(np.sum(ssub * (1 - ssub))) / len(ssub)
    plt.plot(2 * lenscale, 'k')
    plt.plot(-2 * lenscale, 'k')
    kwargs = {
        'head_length': 2 * lenscale, 'head_width': len(ssub) / 20, 'width': 0,
        'linewidth': 0, 'length_includes_head': True, 'color': 'k'}
    plt.arrow(.1e-100, -2 * lenscale, 0, 4 * lenscale, shape='left', **kwargs)
    plt.arrow(.1e-100, 2 * lenscale, 0, -4 * lenscale, shape='right', **kwargs)
    plt.margins(x=0, y=0)
    # Label the major ticks of the lower axis with the values of s.
    ss = [
        '{:.2f}'.format(x)
        for x in np.insert(ssub, 0, [0])[::(len(ssub) // majorticks)].tolist()]
    plt.xticks(
        np.arange(majorticks) * len(ssub) // majorticks, ss[:majorticks])
    if len(ssub) >= 300 and minorticks >= 50:
        # Indicate the distribution of s via unlabeled minor ticks.
        plt.minorticks_on()
        ax.tick_params(which='minor', axis='x')
        ax.tick_params(which='minor', axis='y', left=False)
        ax.set_xticks(np.cumsum(histcounts(minorticks, ssub)), minor=True)
    # Label the axes.
    plt.xlabel('$S_k$')
    plt.ylabel('$F_k - \\tilde{F}_k$')
    plt.twiny()
    plt.xlabel('$k/n$')
    # Title the plot.
    plt.title(title)
    # Clean up the whitespace in the plot.
    plt.tight_layout()
    # Save the plot.
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    # Calculate summary statistics.
    fft = np.insert(f - ft, 0, [0])[:(int(len(f) * fraction) + 1)]
    kuiper = np.max(fft) - np.min(fft)
    kolmogorov_smirnov = np.max(np.abs(fft))
    return kuiper, kolmogorov_smirnov, lenscale


def equiprob(r, s, nbins, filename='equiprob.pdf', n_resamp=0):
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

    Returns
    -------
    None
    """

    def bintwo(nbins, a, b, q):
        # Counts the number of entries of q falling into each of nbins
        # equispaced bins and calculates the averages per bin of the arrays
        # a and b, returning np.nan as the "average" for any bin that is empty.
        j = 0
        bina = np.zeros(nbins)
        binb = np.zeros(nbins)
        nbin = np.zeros(nbins)
        for k in range(len(q)):
            if q[k] > (j + 1) / nbins:
                j += 1
            if j == nbins:
                break
            bina[j] += a[k]
            binb[j] += b[k]
            nbin[j] += 1
        # Normalize the sum for each bin to compute the arithmetic average.
        bina = np.divide(bina, nbin, where=nbin != 0)
        bina[nbin == 0] = np.nan
        binb = np.divide(binb, nbin, where=nbin != 0)
        binb[nbin == 0] = np.nan
        return nbin, bina, binb

    assert all(s[k] <= s[k + 1] for k in range(len(s) - 1))
    plt.figure()
    for _ in range(n_resamp):
        # Resample from s, preserving the pairing of s with r.
        sr = np.asarray(
            [[s[k], r[k]] for k in np.random.randint(0, len(s), (len(s)))])
        perm = np.argsort(sr[:, 0])
        ss = sr[perm, 0]
        rs = sr[perm, 1]
        _, binrs, binss = bintwo(nbins, rs, ss, ss)
        # Use the light gray, "gainsboro".
        plt.plot(binss, binrs, 'gainsboro')
    _, binr, bins = bintwo(nbins, r, s, s)
    # Use the solid black, "k".
    plt.plot(bins, binr, 'k*:')
    zeroone = np.asarray((0, 1))
    plt.plot(zeroone, zeroone, 'k')
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xlabel('average of $S_k$ for $k$ in the bin')
    plt.ylabel('average of $R_k$ for $k$ in the bin')
    plt.title('reliability diagram')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def equisamp(
        r, s, nbins, filename='equisamp.pdf',
        title='reliability diagram (equal number of observations per bin)',
        n_resamp=0, xlabel='average of $S_k$ for $k$ in the bin',
        ylabel='average of $R_k$ for $k$ in the bin'):
    """
    Reliability diagram with an equal number of observations per bin

    Plots a reliability diagram with an equal number of observations per bin.

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
    title : string, optional
        title of the plot
    n_resamp : int, optional
        number of times to resample and plot an extra line for error bars
    xlabel : string, optional
        label for the horizontal axis
    ylabel : string, optional
        label for the vertical axis

    Returns
    -------
    None
    """

    def hist(a, nbins):
        # Calculates the average of a in nbins bins,
        # each containing len(a) // nbins entries of a
        ns = len(a) // nbins
        return np.sum(np.reshape(a[:nbins * ns], (nbins, ns)), axis=1) / ns

    assert all(s[k] <= s[k + 1] for k in range(len(s) - 1))
    plt.figure()
    for _ in range(n_resamp):
        # Resample from s, preserving the pairing of s with r.
        sr = np.asarray(
            [[s[k], r[k]] for k in np.random.randint(0, len(s), (len(s)))])
        perm = np.argsort(sr[:, 0])
        ss = sr[perm, 0]
        rs = sr[perm, 1]
        binrs = hist(rs, nbins)
        binss = hist(ss, nbins)
        # Use the light gray, "gainsboro".
        plt.plot(binss, binrs, 'gainsboro')
    binr = hist(r, nbins)
    bins = hist(s, nbins)
    # Use the solid black, "k".
    plt.plot(bins, binr, 'k*:')
    zeroone = np.asarray((0, 1))
    plt.plot(zeroone, zeroone, 'k')
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
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
                # The success probabilities must be in non-decreasing order.
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
                    dir = 'unweighted'
                    try:
                        os.mkdir(dir)
                    except FileExistsError:
                        pass
                    dir = 'unweighted/' + str(n) + '_' + str(nbins)
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
                        r, s, majorticks, minorticks, filename)
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
                    equiprob(r, s, nbins, filename, n_resamp=20)
                    filename = dir + 'equisamp.pdf'
                    equisamp(r, s, nbins, filename, n_resamp=20)
                    filename = dir + 'cumulative_exact.pdf'
                    _, _, _ = cumulative(
                        s + t, s, majorticks, minorticks, filename,
                        title='exact expectations')
                    filename = dir + 'exact.pdf'
                    equisamp(
                        s + t, s, n, filename, title='exact expectations',
                        xlabel='score $S_k$',
                        ylabel='expected value ($P_k$) of outcome $R_k$')
