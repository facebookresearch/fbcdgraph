#!/usr/bin/env python3

"""
Copyright (c) Facebook, Inc. and its affiliates.

Plots of deviation of a subpop. from the full pop., both cumulative and classic

*
This implementation considers only responses r that are restricted to taking
values 0 or 1 (Bernoulli variates).
*

Functions
---------
cumulative
    Cumulative difference between observations from a subpop. & the full pop.
equiscore
    Reliability diagram with roughly equispaced average scores over bins
equisamps
    Reliability diagram with an equal number of observations per bin
exactplot
    Reliability diagram with exact values plotted

This source code is licensed under the MIT license found in the LICENSE file in
the root directory of this source tree.
"""


import math
import os
import subprocess
import random
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def cumulative(r, s, inds, majorticks, minorticks, filename='cumulative.pdf',
               title='subpop. deviation is the slope as a function of $k/n$',
               fraction=1):
    """
    Cumulative difference between observations from a subpop. & the full pop.

    Saves a plot of the difference between the normalized cumulative sums of r
    for the subpopulation indices inds and the normalized cumulative sums of r
    from the full population interpolated to the subpopulation indices,
    with majorticks major ticks and minorticks minor ticks on the lower axis,
    labeling the major ticks with the corresponding values from s.

    Parameters
    ----------
    r : array_like
        class labels (0 for incorrect and 1 for correct classification)
    s : array_like
        scores (must be unique and in strictly increasing order)
    inds : array_like
        indices of the subset within s that defines the subpopulation
        (must be unique and in strictly increasing order)
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
            if x[k] > x[-1] * (j + 1) / nbins:
                j += 1
            if j == nbins:
                break
            nbin[j] += 1
        return nbin

    def aggregate(r, s, inds):
        # Counts the fraction that are 1s of entries of r in a bin
        # around each entry of s corresponding to the subset of s
        # specified by the indices inds. The bin ranges from halfway
        # to the nearest entry of s from inds on the left to halfway
        # to the nearest entry of s from inds on the right.
        ss = s[inds]
        q = np.insert(np.append(ss, [1e20]), 0, [-1e20])
        t = np.asarray([(q[k] + q[k + 1]) / 2 for k in range(len(q) - 1)])
        rc = np.zeros((len(inds)))
        sc = np.zeros((len(inds)))
        j = 0
        for k in range(len(s)):
            if s[k] > t[j + 1]:
                j += 1
                if j == len(inds):
                    break
            if s[k] >= t[0]:
                sc[j] += 1
                rc[j] += r[k]
        return rc / sc

    assert all(s[k] < s[k + 1] for k in range(len(s) - 1))
    assert all(inds[k] < inds[k + 1] for k in range(len(inds) - 1))
    assert len(r) == len(s)
    plt.figure()
    ax = plt.axes()
    # Aggregate r according to inds and s.
    rt = aggregate(r, s, inds)
    # Subsample r and s.
    rs = r[inds]
    ss = s[inds]
    # Accumulate and normalize.
    f = np.cumsum(rs) / int(len(rs) * fraction)
    ft = np.cumsum(rt) / int(len(rt) * fraction)
    # Plot the difference.
    plt.plot(np.insert(f - ft, 0, [0])[:(int(len(f) * fraction) + 1)], 'k')
    # Make sure the plot includes the origin.
    plt.plot(0, 'k')
    # Add an indicator of the scale of 1/sqrt(n) to the vertical axis.
    rtsub = rt[:int(len(rt) * fraction)]
    lenscale = np.sqrt(np.sum(rtsub * (1 - rtsub))) / len(rtsub)
    plt.plot(2 * lenscale, 'k')
    plt.plot(-2 * lenscale, 'k')
    kwargs = {
        'head_length': 2 * lenscale, 'head_width': len(rtsub) / 20, 'width': 0,
        'linewidth': 0, 'length_includes_head': True, 'color': 'k'}
    plt.arrow(.1e-100, -2 * lenscale, 0, 4 * lenscale, shape='left', **kwargs)
    plt.arrow(.1e-100, 2 * lenscale, 0, -4 * lenscale, shape='right', **kwargs)
    plt.margins(x=0, y=0)
    # Label the major ticks of the lower axis with the values of ss.
    sl = [
        '{:.2f}'.format(x)
        for x in np.insert(
            ss[:len(rtsub)], 0, [0])[::(len(rtsub) // majorticks)].tolist()]
    plt.xticks(
        np.arange(majorticks) * len(rtsub) // majorticks, sl[:majorticks])
    if len(rtsub) >= 300 and minorticks >= 50:
        # Indicate the distribution of s via unlabeled minor ticks.
        plt.minorticks_on()
        ax.tick_params(which='minor', axis='x')
        ax.tick_params(which='minor', axis='y', left=False)
        ax.set_xticks(np.cumsum(histcounts(minorticks, ss[:len(rtsub)])),
                      minor=True)
    # Label the axes.
    plt.xlabel('$S_{i_k}$ (the subscript on $S$ is $i_k$)')
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


def equiscore(r, s, inds, nbins, filename='equiscore.pdf'):
    """
    Reliability diagram with roughly equispaced average scores over bins

    Plots a reliability diagram with roughly equispaced average scores
    for the bins, for both the full population and the subpopulation specified
    by the indices inds.

    Parameters
    ----------
    r : array_like
        class labels (0 for incorrect and 1 for correct classification)
    s : array_like
        scores (must be in non-decreasing order)
    inds : array_like
        indices of the subset within s that defines the subpopulation
        (must be unique and in strictly increasing order)
    nbins : int
        number of bins
    filename : string, optional
        name of the file in which to save the plot

    Returns
    -------
    None
    """

    def bintwo(nbins, a, b, q, qmax):
        # Counts the number of entries of q falling into each of nbins bins,
        # and calculates the averages per bin of the arrays a and b,
        # returning np.nan as the "average" for any bin that is empty.
        j = 0
        bina = np.zeros(nbins)
        binb = np.zeros(nbins)
        nbin = np.zeros(nbins)
        for k in range(len(q)):
            if q[k] > qmax * (j + 1) / nbins:
                j += 1
            if j == nbins:
                break
            bina[j] += a[k]
            binb[j] += b[k]
            nbin[j] += 1
        # Normalize the sum for each bin to compute the arithmetic average.
        bina = np.divide(bina, nbin, where=nbin != 0)
        bina[np.where(nbin == 0)] = np.nan
        binb = np.divide(binb, nbin, where=nbin != 0)
        binb[np.where(nbin == 0)] = np.nan
        return nbin, bina, binb

    assert all(s[k] <= s[k + 1] for k in range(len(s) - 1))
    assert all(inds[k] < inds[k + 1] for k in range(len(inds) - 1))
    plt.figure()
    _, binr, bins = bintwo(nbins, r, s, s, s[inds[-1]])
    _, binrs, binss = bintwo(nbins, r[inds], s[inds], s[inds], s[inds[-1]])
    plt.plot(bins, binr, '*:', color='gray')
    plt.plot(binss, binrs, '*:', color='black')
    plt.xlim((0, s[inds[-1]]))
    plt.ylim((0, 1))
    plt.xlabel('average of $S_k$ for $k$ in the bin')
    plt.ylabel('average of $R_k$ for $k$ in the bin')
    plt.title('reliability diagram')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def equisamps(
        r, s, inds, nbins, filename='equisamps.pdf',
        title='reliability diagram (equal number of subpop. scores per bin)'):
    """
    Reliability diagram with an equal number of observations per bin

    Plots a reliability diagram with an equal number of observations per bin,
    for both the full population and the subpop. specified by indices inds.

    Parameters
    ----------
    r : array_like
        class labels (0 for incorrect and 1 for correct classification)
    s : array_like
        scores (must be in non-decreasing order)
    inds : array_like
        indices of the subset within s that defines the subpopulation
        (must be unique and in strictly increasing order)
    nbins : int
        number of bins
    filename : string, optional
        name of the file in which to save the plot
    title : string, optional
        title of the plot

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
    assert all(inds[k] < inds[k + 1] for k in range(len(inds) - 1))
    plt.figure()
    binr = hist(r, nbins)
    bins = hist(s, nbins)
    plt.plot(bins, binr, '*:', color='gray')
    rs = r[inds]
    ss = s[inds]
    binrs = hist(rs, nbins)
    binss = hist(ss, nbins)
    plt.plot(binss, binrs, '*:', color='black')
    plt.xlim((0, max(np.max(bins), np.max(binss))))
    plt.ylim((0, 1))
    plt.xlabel('average of $S_k$ for $k$ in the bin')
    plt.ylabel('average of $R_k$ for $k$ in the bin')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def exactplot(r, s, inds, filename='exact.pdf', title='exact expectations'):
    """
    Reliability diagram with exact values plotted

    Plots a reliability diagram at full resolution with fractional numbers,
    for both the full population and the subpop. specified by indices inds.
    The entries of r should be the expected values of class labels,
    not necessarily just 0s and 1s.

    Parameters
    ----------
    r : array_like
        expected value of class labels
    s : array_like
        scores (must be in non-decreasing order)
    inds : array_like
        indices of the subset within s that defines the subpopulation
        (must be unique and in strictly increasing order)
    filename : string, optional
        name of the file in which to save the plot
    title : string, optional
        title of the plot

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
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xlabel('score $S_k$')
    plt.ylabel('expected value ($P_k$) of outcome $R_k$')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    #
    # Generate directories with plots as specified via the code below,
    # with each directory named m_len(inds)_nbins_iex
    # (where m, inds, nbins, and iex are defined in the code below).
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
    # Consider 3 examples.
    for iex in range(3):
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
                inds = np.concatenate((inds1, inds2, inds3))
                # Indices must be sorted and unique.
                inds = np.unique(inds)
                inds = inds[0:(m // (m // len(inds)))]

                # Construct scores.
                sl = np.arange(0, 1, 1 / m) + 1 / (2 * m)
                s = np.square(sl)
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

            if iex == 1:
                # Define the indices of the subset for the subpopulation.
                np.random.seed(987654321)
                inds = np.sort(np.random.permutation((m))[:n])

                # Construct scores.
                s = np.arange(0, 1, 1 / m) + 1 / (2 * m)
                s = np.sqrt(s)
                # The scores must be in non-decreasing order.
                s = np.sort(s)

                # Construct perturbations to the scores for sampling rates.
                d = math.sqrt(1 / 2)
                tl = np.arange(-d, d, 2 * d / m) - d / m
                t = (1 + np.sin(np.arange((m)))) / 2 * (np.square(tl) - d**2)
                u = np.square(tl) - d**2
                u *= .75 * np.round(1 + np.sin(10 * np.arange((m)) / m)) / 2

                # Construct the exact sampling probabilities.
                exact = s + t
                exact[inds] = s[inds] + u[inds]

            if iex == 2:
                # Define the indices of the subset for the subpopulation.
                np.random.seed(987654321)
                inds = np.arange(0, m ** (3 / 4), 1)
                inds = np.unique(np.round(np.power(inds, 4 / 3))).astype(int)
                inds = inds[0:(50 * (len(inds) // 50))]

                # Construct scores.
                s = np.arange(0, 1, 1 / m) + 1 / (2 * m)
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

            # Set a unique directory for each collection of experiments
            # (creating the directory if necessary).
            dir = 'unweighted'
            try:
                os.mkdir(dir)
            except FileExistsError:
                pass
            dir = 'unweighted/' + str(m) + '_' + str(len(inds))
            dir = dir + '_' + str(nbins)
            dir = dir + '_' + str(iex)
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
            random.seed(987654321)
            uniform = np.asarray([random.random() for _ in range(m)])
            r = (uniform <= exact).astype(float)

            # Generate five plots and a text file reporting metrics.
            filename = dir + 'cumulative.pdf'
            kuiper, kolmogorov_smirnov, lenscale = cumulative(
                r, s, inds, majorticks, minorticks, filename)
            filename = dir + 'metrics.txt'
            with open(filename, 'w') as f:
                f.write('n:\n')
                f.write(f'{len(inds)}\n')
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
            filename = dir + 'cumulative_exact.pdf'
            _, _, _ = cumulative(
                exact, s, inds, majorticks, minorticks, filename,
                title='exact expectations')
            filename = dir + 'equiscore.pdf'
            equiscore(r, s, inds, nbins, filename)
            filename = dir + 'equisamps.pdf'
            equisamps(r, s, inds, nbins, filename)
            filepdf = dir + 'exact.pdf'
            filejpg = dir + 'exact.jpg'
            exactplot(exact, s, inds, filepdf)
            args = ['convert', '-density', '1200', filepdf, filejpg]
            procs.append(subprocess.Popen(args))
    print('waiting for conversion from pdf to jpg to finish....')
    for iproc, proc in enumerate(procs):
        proc.wait()
        print(f'{iproc + 1} of {len(procs)} conversions are done....')
