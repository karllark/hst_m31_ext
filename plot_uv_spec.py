#!/usr/bin/env python
#
# Program to plot a list of spectra used for extinction curves
#
# written Dec 2014/Jan 2015 by Karl Gordon (kgordon@stsci.edu)
# based strongly on IDL programs created over the previous 10 years
#
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from measure_extinction.stardata import StarData


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def plot_uv_set(ax, starnames, extra_off_val=0.0,
                norm_wave_range=[0.2, 0.3],
                col_vals=['b', 'g', 'r', 'm', 'c', 'y'],
                ann_xvals=[0.25],
                ann_wave_range=[0.2, 0.3],
                fontsize=12):
    """
    Plot a set of spectra
    """

    spec_name = 'STIS'
    for i in range(len(starnames)):
        stardata = StarData('DAT_files/'+starnames[i]+'.dat',
                            path='/home/kgordon/Python_git/extstar_data/',
                            use_corfac=True)

        ymult = np.full((len(stardata.data[spec_name].waves)), 1.0)

        # get the value to use for normalization and offset
        norm_indxs = np.where((stardata.data[spec_name].waves
                               >= norm_wave_range[0]) &
                              (stardata.data[spec_name].waves
                               <= norm_wave_range[1]))
        norm_val = 1.0/np.average(
            stardata.data[spec_name].fluxes[norm_indxs]
            * ymult[norm_indxs])
        off_val = extra_off_val + 10.0*(i) + 1.0
        print(off_val)

        # plot the spectroscopic data
        gindxs = np.where(stardata.data[spec_name].npts > 0)
        # max_gwave = max(stardata.data[spec_name].waves[gindxs])
        bnpts = 11
        xvals = smooth(stardata.data[spec_name].waves[gindxs], bnpts)
        yvals = smooth(stardata.data[spec_name].fluxes[gindxs], bnpts)
        ax.plot(xvals[0:-5],
                (yvals[0:-5]*ymult[gindxs][0:-5]
                 * norm_val * off_val),
                col_vals[i % 6] + '-')

        # annotate the spectra
        # ann_wave_range = np.array([max_gwave-5.0, max_gwave-1.0])
        ann_indxs = np.where((stardata.data[spec_name].waves
                              >= ann_wave_range[0]) &
                             (stardata.data[spec_name].waves
                              <= ann_wave_range[1]) &
                             (stardata.data[spec_name].npts > 0))
        ann_val = np.median(stardata.data[spec_name].fluxes[ann_indxs]
                            * ymult[ann_indxs])
        ann_val *= norm_val
        ann_val += off_val + 2.0
        # ax.annotate(starnames[i]+' '+stardata.sptype, xy=(ann_xvals[0],
        #                                                   ann_val),
        #             xytext=(ann_xvals[1], ann_val),
        #             verticalalignment="center",
        #             arrowprops=dict(facecolor=col_vals[i % 6], shrink=0.1),
        #             fontsize=0.85*fontsize, rotation=-0.)

        # plot the band fluxes
        ymult = np.full((len(stardata.data['BAND'].waves)), 1.0)
        ax.plot(stardata.data['BAND'].waves,
                stardata.data['BAND'].fluxes*ymult
                * norm_val * off_val, col_vals[i % 6] + 'o')


def ann_set(ax, fontsize,
            bracket_x, bracket_y,
            text_x,
            texta, textb):
    """
    Annotate a set of spectra with text
    """
    ax.plot([bracket_x[0], bracket_x[1], bracket_x[1], bracket_x[0]],
            [bracket_y[0], bracket_y[0], bracket_y[1], bracket_y[1]],
            'k-',
            linewidth=3.)
    text_y = 0.5*(bracket_y[0] + bracket_y[1])
    ax.text(text_x[0], text_y, texta, rotation=270.,
            fontsize=1.2*fontsize,
            horizontalalignment='center', verticalalignment="center")
    ax.text(text_x[1], text_y, textb, rotation=270.,
            fontsize=.9*fontsize,
            horizontalalignment='center', verticalalignment="center")


if __name__ == "__main__":

    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--png", help="save figure as a png file",
                        action="store_true")
    parser.add_argument("--eps", help="save figure as an eps file",
                        action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file",
                        action="store_true")
    args = parser.parse_args()

    fontsize = 12

    font = {'size': fontsize}

    matplotlib.rc('font', **font)

    matplotlib.rc('lines', linewidth=1)
    matplotlib.rc('axes', linewidth=2)
    matplotlib.rc('xtick.major', width=2)
    matplotlib.rc('xtick.minor', width=2)
    matplotlib.rc('ytick.major', width=2)
    matplotlib.rc('ytick.minor', width=2)

    # fig, ax = pyplot.subplots(nrows=1, ncols=1, figsize=(10, 13))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

    # 'j004354.05+412626.0', 'j004427.47+415150.0',
    starnames = [
                 'j004413.84+414903.9',
                 'j004420.52+411751.1']
    plot_uv_set(ax, starnames,
                ann_xvals=[0.25, 0.25], ann_wave_range=[0.20, 0.3])

    ax.set_yscale('linear')
    ax.set_ylim(0.3, 30.)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.1, 0.6)
    ax.set_xlabel(r'$\lambda$ [$\mu m$]', fontsize=1.3 * fontsize)
    ax.set_ylabel(r'$F(\lambda)/F(BV)$ + offset',
                  fontsize=1.3*fontsize)

    ax.tick_params('both', length=10, width=2, which='major')
    ax.tick_params('both', length=5, width=1, which='minor')

    fig.tight_layout()

    save_file = 'm31_uv_mspec'
    if args.png:
        fig.savefig(save_file + '.png')
    elif args.eps:
        fig.savefig(save_file + '.eps')
    elif args.pdf:
        fig.savefig(save_file + '.pdf')
    else:
        plt.show()
