import argparse
import os.path

import numpy as np
import matplotlib.pyplot as pyplot
import matplotlib
from matplotlib.ticker import ScalarFormatter

# from astropy.table import Table
import astropy.units as u

# from calc_ext import P92_Elv
from dust_extinction.shapes import FM90
from measure_extinction.extdata import ExtData
from dust_extinction.shapes import G21


def plot_all_ext(
    ax, extdatas, kxrange, kyrange, normvals=None, yoffset_factor=0.0, annotate_key=None
):
    """
    plot all the extintion info on the specified plot
    """
    # sindxs = np.argsort(avs)
    sindxs = np.arange(len(avs))

    # ann_wave_range = [5.0, 10.0] * u.micron
    col_vals = ["b", "g"]  # , "r", "m", "c", "y"]
    lin_vals = ["--", ":", "-."]
    n_cols = len(col_vals)

    # mod_x = np.logspace(0.0, 2.0, 200) * u.micron
    mod_x_g21 = np.logspace(0.1, np.log10(35.0), 200) * u.micron
    mod_x_fm90 = np.logspace(-1.0, -0.5, 200) * u.micron
    for i in range(len(extnames)):
        k = sindxs[i]

        if normvals is not None:
            normval = normvals[k]
        else:
            normval = 1.0

        # plot the extinction curves
        if extnames[k].split("_")[0] == "hd283809":
            extdatas[k].npts["IUE"][extdatas[k].waves["IUE"] > 0.315 * u.micron] = 0

        if not args.modonly:
            extdatas[k].plot(
                ax,
                color=col_vals[i % n_cols],
                alax=extdatas[k].type != "alax",
                normval=normval,
                yoffset=i * yoffset_factor,
                alpha=1.0,
                rebin_fac=args.rebin_fac,
                fontsize=fontsize,
                wavenum=True,
            )

        if args.models:

            if hasattr(extdatas[k], "g21_best_fit"):
                # best fit G21 model
                if extdatas[k] is not None:
                    G21_best = G21(
                        scale=extdatas[k].g21_best_fit["SCALE"],
                        alpha=extdatas[k].g21_best_fit["ALPHA"],
                        sil1_amp=extdatas[k].g21_best_fit["SIL1_AMP"],
                        sil1_center=extdatas[k].g21_best_fit["SIL1_CENTER"],
                        sil1_fwhm=extdatas[k].g21_best_fit["SIL1_FWHM"],
                        sil1_asym=extdatas[k].g21_best_fit["SIL1_ASYM"],
                        sil2_amp=extdatas[k].g21_best_fit["SIL2_AMP"],
                        sil2_center=extdatas[k].g21_best_fit["SIL2_CENTER"],
                        sil2_fwhm=extdatas[k].g21_best_fit["SIL2_FWHM"],
                        sil2_asym=extdatas[k].g21_best_fit["SIL2_ASYM"],
                    )

                mod_y = G21_best(mod_x_g21) / normval + i * yoffset_factor

                if annotate_key == "MIRI_IFU":
                    annx = 4.0
                    annx_delta = 1.0
                    annvals = np.absolute(mod_x_g21.value - annx) < annx_delta
                    anny = np.mean(mod_y[annvals]) + 0.1 * yoffset_factor - 0.005
                    ax.text(
                        annx,
                        anny,
                        extnames[k].split("_")[0],
                        color=col_vals[i % n_cols],
                        alpha=0.75,
                        fontsize=12,
                        horizontalalignment="center",
                        rotation=-12.0,
                    )

                ax.plot(
                    mod_x_g21,
                    mod_y,
                    lin_vals[i % 3],
                    color=col_vals[i % n_cols],
                    alpha=0.5,
                )

                if annotate_key == "STIS":
                    annx = 0.25
                    annx_delta = 0.02
                    annvals = np.absolute(extdatas[k].waves["STIS"].value - annx) < annx_delta
                    mod_y = extdatas[k].exts["STIS"] / normval + i * yoffset_factor
                    anny = np.mean(mod_y[annvals]) - 0.1 * yoffset_factor - 0.1
                    ax.text(
                        annx,
                        anny,
                        extnames[k].split("_")[0],
                        color=col_vals[i % n_cols],
                        alpha=0.75,
                        fontsize=12,
                        rotation=-25.0,
                        horizontalalignment="center",
                    )

            if extdatas_fm90[k] is not None:
                if hasattr(extdatas_fm90[k], "fm90_best_fit"):
                    # best fit FM90 model
                    if extdatas_fm90[k] is not None:

                        FM90_p50 = FM90(
                            C1=extdatas_fm90[k].fm90_p50_fit["C1"][0],
                            C2=extdatas_fm90[k].fm90_p50_fit["C2"][0],
                            C3=extdatas_fm90[k].fm90_p50_fit["C3"][0],
                            C4=extdatas_fm90[k].fm90_p50_fit["C4"][0],
                            xo=extdatas_fm90[k].fm90_p50_fit["XO"][0],
                            gamma=extdatas_fm90[k].fm90_p50_fit["GAMMA"][0],
                        )

                        mod_y = FM90_p50(mod_x_fm90) / normval + i * yoffset_factor

                        if annotate_key == "STIS":
                            annx = 0.28
                            annx_delta = 0.02
                            annvals = np.absolute(mod_x_fm90.value - annx) < annx_delta
                            anny = np.mean(mod_y[annvals]) + 0.1 * yoffset_factor
                            ax.text(
                                annx,
                                anny,
                                extnames[k].split("_")[0],
                                color=col_vals[i % n_cols],
                                alpha=0.75,
                                fontsize=12,
                                rotation=-10.0,
                                horizontalalignment="center",
                            )

                        ax.plot(
                            mod_x_fm90,
                            mod_y,
                            lin_vals[i % 3],
                            color="k",  # col_vals[i % n_cols],
                            alpha=0.5,
                        )

    ax.set_yscale("linear")
    #ax.set_xscale("log")
    ax.set_xlim(kxrange)
    ax.set_ylabel(r"$A(\lambda)/A(V)$", fontsize=1.3 * fontsize)

    ax.set_xlabel(r"$\lambda$ [$\mu m$]")

    ax.tick_params("both", length=10, width=2, which="major")
    ax.tick_params("both", length=5, width=1, which="minor")


if __name__ == "__main__":

    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rebin_fac", type=int, default=None, help="rebin factor for spectra"
    )
    parser.add_argument(
        "--models", help="plot the best fit models", action="store_true"
    )
    parser.add_argument(
        "--modonly", help="only plot the best fit models", action="store_true"
    )
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    all = "m31_e1_j004354.05+412626.0 m31_e2_j004413.84+414903.9 m31_e3_j004420.52+411751.1 m31_e4_j004427.47+415150.0 m31_e5_j004431.66+413612.4 m31_e6_j004438.71+415553.5 m31_e7_j004454.37+412823.9 m31_e8_j004511.82+415025.3 m31_e9_j004511.85+413712.9 m31_e12_j004539.00+415439.0 m31_e13_j004539.70+415054.8 m31_e14_j004543.46+414513.6 m31_e15_j004546.81+415431.7 m31_e17_j003944.71+402056.2 m31_e18_j003958.22+402329.0 m31_e22_j004034.61+404326.1 m31_e24_j004412.17+413324.2"

    starnames = np.array(all.split())

    extnames = []
    extdatas = []
    extdatas_fm90 = []
    avs = []

    normtype = "STIS"
    norm_wave_range = [0.25, 0.30] * u.micron
    normvals = []

    for name in starnames:
        extnames.append(name)
        bfilename = f"exts/{name}_mefit_elv.fits"
        text = ExtData(filename=bfilename)
        text.trans_elv_alav()
        extdatas.append(text)
        avs.append(text.columns["AV"][0])

        # determine the extinction in the near-UV
        # useful for sorting to make a pretty plot
        if "STIS" in text.exts.keys():
            (gindxs,) = np.where(
                (text.npts[normtype] > 0)
                & (
                    (text.waves[normtype] >= norm_wave_range[0])
                    & (text.waves[normtype] <= norm_wave_range[1])
                )
            )
            normvals.append(
                np.average(
                    (text.exts[normtype][gindxs]) / float(text.columns["AV"][0])
                    + 1.0
                )
            )
        else:
            normvals.append(1.0)

    # print(normvals)
    normvals = np.array(normvals)
    sindxs = np.flip(np.argsort(normvals))
    normvals = normvals[sindxs]
    extnames = np.array(extnames)[sindxs]

    extdatas = []
    avs = []

    for extname in extnames:
        bfilename = f"exts/{extname}_mefit_elv.fits"
        text = ExtData(filename=bfilename)
        text.trans_elv_alav()
        extdatas.append(text)
        avs.append(text.columns["AV"][0])

        fm90_filename = bfilename.replace(".fits", "_FM90.fits")
        if os.path.isfile(fm90_filename):
            textfm90 = ExtData(filename=fm90_filename)
            extdatas_fm90.append(textfm90)
        else:
            extdatas_fm90.append(None)

    fontsize = 18

    font = {"size": fontsize}

    matplotlib.rc("font", **font)

    matplotlib.rc("lines", linewidth=1)
    matplotlib.rc("axes", linewidth=2)
    matplotlib.rc("xtick.major", width=2)
    matplotlib.rc("xtick.minor", width=2)
    matplotlib.rc("ytick.major", width=2)
    matplotlib.rc("ytick.minor", width=2)

    figsize = (8, 11)
    fig, ax = pyplot.subplots(nrows=1, ncols=1, figsize=figsize)
    ax = [ax]

    plot_all_ext(
        ax[0],
        extdatas,
        kxrange=[0.3, 9.0],
        kyrange=[1.0, 30.0],
        normvals=normvals,
        # annotate_key=None,
        annotate_key="STIS",
        yoffset_factor=0.6,
    )

    ax[0].set_ylim(-0.5, 12.0)
    ax[0].set_ylabel(r"$A(\lambda)/A(V)$ + constant")

    fig.tight_layout()  # rect=(0.9,0.9))

    save_str = "m31_uvoptir_ext"
    if args.png:
        fig.savefig(f"figs/{save_str}.png")
    elif args.pdf:
        fig.savefig(f"figs/{save_str}.pdf")
    else:
        pyplot.show()
