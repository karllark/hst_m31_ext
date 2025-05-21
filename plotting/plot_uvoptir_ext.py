import argparse

import numpy as np
import matplotlib.pyplot as pyplot
import matplotlib

import astropy.units as u

from dust_extinction.shapes import FM90_B3
from measure_extinction.extdata import ExtData


def mask_bad(cdata, cname):
    # mask bad data (lines)
    mask = [
        (8.4, 8.05),
        (7.27, 7.10),
        (7.55, 7.45),
        (7.75, 7.65),
        (7.95, 7.9),
        (6.6, 6.4),
        (4.22, 4.17),
        (4.28, 4.255),
        (3.51, 3.49),
        (3.59, 3.56),
        (3.87, 3.82),
    ]

    # add exclude regions
    exreg = {
        "e1": [0.17, 0.2],
        "e2": None,
        "e3": None,
        "e4": [0.17, 0.2],
        "e5": None,
        "e6": None,
        "e7": None,
        "e8": [0.162, 0.2],
        "e9": [0.162, 0.2],
        "e12": [0.165, 0.205],
        "e13": [0.165, 0.205],  # [.137, .143, .165, .205]
        "e14": [0.165, 0.21],
        "e15": None,
        "e17": None,
        "e18": None,
        "e22": None,
        "e24": None,
    }
    if exreg[cname] is not None:
        mask.append(1.0 / np.array(exreg[cname]))

    mask = 1.0 / np.array(mask)

    for region in mask:
        for src in cdata.waves.keys():
            cdata.npts[src][
                (cdata.waves[src].value >= region[0])
                & (cdata.waves[src].value <= region[1])
            ] = 0

    return cdata


def plot_all_ext(
    ax,
    extdatas,
    snames,
    kxrange,
    normvals=None,
    yoffset_factor=0.0,
    annotate_key=None,
    topxaxis=True,
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
    # mod_x_g21 = np.logspace(0.1, np.log10(35.0), 200) * u.micron
    mod_x_fm90 = 1.0 / (np.logspace(-1.0, -0.5, 200) * u.micron)
    for i in range(len(extnames)):
        k = sindxs[i]

        if normvals is not None:
            normval = normvals[k]
        else:
            normval = 1.0

        # plot the extinction curves
        # if extnames[k].split("_")[0] == "hd283809":
        #    extdatas[k].npts["IUE"][extdatas[k].waves["IUE"] > 0.315 * u.micron] = 0

        if not args.modonly:
            cdata = extdatas[k]
            cname = snames[k]
            cdata = mask_bad(cdata, cname.split("_")[1])
            cdata.plot(
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

        fdata = extdatas[k].fit_params["MCMC"]
        C2 = fdata[np.where(fdata["name"] == "C2")[0]]["value"].data[0]
        FM90_p50 = FM90_B3(
            C1=2.18 - 2.91 * C2,
            C2=C2,
            B3=fdata[np.where(fdata["name"] == "B3")[0]]["value"].data[0],
            C4=fdata[np.where(fdata["name"] == "C4")[0]]["value"].data[0],
            xo=fdata[np.where(fdata["name"] == "xo")[0]]["value"].data[0],
            gamma=fdata[np.where(fdata["name"] == "gamma")[0]]["value"].data[0],
        )

        mod_y = FM90_p50(mod_x_fm90)
        rv = extdatas[k].columns["RV"][0]
        mod_y = (mod_y / rv) + 1

        mod_y = mod_y / normval + (i * yoffset_factor)

        if annotate_key == "STIS":
            annx = 3.5
            annx_delta = 0.25
            annvals = np.absolute(mod_x_fm90.value - annx) < annx_delta
            anny = np.mean(mod_y[annvals]) + 0.1 * yoffset_factor
            ax.text(
                annx,
                anny,
                extnames[k].split("_")[1],
                color=col_vals[i % n_cols],
                alpha=0.75,
                fontsize=12,
                rotation=10.0,
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
    # ax.set_xscale("log")
    ax.set_xlim(kxrange)
    ax.set_ylabel(r"$A(\lambda)/A(V)$", fontsize=1.3 * fontsize)

    ax.set_xlabel(r"$\lambda$ [$\mu m$]")

    ax.tick_params("both", length=10, width=2, which="major")
    ax.tick_params("both", length=5, width=1, which="minor")

    if topxaxis:
        # for 2nd x-axis with lambda values
        axis_xs = np.array([0.12, 0.15, 0.2, 0.3, 0.5, 1.0])
        new_ticks = 1 / axis_xs
        new_ticks_labels = ["%.2f" % z for z in axis_xs]
        tax = ax.twiny()
        tax.set_xlim(ax.get_xlim())
        tax.set_xticks(new_ticks)
        tax.set_xticklabels(new_ticks_labels, fontsize=0.8 * fontsize)
        tax.set_xlabel(r"$\lambda$ [$\mu$m]")


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
    norm_wave_range = [0.15, 0.20] * u.micron
    normvals = []

    for name in starnames:
        extnames.append(name)
        bfilename = f"exts/{name}_mefit_ext_elv.fits"
        text = ExtData(filename=bfilename)
        text.trans_elv_alav()
        extdatas.append(text)
        avs.append(text.columns["AV"][0])

        fdata = text.fit_params["MCMC"]
        C2 = fdata[np.where(fdata["name"] == "C2")[0]]["value"].data[0]
        rv = fdata[np.where(fdata["name"] == "Rv")[0]]["value"].data[0]
        normvals.append(C2 / rv + 1)

    normvals = np.array(normvals)
    sindxs = np.argsort(normvals)
    normvals = normvals[sindxs]
    extnames = np.array(extnames)[sindxs]
    starnames = np.array(starnames)[sindxs]

    extdatas = []
    avs = []

    for extname in extnames:
        bfilename = f"exts/{extname}_mefit_ext_elv.fits"
        text = ExtData(filename=bfilename)
        text.trans_elv_alav()
        extdatas.append(text)
        avs.append(text.columns["AV"][0])

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
        starnames,
        kxrange=[0.3, 9.0],
        # normvals=normvals,
        normvals=None,
        # annotate_key=None,
        annotate_key="STIS",
        yoffset_factor=1.0,
    )

    ax[0].set_ylim(-0.5, 22.0)
    ax[0].set_ylabel(r"$A(\lambda)/A(V)$ + constant")

    fig.tight_layout()  # rect=(0.9,0.9))

    save_str = "m31_uvoptir_ext"
    if args.png:
        fig.savefig(f"figs/{save_str}.png")
    elif args.pdf:
        fig.savefig(f"figs/{save_str}.pdf")
    else:
        pyplot.show()
