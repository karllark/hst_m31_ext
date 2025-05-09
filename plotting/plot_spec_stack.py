import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import astropy.units as u

from measure_extinction.stardata import StarData


def plot_set(
    ax,
    starnames,
    extra_off_val=0.0,
    norm_wave_range=[0.7, 0.8] * u.micron,
    col_vals=["b", "g", "r", "m", "c", "y"],
    ann_wave_range=[0.7, 0.8] * u.micron,
    ann_rot=5.0,
    ann_offset=0.2,
    fontsize=12,
    path="/home/kgordon/Python/extstar_data/MW/",
    subpath="",
):
    """
    Plot a set of spectra
    """
    only_bands = ["J", "H", "K"]

    n_col = len(col_vals)
    for i in range(len(starnames)):
        stardata = StarData(subpath + starnames[i] + ".dat", path=path, only_bands=only_bands)

        stardata.plot(
            ax,
            norm_wave_range=norm_wave_range,
            yoffset=extra_off_val + 0.5 * i,
            yoffset_type="multiply",
            pcolor=col_vals[i % n_col],
            annotate_key="STIS_Opt",
            annotate_wave_range=ann_wave_range,
            annotate_text=starnames[i] + " " + stardata.sptype,
            fontsize=fontsize*0.8,
            annotate_rotation=ann_rot,
            annotate_yoffset=ann_offset,
        )


if __name__ == "__main__":

    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    fontsize = 12

    font = {"size": fontsize}

    plt.rc("font", **font)

    plt.rc("lines", linewidth=1)
    plt.rc("axes", linewidth=2)
    plt.rc("xtick.major", width=2)
    plt.rc("xtick.minor", width=2)
    plt.rc("ytick.major", width=2)
    plt.rc("ytick.minor", width=2)

    # fig, ax = pyplot.subplots(nrows=1, ncols=1, figsize=(10, 13))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

    all = "2massj085747 2massj130152 2massj150958 2massj170756 2massj173628 2massj181129 2massj182302 2massj203110 2massj203234 2massj203311 2massj203326 2massj204521"
    # all = "2massj130152 2massj170756 2massj182302 2massj203234 2massj203311 2massj203326 2massj204521 2massj085747 2massj150958 2massj173628 2massj181129 2massj203110"

    starnames = np.array(all.split())

    # avs = [4.76, 5.53, 4.49, 5.29, 4.99, 4.45, 5.50, 4.97, 6.24, 6.79, 6.18, 6.17]
    # sindxs = np.argsort(avs)
    # starnames = starnames[sindxs]

    # read in the data and determine the optical spectral slope
    # then sort by that
    path = "/home/kgordon/Python/extstar_data/MW/"
    sslope = []
    for cstar in starnames:
        stardata = StarData(f"{cstar}.dat", path=path)
        gvals = (
            np.absolute(stardata.data["STIS_Opt"].waves - 0.35 * u.micron)
            < 0.01 * u.micron
        )
        bval = np.nanmedian(stardata.data["STIS_Opt"].fluxes[gvals])
        ctype = "MIRI_IFU"
        gvals = (
            np.absolute(stardata.data[ctype].waves - 15.0 * u.micron)
            < 0.1 * u.micron
        )
        rval = np.nanmedian(stardata.data[ctype].fluxes[gvals])
        sslope.append(bval / rval)

    sindxs = np.argsort(sslope)
    starnames = starnames[sindxs]

    col_vals = ["b", "g"]
    plot_set(
        ax,
        starnames,
        ann_rot=0.0,
        ann_offset=0.25,
        col_vals=col_vals,
    )

    ax.set_yscale("log")
    ax.set_ylim(1e-5, 1e6)
    ax.set_xscale("log")
    # ax.set_xlim(kxrange)
    ax.set_xlabel(r"$\lambda$ [$\mu m$]", fontsize=1.3 * fontsize)
    ax.set_ylabel(r"$F(\lambda)/F(0.75 \mu m)$ + offset", fontsize=1.3 * fontsize)

    ax.tick_params("both", length=10, width=2, which="major")
    ax.tick_params("both", length=5, width=1, which="minor")

    # ax.spines["right"].set_visible(False)
    # ax.spines["top"].set_visible(False)

    ax.xaxis.set_minor_formatter(ScalarFormatter())
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xticks([0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2, 3, 5, 8, 10, 15.0, 20.0, 30.0], minor=True)

    fig.tight_layout()

    save_file = "figs/wisci_ext_spec"
    if args.png:
        fig.savefig(save_file + ".png")
    elif args.pdf:
        fig.savefig(save_file + ".pdf")
    else:
        plt.show()
