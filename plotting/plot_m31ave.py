import argparse
import copy

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import QTable
import astropy.units as u

from dust_extinction.averages import G03_LMCAvg, G03_LMC2, G24_SMCAvg
from dust_extinction.parameter_averages import G23

from measure_extinction.extdata import ExtData, AverageExtData


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    fontsize = 16

    font = {"size": fontsize}

    plt.rc("font", **font)

    plt.rc("lines", linewidth=1)
    plt.rc("axes", linewidth=2)
    plt.rc("xtick.major", width=2)
    plt.rc("xtick.minor", width=2)
    plt.rc("ytick.major", width=2)
    plt.rc("ytick.minor", width=2)

    figsize = (8, 6)
    fig, ax = plt.subplots(figsize=figsize)

    filelist = "data/m31_exts.dat"
    klabel = "M31 Average"
    ofilename = "exts/m31_ext.fits"
    f = open(filelist, "r")
    file_lines = list(f)
    starnames = []
    extdatas = []
    spslopes = []
    rvs = []
    # rebinfac = None
    rebinfac = 10
    for line in file_lines:
        if (line.find("#") != 0) & (len(line) > 0):
            name = line.rstrip()
            starnames.append(name)
            text = ExtData()
            filebase = "./exts/" + starnames[-1] + "_mefit_ext_elv"
            text.read(f"{filebase}.fits")
            text.trans_elv_alav()

            extdatas.append(text)
            rvs.append(text.columns["RV"][0])

            text.plot(
                ax,
                color="k",
                alpha=0.1,
                rebin_fac=rebinfac,
                wavenum=True,
            )
    aveext = AverageExtData(extdatas, min_number=2)

    ave_rv = np.average(rvs)
    print(f"average R(V) = {ave_rv}")
    aveext.columns["RV"] = (ave_rv, 0.0)

    aveext.save(ofilename)

    mod_x = np.arange(0.5, 9.0, 0.1)
    rv = 3.1
    mod = G23(Rv=rv)
    mod_y = mod(mod_x)
    ax.plot(
        mod_x,
        mod(mod_x),
        "r--",
        alpha=0.75,
        label=f"Milky Way: G23 R(V)={rv:.2f}",
        linewidth=3.0,
    )

    mod = G03_LMCAvg()
    mod_y = mod(mod_x)
    ax.plot(
        mod_x,
        mod(mod_x),
        "c--",
        alpha=0.75,
        label=f"LMC Average: G03",
        linewidth=3.0,
    )

    mod = G03_LMC2()
    ax.plot(
        mod_x,
        mod(mod_x),
        "y--",
        alpha=0.75,
        label=f"LMC2 Supershell: G03",
        linewidth=3.0,
    )

    mod = G24_SMCAvg()
    ax.plot(
        mod_x,
        mod(mod_x),
        "g--",
        alpha=0.75,
        label=f"SMC Average: G24",
        linewidth=3.0,
    )

    aveext.plot(
        ax,
        color="b",
        rebin_fac=rebinfac,
        wavenum=True,
        legend_key="STIS",
        legend_label=f"{klabel}: R(V) = {ave_rv:.2f}",
    )

    # rebin to a constant resolution for the tables
    # save to a simple data file: useful for DGFit
    aveext.rebin_constres("STIS", np.array([1100.0, 3200.0]) * u.AA, 25)
    waves = []
    exts = []
    uncs = []
    otypes = []
    for ckey in aveext.exts.keys():
        gvals = np.isfinite(aveext.uncs[ckey])
        waves = np.concatenate((waves, aveext.waves[ckey][gvals]))
        exts = np.concatenate((exts, aveext.exts[ckey][gvals]))
        uncs = np.concatenate((uncs, aveext.uncs[ckey][gvals]))
        if ckey == "BAND":
            otypes = np.concatenate((otypes, np.array(aveext.names[ckey])[gvals]))
        else:
            otypes = np.concatenate((otypes, np.full(np.sum(gvals), "spec")))

    # not doing this as then the FM90 fits are inconsistent.  Just explaining this in the paper
    # get the V band value to allow for explicit normalization to V band for A(l)/A(V)
    # not exactly normalized as the E(B-V) values are those from the fits, so the
    # E(lambda-V)/E(B-V) curves are such that V in not exactly 0 and B is not exactly 1
    # sindxs = np.argsort(np.absolute(waves - 0.55 * u.micron))
    # actual_av = (exts[sindxs[0]] / rv) + 1.0

    dgtab = QTable()
    sindxs = np.argsort(waves)
    dgtab["wave"] = waves[sindxs]
    dgtab["A(l)/A(V)"] = exts[sindxs]
    dgtab["unc"] = uncs[sindxs]
    dgtab["type"] = otypes[sindxs]
    dgfname = "M31_Average_Clayton25_ext.dat"
    dgtab.write(dgfname, format="ascii.commented_header", overwrite=True)

    aveext.save(ofilename.replace(".fits", "res25.fits"))

    ax.set_ylim(0.0, 7.0)
    ax.legend(loc="upper left", fontsize=0.8*fontsize)

    fig.tight_layout()

    save_str = "m31ext_ave"
    if args.png:
        fig.savefig(f"{save_str}.png")
    elif args.pdf:
        fig.savefig(f"{save_str}.pdf")
    else:
        plt.show()
