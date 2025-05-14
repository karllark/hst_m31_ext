import argparse
import matplotlib.pyplot as plt
import numpy as np

from measure_extinction.stardata import StarData
from measure_extinction.extdata import ExtData

if __name__ == "__main__":

    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    fontsize = 18

    font = {"size": fontsize}
    plt.rc("font", **font)
    plt.rc("lines", linewidth=1)
    plt.rc("axes", linewidth=2)
    plt.rc("xtick.major", width=2)
    plt.rc("xtick.minor", width=2)
    plt.rc("ytick.major", width=2)
    plt.rc("ytick.minor", width=2)

    figsize = (10, 8)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    path = "/home/kgordon/Python/extstar_data/M31/"

    filename = "data/m31_exts.dat"

    f = open(filename, "r")
    file_lines = list(f)
    starnames = []
    for line in file_lines:
        if (line.find("#") != 0) & (len(line) > 0):
            name = line.rstrip()
            starnames.append(name)

    logteff_init = []
    logteff = []
    logteff_unc = []
    logg_init = []
    logg = []
    logg_unc = []
    sptype = []
    for cname in starnames:
        # get the initial values
        reddened_star = StarData(f"{cname}.dat", path=path)
        logteff_init.append(np.log10(float(reddened_star.model_params["Teff"])))
        logg_init.append(float(reddened_star.model_params["logg"]))
        sptype.append(reddened_star.sptype)

        # get the fit values
        cfile = f"exts/{cname}_mefit_ext.fits"
        edata = ExtData(filename=cfile)

        fdata = edata.fit_params["MCMC"]
        idx, = np.where(fdata["name"] == "logTeff")
        logteff.append(fdata[idx]["value"].data[0])
        logteff_unc.append(fdata[idx]["unc"].data[0])
        idx, = np.where(fdata["name"] == "logg")
        logg.append(fdata[idx]["value"].data[0])
        logg_unc.append(fdata[idx]["unc"].data[0])

    ax.errorbar(logteff_init, logg_init, fmt="ro")
    ax.errorbar(logteff, logg, xerr=logteff_unc, yerr=logg_unc, fmt="bo")

    init_fit_vals = zip(logteff_init, logteff, logg_init, logg, sptype, starnames)
    for logteff1, logteff2, logg1, logg2, csptype, cname in init_fit_vals:
        ax.plot([logteff1, logteff2], [logg1, logg2], "k:")
        ax.text(logteff2, logg2, f"{cname}/{csptype}", fontsize=0.7*fontsize, alpha=0.5)

    ax.invert_yaxis()
    ax.set_xlabel("log(Teff)")
    ax.set_ylabel("log(g)")

    fig.tight_layout()  # rect=(0.9,0.9))

    save_str = "m31_stell_param"
    if args.png:
        fig.savefig(f"figs/{save_str}.png")
    elif args.pdf:
        fig.savefig(f"figs/{save_str}.pdf")
    else:
        plt.show()
