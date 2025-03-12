import argparse
import glob
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from measure_extinction.stardata import StarData
from measure_extinction.extdata import ExtData
from measure_extinction.modeldata import ModelData
from measure_extinction.model import MEModel

import os

os.environ["OMP_NUM_THREADS"] = "1"


def fit_model_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("starname", help="Name of star")
    parser.add_argument("--path", help="Path to star data", default="/home/kgordon/Python/extstar_data/DAT_files/")
    parser.add_argument(
        "--modtype",
        help="Pick the type of model grid",
        choices=["obstars", "whitedwarfs"],
        default="obsstars",
    )
    parser.add_argument("--wind", help="add IR (lambda > 1 um) wind emission model", action="store_true")
    parser.add_argument("--modpath", help="path to the model files", default="/home/kgordon/Python/extstar_data/Models/")
    parser.add_argument(
        "--modstr", help="Alternative model string for grid (expert)", default="tlusty_z100"
    )
    parser.add_argument(
        "--picmodel",
        help="Set to read model grid from pickle file",
        action="store_true",
    )
    parser.add_argument(
        "--Av_init", help="initial A(V) for fitting", default=1.0, type=float
    )    
    parser.add_argument("--mcmc", help="run EMCEE MCMC fitting", action="store_true")
    parser.add_argument(
        "--mcmc_nsteps", help="number of MCMC steps", default=1000, type=int
    )
    parser.add_argument("--showfit", help="display the best fit model plot", action="store_true")
    return parser


def main():
    parser = fit_model_parser()
    args = parser.parse_args()

    outname = f"figs/{args.starname}_mefit"

    # WISCI
    # only_bands = ["B", "V", "R", "I", "J", "H", "K"]
    # only_bands = ["J", "H", "K"]
    only_bands = None

    # get data
    fstarname = f"{args.starname}.dat"
    reddened_star = StarData(fstarname, path=f"{args.path}", only_bands=only_bands)

    if "BAND" in reddened_star.data.keys():
        band_names = reddened_star.data["BAND"].get_band_names()
    else:
        band_names = []
    data_names = list(reddened_star.data.keys())

    # model data
    start_time = time.time()
    print("reading model files")
    if args.modtype == "whitedwarfs":
        modstr = "wd_hubeny_"
    else:
        modstr = "tlusty_"
    if args.modstr is not None:
        modstr = args.modstr
    if args.picmodel:
        modinfo = pickle.load(open(f"{modstr}_modinfo.p", "rb"))
    else:
        tlusty_models_fullpath = glob.glob(f"{args.modpath}/{modstr}*.dat")
        tlusty_models = [
            tfile[tfile.rfind("/") + 1 : len(tfile)] for tfile in tlusty_models_fullpath
        ]
        if len(tlusty_models) > 1:
            print(f"{len(tlusty_models)} model files found.")
        else:
            raise ValueError("no model files found.")

        # get the models with just the reddened star band data and spectra
        modinfo = ModelData(
            tlusty_models,
            path=f"{args.modpath}/",
            band_names=band_names,
            spectra_names=data_names,
        )
        pickle.dump(modinfo, open(f"{modstr}_modinfo.p", "wb"))
    print("finished reading model files")
    print("--- %s seconds ---" % (time.time() - start_time))

    # for wisci (only fit to get the stellar parameters)
    #for ckey in list(reddened_star.data.keys()):
    #    if ckey not in ["BAND", "STIS_Opt"]:
    #        del reddened_star.data[ckey]

    # setup the model
    # memod = MEModel(modinfo=modinfo, obsdata=reddened_star)  # use to activate logf fitting
    memod = MEModel(modinfo=modinfo)

    if "Teff" in reddened_star.model_params.keys():
        memod.logTeff.value = np.log10(float(reddened_star.model_params["Teff"]))
        memod.logTeff.fixed = True
    if "logg" in reddened_star.model_params.keys():
        memod.logg.value = float(reddened_star.model_params["logg"])
        memod.logg.fixed = True
    if "Z" in reddened_star.model_params.keys():
        memod.logZ.value = np.log10(float(reddened_star.model_params["Z"]))
        memod.logZ.fixed = True
    if "velocity" in reddened_star.model_params.keys():
        memod.velocity.value = float(reddened_star.model_params["velocity"])
        memod.velocity.fixed = True

    memod.fit_weights(reddened_star)
    memod.weights["STIS"] *= 100.0

    if args.modtype == "whitedwarfs":
        memod.vturb.value = 0.0
        memod.vturb.fixed = True
        memod.Av.value = 0.5
        memod.weights["BAND"] *= 10.0
        memod.weights["STIS"] *= 10.0
    if args.wind:
        memod.windamp.value = 1e-3
        memod.windamp.fixed = False
        memod.windalpha.fixed = False

    memod.Av.value = args.Av_init

    # for wisci
    memod.velocity.fixed = True
    memod.logTeff.fixed = False
    memod.logg.fixed = False
    memod.velocity.fixed = False
    # memod.C2.fixed = True
    # memod.B3.fixed = True
    # memod.C4.fixed = True
    # memod.xo.fixed = True
    # memod.gamma.fixed = True
    # memod.vel_MW.fixed = True
    # memod.logHI_MW.fixed = True

    memod.set_initial_norm(reddened_star, modinfo)

    print("initial parameters")
    memod.pprint_parameters()

    start_time = time.time()
    print("starting fitting")

    fitmod, result = memod.fit_minimizer(reddened_star, modinfo, maxiter=10000)

    print("finished fitting")
    print("--- %s seconds ---" % (time.time() - start_time))
    # check the fit output
    print(result["message"])

    print("best parameters")
    fitmod.pprint_parameters()

    fitmod.plot(reddened_star, modinfo)
    plt.savefig(f"{outname}_minimizer.pdf")
    plt.savefig(f"{outname}_minimizer.png")
    plt.close()

    if args.mcmc:
        print("starting sampling")
        # using an MCMC sampler to define nD probability function
        # use best fit result as the starting point
        fitmod2, flat_samples, sampler = fitmod.fit_sampler(
            reddened_star,
            modinfo,
            nsteps=args.mcmc_nsteps,
            save_samples=f"{outname.replace("figs", "exts")}_.h5",
        )

        print("finished sampling")

        print("p50 parameters")
        fitmod2.pprint_parameters()

        fitmod2.plot(reddened_star, modinfo)
        plt.savefig(f"{outname}_mcmc.pdf")
        plt.savefig(f"{outname}_mcmc.png")
        plt.close()

        fitmod2.plot_sampler_chains(sampler)
        plt.savefig(f"{outname}_mcmc_chains.pdf")
        plt.savefig(f"{outname}_mcmc_chains.png")
        plt.close()

        fitmod2.plot_sampler_corner(flat_samples)
        plt.savefig(f"{outname}_mcmc_corner.pdf")
        plt.savefig(f"{outname}_mcmc_corner.png")
        plt.close()

        fitmod = fitmod2

    # create a stardata object with the best intrinsic (no extinction) model
    modsed = fitmod.stellar_sed(modinfo)
    modsed_stardata = modinfo.SED_to_StarData(modsed)

    # create an extincion curve and save it
    extdata = ExtData()
    # get the reddened star data again to have all the possible spectra
    reddened_star_full = StarData(fstarname, path=f"{args.path}", only_bands=only_bands)
    extdata.calc_elx(reddened_star_full, modsed_stardata, rel_band="ACS_F475W")
    col_info = {"av": fitmod.Av.value, "rv": fitmod.Rv.value}
    extdata.columns = {"AV": (fitmod.Av.value, 0.0), "RV": (fitmod.Rv.value, 0.0)}
    extdata.save(f"{outname.replace("figs", "exts")}_ext.fits", column_info=col_info)

    if args.showfit:
        fitmod.plot(reddened_star, modinfo)
        plt.show()


if __name__ == "__main__":
    main()
