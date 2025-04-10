import argparse
import glob
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.table import QTable

from measure_extinction.stardata import StarData
from measure_extinction.extdata import ExtData
from measure_extinction.modeldata import ModelData
from measure_extinction.model import MEModel

import os

os.environ["OMP_NUM_THREADS"] = "1"


def fit_model_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("starname", help="Name of star")
    parser.add_argument(
        "--path",
        help="Path to star data",
        default="/home/kgordon/Python/extstar_data/DAT_files/",
    )
    parser.add_argument(
        "--modtype",
        help="Pick the type of model grid",
        choices=["obstars", "whitedwarfs"],
        default="obsstars",
    )
    parser.add_argument(
        "--wind", help="add IR (lambda > 1 um) wind emission model", action="store_true"
    )
    parser.add_argument(
        "--modpath",
        help="path to the model files",
        default="/home/kgordon/Python/extstar_data/Models/",
    )
    parser.add_argument(
        "--modstr",
        help="Alternative model string for grid (expert)",
        default="",
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
    parser.add_argument(
        "--showfit", help="display the best fit model plot", action="store_true"
    )
    return parser


def main():
    parser = fit_model_parser()
    args = parser.parse_args()

    outname = f"figs/{args.starname}_mefit"
    resid_range = 50.0
    lyaplot = True

    # WISCI
    # only_bands = ["B", "V", "R", "I", "J", "H", "K"]
    # only_bands = ["J", "H", "K"]
    only_bands = None

    # get data
    fstarname = f"{args.starname}.dat"
    reddened_star = StarData(fstarname, path=f"{args.path}", only_bands=only_bands)

    # read in the foreground HI values
    htab = QTable.read("data/hi_foregrounds_m31_id.fits")
    gvals = args.starname.split("_")[1] == htab["star"]
    forehi = htab["nhi"][gvals][0] * 1e20
    forehi_unc = htab["nhi_err"][gvals][0] * 1e20

    # remove low S/N STIS data - affected by systematics
    sn_cut = 1.5
    snr = reddened_star.data["STIS"].fluxes / reddened_star.data["STIS"].uncs
    bvals = np.logical_and(
        snr < sn_cut, reddened_star.data["STIS"].waves > 0.17 * u.micron
    )
    reddened_star.data["STIS"].npts[bvals] = 0
    reddened_star.data["STIS"].fluxes[bvals] = 0

    band_names = None
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
        # tlusty_models_fullpath = glob.glob(f"{args.modpath}/{modstr}*.dat")
        # fmt: off
        tlusty_models_fullpath = (glob.glob(f"{args.modpath}/tlusty_z02*.dat")
                                  + glob.glob(f"{args.modpath}/tlusty_z05*.dat")
                                  + glob.glob(f"{args.modpath}/tlusty_z10*.dat")
                                  + glob.glob(f"{args.modpath}/tlusty_z20*.dat"))
        # fmt: on
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

    # setup the model
    # memod = MEModel(modinfo=modinfo, obsdata=reddened_star)  # use to activate logf fitting
    memod = MEModel(modinfo=modinfo, obsdata=reddened_star)

    if "Teff" in reddened_star.model_params.keys():
        memod.logTeff.value = np.log10(float(reddened_star.model_params["Teff"]))
        if "Teff_unc" in reddened_star.model_params.keys():
            memod.logTeff.prior = (
                memod.logTeff.value,
                float(reddened_star.model_params["Teff_unc"])
                / (float(reddened_star.model_params["Teff"]) * np.log(10.0)),
            )
        else:
            memod.logTeff.fixed = True
    if "logg" in reddened_star.model_params.keys():
        memod.logg.value = float(reddened_star.model_params["logg"])
        if "logg_unc" in reddened_star.model_params.keys():
            memod.logg.prior = (
                memod.logg.value,
                float(reddened_star.model_params["logg_unc"]),
            )
        else:
            memod.logg.fixed = True
    if "Z" in reddened_star.model_params.keys():
        memod.logZ.value = np.log10(float(reddened_star.model_params["Z"]))
        memod.logZ.fixed = True
    if "velocity" in reddened_star.model_params.keys():
        memod.velocity.value = float(reddened_star.model_params["velocity"])
        memod.velocity.fixed = True

    # add exclude regions
    exreg = {
        "e1": [0.17, 0.2],
        "e2": [0.1701, 0.1702],
        "e3": [0.1701, 0.1702],
        "e4": [0.17, 0.2],
        "e5": [0.18, 0.2],
        "e6": [0.183, 0.2],
        "e7": [0.17, 0.19],
        "e8": [0.162, 0.2],
        "e9": [0.1701, 0.1702],
        "e12": [0.165, 0.205],
        "e13": [0.165, 0.205],  # [.137, .143, .165, .205]
        "e14": [0.165, 0.21],
        "e15": [0.183, 0.2],
        "e17": [0.185, 0.19],
        "e18": [0.18, 0.21],
        "e22": [0.17, 0.19],
        "e24": [0.18, 0.2],
    }

    snum = (args.starname.split("_"))[1]
    memod.add_exclude_region(np.flip(1.0 / (np.array(exreg[snum]) * u.micron)))

    memod.fit_weights(reddened_star)

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

    # setup HI values
    memod.logHI_exgal.value = np.log10(1.61e21 * memod.Av.value)
    memod.logHI_exgal.fixed = False
    memod.vel_exgal.value = -10.0
    memod.vel_exgal.fixed = False

    memod.logHI_MW.value = np.log10(forehi)
    memod.fore_Av.value = forehi / (8.3e21 / 3.1)
    memod.logHI_MW.fixed = True
    memod.vel_MW.fixed = True

    # set velocities to non-zero to help fitting
    memod.velocity.value = -300.0
    memod.velocity.fixed = True

    # for M31
    # memod.logZ.fixed = True
    # memod.velocity.fixed = True
    memod.logTeff.fixed = False
    memod.logTeff.prior = (memod.logTeff.value, 0.05)

    memod.logg.fixed = False
    memod.logg.prior = (memod.logg.value, 0.1)

    memod.logZ.value = 0.0
    memod.logZ.prior = (0.0, 0.2)

    # memod.C2.fixed = True
    # memod.B3.fixed = True
    # memod.C4.fixed = True
    # memod.xo.fixed = True
    # memod.gamma.fixed = True
    # memod.vel_MW.fixed = True
    # memod.logHI_MW.fixed = True

    memod.set_initial_norm(reddened_star, modinfo)

    # dictonary for fit parameter tables
    fit_params = {}

    print("initial parameters")
    memod.pprint_parameters()

    # memod.plot(reddened_star, modinfo, resid_range=resid_range, lyaplot=lyaplot)
    # plt.show()
    # exit()

    start_time = time.time()
    print("starting fitting")

    fitmod, result = memod.fit_minimizer(reddened_star, modinfo, maxiter=10000)

    print("finished fitting")
    print("--- %s seconds ---" % (time.time() - start_time))
    # check the fit output
    print(result["message"])

    print("best parameters")
    fitmod.pprint_parameters()
    fit_params["MIN"] = fitmod.save_parameters()

    dust_columns = {"AV": (fitmod.Av.value, 0.0), "RV": (fitmod.Rv.value, 0.0)}

    fitmod.plot(reddened_star, modinfo, resid_range=resid_range, lyaplot=lyaplot)
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
        fit_params["MCMC"] = fitmod2.save_parameters()

        dust_columns = {
            "AV": (fitmod2.Av.value, fitmod2.Av.unc),
            "RV": (fitmod2.Rv.value, fitmod2.Rv.unc),
        }

        fitmod2.plot(reddened_star, modinfo, resid_range=resid_range, lyaplot=lyaplot)
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
    if "BAND" in reddened_star.data.keys():
        modinfo.band_names = reddened_star.data["BAND"].get_band_names()
    modsed_stardata = modinfo.SED_to_StarData(modsed)

    # create an extincion curve and save it
    extdata = ExtData()
    # get the reddened star data again to have all the possible spectra
    reddened_star_full = StarData(fstarname, path=f"{args.path}", only_bands=only_bands)
    extdata.calc_elx(reddened_star_full, modsed_stardata, rel_band="ACS_F475W")
    extdata.columns = dust_columns
    extdata.save(f"{outname.replace("figs", "exts")}_ext.fits", fit_params=fit_params)

    if args.showfit:
        fitmod.plot(reddened_star, modinfo, resid_range=resid_range, lyaplot=lyaplot)
        plt.show()


if __name__ == "__main__":
    main()
