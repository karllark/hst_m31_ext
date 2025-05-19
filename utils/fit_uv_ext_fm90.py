import matplotlib.pyplot as plt
import numpy as np
import argparse
import warnings

# from multiprocessing import Pool
# using a Pool does not work in this setup it seems

from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.fitting import fitter_to_model_params
import astropy.units as u
from astropy.table import QTable

from models_mcmc_extension import EmceeFitter

from dust_extinction.shapes import FM90_B3

# from measure_extinction.extdata import ExtData
from measure_extinction.extdata import ExtData as ExtDataStock


class ExtData(ExtDataStock):
    def get_fitdata(
        self,
        req_datasources,
        remove_uvwind_region=False,
        remove_lya_region=False,
        remove_irsblue=False,
    ):
        """
        Get the data to use in fitting the extinction curve

        Parameters
        ----------
        req_datasources : list of str
            list of data sources (e.g., ['IUE', 'BAND'])

        remove_uvwind_region : boolean, optional (default=False)
            remove the UV wind regions from the returned data

        remove_lya_region : boolean, optional (default=False)
            remove the Ly-alpha regions from the returned data

        remove_irsblue : boolean, optional (default=False)
            remove the IRS blue photometry from the returned data

        Returns
        -------
        (wave, y, y_unc) : tuple of arrays
            wave is wavelength in microns
            y is extinction (no units)
            y_unc is uncertainty on y (no units)
        """
        xdata = []
        ydata = []
        uncdata = []
        nptsdata = []
        for cursrc in req_datasources:
            if cursrc in self.waves.keys():
                if (cursrc == "BAND") & remove_irsblue:
                    ibloc = np.logical_and(
                        14.0 * u.micron <= self.waves[cursrc],
                        self.waves[cursrc] < 16.0 * u.micron,
                    )
                    self.npts[cursrc][ibloc] = 0
                xdata.append(self.waves[cursrc].to(u.micron).value)
                ydata.append(self.exts[cursrc])
                uncdata.append(self.uncs[cursrc])
                nptsdata.append(self.npts[cursrc])
        wave = np.concatenate(xdata) * u.micron
        y = np.concatenate(ydata)
        unc = np.concatenate(uncdata)
        npts = np.concatenate(nptsdata)

        # remove uv wind line regions
        x = wave.to(1.0 / u.micron, equivalencies=u.spectral())
        if remove_uvwind_region:
            npts[np.logical_and(3.55 / u.micron <= x, x < 3.58 / u.micron)] = 0
            npts[np.logical_and(3.83 / u.micron <= x, x < 3.90 / u.micron)] = 0
            npts[np.logical_and(4.17 / u.micron <= x, x < 4.22 / u.micron)] = 0
            npts[np.logical_and(4.25 / u.micron <= x, x < 4.28 / u.micron)] = 0
            npts[np.logical_and(6.4 / u.micron <= x, x < 6.6 / u.micron)] = 0
            npts[np.logical_and(7.15 / u.micron <= x, x < 7.25 / u.micron)] = 0
            npts[np.logical_and(7.45 / u.micron <= x, x < 7.55 / u.micron)] = 0
            npts[np.logical_and(7.62 / u.micron <= x, x < 7.74 / u.micron)] = 0
            npts[np.logical_and(7.90 / u.micron <= x, x < 7.95 / u.micron)] = 0
            npts[8.70 / u.micron <= x] = 0

        # remove Lya line region
        if remove_lya_region:
            npts[np.logical_and(8.05 / u.micron <= x, x < 8.4 / u.micron)] = 0

        # sort the data
        # at the same time, remove points with no data
        (gindxs,) = np.where(npts > 0)
        sindxs = np.argsort(x[gindxs])
        gindxs = gindxs[sindxs]
        wave = wave[gindxs]
        y = y[gindxs]
        unc = unc[gindxs]
        return (wave, y, unc)


if __name__ == "__main__":

    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument("extfile", help="file with extinction curve")
    parser.add_argument(
        "--nsteps", type=int, default=100, help="# of steps in MCMC chain"
    )
    parser.add_argument(
        "--burnfrac", type=float, default=0.1, help="fraction of MCMC chain to burn"
    )
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    # get a saved extinction curve
    file = args.extfile
    # file = '/home/kgordon/Python_git/spitzer_mir_ext/fits/hd147889_hd064802_ext.fits'
    ofile = file.replace(".fits", "_FM90.fits")
    ext = ExtData(filename=file)
    if "IUE" in ext.exts.keys():
        spectype = "IUE"
    else:
        spectype = "STIS"

    # get the extinction curve in alav - (using K band to extrapolate for A(V))
    #ext.trans_elv_alav()
    if ext.type == "elx":
        ext.trans_elv_elvebv()

    if ext.type == "alax":        
        for curname in ext.exts.keys():
            # only compute where there is data and exts is not zero
            gvals = (ext.exts[curname] != 0) & (ext.npts[curname] > 0)
            ext.exts[curname] = (ext.exts[curname] - 1.0) * ext.columns["RV"][0]

        ext.type = "elvebv"

    wave, y, y_unc = ext.get_fitdata(
        [spectype],
        remove_uvwind_region=True,
        remove_lya_region=True,
    )
    x = 1.0 / wave.value

    # initialize the model
    fm90_init = FM90_B3()

    # Set up the backend to save the samples for the emcee runs
    emcee_samples_file = ofile.replace(".fits", ".h5")

    # pick the fitter
    fit = LevMarLSQFitter()
    nsteps = args.nsteps
    fit3 = EmceeFitter(
        nsteps=nsteps, burnfrac=args.burnfrac, save_samples=emcee_samples_file
    )

    # modify weights to make sure the 2175 A bump is fit
    weights = np.full(len(x), 1.0 / 0.1)
    weights[(x > 4.0) & (x < 5.1)] *= 2.0

    # fit the data to the FM90 model using the fitter
    #   use the initialized model as the starting point
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        fm90_fit = fit(fm90_init, x, y, weights=weights)
        #print(fm90_fit.param_names)
        #print(fm90_fit.parameters)

        # empirically determine the uncertainties based on the best fit
        # noise generally different for IUE between NUV and FUV
        # not true for STIS, but fine to split
        dev = y - fm90_fit(x)
        estd_nuv = np.std(dev[x < 5.2])
        estd_fuv = np.std(dev[x >= 5.2])

        print(f"empirically noise (NUV/FUV): {estd_nuv} / {estd_fuv}")

        weights[x < 5.2] = weights[x < 5.2] * 0.0 + 1.0 / estd_nuv
        weights[x >= 5.2] = weights[x >= 5.2] * 0.0 + 1.0 / estd_fuv

        fm90_fit3 = fit3(fm90_fit, x, y, weights=weights)

    print("autocorr tau = ", fit3.fit_info["sampler"].get_autocorr_time(quiet=True))

    # setup the parameters for saving
    fm90_best_params = (fm90_fit.param_names, fm90_fit.parameters)
    fm90_per_param_vals = zip(
        fm90_fit3.parameters, fm90_fit3.uncs_plus, fm90_fit3.uncs_minus
    )
    fm90_per_params = (fm90_fit3.param_names, list(fm90_per_param_vals))

    fit_params = {}

    otab = QTable()
    otab["name"] = fm90_fit.param_names
    otab["value"] = fm90_fit.parameters
    fit_params["MIN"] = otab

    otab2 = QTable()
    otab2["name"] = fm90_fit3.param_names
    otab2["value"] = fm90_fit3.parameters
    otab2["unc"] = 0.5 * (fm90_fit3.uncs_plus + fm90_fit3.uncs_minus)
    fit_params["MCMC"] = otab2

    print(fit_params)

    # save extinction and fit parameters
    #if "AV" not in ext.columns.keys():
    #    ext.calc_AV()
    #column_info = {"ebv": ext.columns["EBV"][0], "av": ext.columns["AV"][0]}
    ext.save(
        ofile,
        fit_params=fit_params,
    )

    # make the standard mcmc plots
    fit3.plot_emcee_results(fm90_fit3, filebase=ofile.replace(".fits", ""))

    # plot the observed data, initial guess, and final fit
    fig, fax = plt.subplots(
        nrows=2, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [5, 1]}
    )

    # remove pesky x without units warnings
    x /= u.micron

    # ax.plot(x[gindxs], fm90_init(x[gindxs]), label='Initial guess')
    ax = fax[0]
    ax.plot(x, y, label="Observed Curve")
    ax.plot(x, fm90_fit3(x), label="emcee")
    ax.plot(x, fm90_fit(x), label="LevMarLSQ")

    # plot samples from the mcmc chaing
    flat_samples = fit3.fit_info["sampler"].get_chain(
        discard=int(0.1 * nsteps), flat=True
    )
    inds = np.random.randint(len(flat_samples), size=100)
    model_copy = fm90_fit3.copy()
    for ind in inds:
        sample = flat_samples[ind]
        fitter_to_model_params(model_copy, sample)
        ax.plot(x, model_copy(x), "C1", alpha=0.05)

    ax.set_xlabel(r"$x$ [$\mu m^{-1}$]")
    ax.set_ylabel(r"$E(\lambda - V)/E(B - V)$")
    ax.set_ylim(0, 15)

    ax.set_title(file)

    ax.legend(loc="best")

    # residuals
    ax = fax[1]
    ax.plot(x, np.zeros((len(x))), "k--")
    ax.plot(x, y - fm90_fit(x))
    ax.set_ylim(np.array([-1.0, 1.0]) * 1.0)

    plt.tight_layout()

    # plot or save to a file
    outname = ofile.replace(".fits", "")
    if args.png:
        fig.savefig(outname + ".png")
    elif args.pdf:
        fig.savefig(outname + ".pdf")
    else:
        plt.show()
