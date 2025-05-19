import matplotlib.pyplot as plt
import numpy as np

from astropy.modeling.fitting import (
    Fitter,
    _validate_model,
    fitter_to_model_params,
    model_to_fit_params,
    _convert_input,
)
from astropy.modeling.optimizers import Optimization
from astropy.modeling.statistic import leastsquare
from astropy import uncertainty as astrounc

import emcee
import corner


all = ["EmceeOpt", "EmceeFitter", "plot_emcee_results"]


class EmceeOpt(Optimization):
    """
    Interface to emcee sampler.
    """

    supported_constraints = ["bounds", "fixed", "tied"]

    def __init__(self):
        super().__init__(emcee)
        self.fit_info = {"perparams": None, "samples": None, "sampler": None}

    @staticmethod
    def _get_best_fit_params(sampler):
        """
        Determine the best fit parameters given an emcee sampler object
        """
        # very likely a faster way
        max_lnp = -1e6
        nwalkers, nsteps = sampler.lnprobability.shape
        for k in range(nwalkers):
            tmax_lnp = np.nanmax(sampler.lnprobability[k])
            if tmax_lnp > max_lnp:
                max_lnp = tmax_lnp
                (indxs,) = np.where(sampler.lnprobability[k] == tmax_lnp)
                fit_params_best = sampler.chain[k, indxs[0], :]

        return fit_params_best

    def __call__(self, objfunc, initval, fargs, nsteps, save_samples=None, **kwargs):
        """
        Run the sampler.

        Parameters
        ----------
        objfunc : callable
            objection function
        initval : iterable
            initial guess for the parameter values
        fargs : tuple
            other arguments to be passed to the statistic function
        kwargs : dict
            other keyword arguments to be passed to the solver
        """
        # optresult = self.opt_method(objfunc, initval, args=fargs)
        # fitparams = optresult['x']

        ndim = len(initval)
        nwalkers = 2 * ndim
        pos = initval + 1e-4 * np.random.randn(nwalkers, ndim)

        # ensure all the walkers start within the bounds
        model = fargs[0]
        for cp in pos:
            k = 0
            for cname in model.param_names:
                if not model.fixed[cname]:
                    if model.bounds[cname][0] is not None:
                        if cp[k] < model.bounds[cname][0]:
                            cp[k] = model.bounds[cname][0]
                    if model.bounds[cname][1] is not None:
                        if cp[k] > model.bounds[cname][1]:
                            cp[k] = model.bounds[cname][1]
                    # only non-fixed parameters are in initval
                    # so only increment k when non-fixed
                    k += 1

        # Set up the backend
        if save_samples:
            # Don't forget to clear it in case the file already exists
            save_backend = emcee.backends.HDFBackend(save_samples)
            save_backend.reset(nwalkers, ndim)

        sampler = self.opt_method.EnsembleSampler(
            nwalkers, ndim, objfunc, backend=save_backend, args=fargs
        )
        sampler.run_mcmc(pos, nsteps, progress=True)
        samples = sampler.get_chain()

        fitparams = self._get_best_fit_params(sampler)
        self.fit_info["sampler"] = sampler
        self.fit_info["samples"] = samples

        return fitparams, self.fit_info


class EmceeFitter(Fitter):
    """
    Use emcee and least squares statistic
    """

    def __init__(self, nsteps=100, burnfrac=0.1, save_samples=None):
        super().__init__(optimizer=EmceeOpt, statistic=leastsquare)
        self.nsteps = nsteps
        self.burnfrac = burnfrac
        self.fit_info = {}
        self.save_samples = save_samples

    # add lnlike and lnprior and have log_probability just be the combo of the two
    def log_prior(self, fps, *args):
        """
        Computes the natural log of the prior.
        Currently only handles flat priors set using parameter bounds.

        Parameters
        ----------
        fps : list
            parameters returned by the fitter
        args : list
            [model, [other_args], [input coordinates]]
            other_args may include weights or any other quantities specific for
            a statistic

        Returns
        -------
        log(prior) : float
            natural log of the prior probability
        """
        # pameter bound priors = flat priors between two limits
        #   EMCEE uses an explicit return of -np.inf to designate such bounds
        # need to be handled explicitly to get good sampler chains
        #   standard astropy modeling fitting results in chains not accurately
        #   reflecting the bounds
        model = args[0]
        k = 0
        for cname in model.param_names:
            if not model.fixed[cname]:
                if model.bounds[cname][0] is not None:
                    if fps[k] < model.bounds[cname][0]:
                        return -np.inf
                if model.bounds[cname][1] is not None:
                    if fps[k] > model.bounds[cname][1]:
                        return -np.inf
                k += 1

        # no other priors, so return 0.0 = log(1.0)
        return 0.0

    def log_likelihood(self, fps, *args):
        """
        Computes the natural log of the likelihood.

        Parameters
        ----------
        fps : list
            parameters returned by the fitter
        args : list
            [model, [other_args], [input coordinates]]
            other_args may include weights or any other quantities specific for
            a statistic

        Returns
        -------
        log(likelihood) : float
            natural log of the likelihood probability
        """
        # assume the standard leastsquare
        res = self.objective_function(fps, *args)

        # convert to a log value - assumes chisqr/Gaussian unc model
        return -0.5 * res

    def log_probability(self, fps, *args):
        """
        Compute the natural log of the probability by combining the
        likelihood and prior probabilties.

        Parameters
        ----------
        fps : list
            parameters returned by the fitter
        args : list
            [model, [other_args], [input coordinates]]
            other_args may include weights or any other quantities specific for
            a statistic

        Returns
        -------
        log(prob) : float
            natural log of the probability

        Notes
        -----
        The list of arguments (args) is set in the `__call__` method.
        Fitters may overwrite this method, e.g. when statistic functions
        require other arguments.
        """
        lp = self.log_prior(fps, *args)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(fps, *args)

    def _set_uncs_and_posterior(self, model):
        """
        Set the symmetric and asymmetric Gaussian uncertainties
        and sets the posteriors to astropy.unc distributions

        Parameters
        ----------
        model : astropy model
            model giving the result from the fitting

        Returns
        -------
        model : astropy model
            model updated with uncertainties
        """
        sampler = self.fit_info["sampler"]
        nwalkers, nsteps = sampler.lnprobability.shape
        # discard the 1st burn_frac (burn in)
        flat_samples = sampler.get_chain(discard=int(self.burnfrac * nsteps), flat=True)
        nflatsteps, ndim = flat_samples.shape

        nparams = len(model.parameters)
        model.uncs = np.zeros((nparams))
        model.uncs_plus = np.zeros((nparams))
        model.uncs_minus = np.zeros((nparams))
        k = 0
        for i, pname in enumerate(model.param_names):
            if not model.fixed[pname]:
                mcmc = np.percentile(flat_samples[:, k], [16, 50, 84])

                # set the uncertainty arrays - could be done via the parameter objects
                # but would need an update to the model properties to make this happen
                # model.parameters[i] = mcmc[1]
                model.uncs[i] = 0.5 * (mcmc[2] - mcmc[0])
                model.uncs_plus[i] = mcmc[2] - mcmc[1]
                model.uncs_minus[i] = mcmc[1] - mcmc[0]

                # set the posterior distribution to the samples
                param = getattr(model, pname)
                param.value = mcmc[1]
                param.posterior = astrounc.Distribution(flat_samples[:, k])
                k += 1
            else:
                model.uncs[i] = 0.0
                model.uncs_plus[i] = 0.0
                model.uncs_minus[i] = 0.0
                # set the posterior distribution to the samples
                param = getattr(model, pname)
                param.posterior = None

            # now set uncertainties on the parameter objects themselves
            param = getattr(model, pname)
            param.unc = model.uncs[i]
            param.unc_plus = model.uncs_plus[i]
            param.unc_minus = model.uncs_minus[i]

        return model

    def __call__(self, model, x, y, weights=None, **kwargs):
        """
        Fit data to this model.

        Parameters
        ----------
        model : `~astropy.modeling.FittableModel`
            model to fit to x, y
        x : array
            input coordinates
        y : array
            input coordinates
        weights : array, optional
            Weights for fitting.
            For data with Gaussian uncertainties, the weights should be
            1/sigma.
        kwargs : dict
            optional keyword arguments to be passed to the optimizer or the statistic

        Returns
        -------
        model_copy : `~astropy.modeling.FittableModel`
            a copy of the input model with parameters set by the fitter
        """

        model_copy = _validate_model(model, self._opt_method.supported_constraints)
        farg = _convert_input(x, y)
        farg = (model_copy, weights) + farg
        p0, _, _ = model_to_fit_params(model_copy)

        fitparams, self.fit_info = self._opt_method(
            self.log_probability,
            p0,
            farg,
            self.nsteps,
            save_samples=self.save_samples,
            **kwargs
        )

        # set the output model parameters to the "best fit" parameters
        fitter_to_model_params(model_copy, fitparams)

        # get and set the symmetric and asymmetric uncertainties on each parameter
        model_copy = self._set_uncs_and_posterior(model_copy)

        return model_copy

    def plot_emcee_results(self, fitted_model, filebase=""):
        """
        Plot the standard triangle and diagnostic walker plots
        """
        # get the samples to use
        sampler = self.fit_info["sampler"]

        # only the non fixed parameters were fit
        fit_param_names = []
        for pname in fitted_model.param_names:
            if not fitted_model.fixed[pname]:
                fit_param_names.append(pname)

        # plot the walker chains for all parameters
        nwalkers, nsteps, ndim = sampler.chain.shape
        fig, ax = plt.subplots(ndim, sharex=True, figsize=(13, 13))
        walk_val = np.arange(nsteps)
        for i in range(ndim):
            for k in range(nwalkers):
                ax[i].plot(walk_val, sampler.chain[k, :, i], "-")
                ax[i].set_ylabel(fit_param_names[i])
        fig.savefig("%s_walker_param_values.png" % filebase)
        plt.close(fig)

        # plot the 1D and 2D likelihood functions in a traditional triangle plot
        nwalkers, nsteps = sampler.lnprobability.shape
        # discard the 1st burn_frac (burn in)
        flat_samples = sampler.get_chain(discard=int(self.burnfrac * nsteps), flat=True)
        nflatsteps, ndim = flat_samples.shape
        fig = corner.corner(
            flat_samples,
            labels=fit_param_names,
            show_titles=True,
            title_fmt=".3f",
            use_math_text=True,
        )
        fig.savefig("%s_param_triangle.png" % filebase)
        plt.close(fig)
