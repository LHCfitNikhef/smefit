# -*- coding: utf-8 -*-
"""Fitting the Wilson coefficients with |NS|"""
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import ultranest
from rich.style import Style
from rich.table import Table
from ultranest import stepsampler

from .. import chi2, log
from ..coefficients import CoefficientManager
from ..compute_theory import evaluate_external_coefficients
from ..loader import load_datasets
from . import Optimizer

try:
    from mpi4py import MPI

    run_parallel = True
except ModuleNotFoundError:
    run_parallel = False

_logger = log.logging.getLogger(__name__)


class USOptimizer(Optimizer):
    """Optimizer specification for Ultra nest.

    Parameters
    ----------
    loaded_datasets : `smefit.loader.DataTuple`,
        dataset tuple
    coefficients : `smefit.coefficients.CoefficientManager`
        instance of `CoefficientManager` with all the relevant coefficients to fit
    result_path: pathlib.Path
        path to result folder
    use_quad : bool
        If True use also |HO| corrections
    result_ID : str
        result name
    single_parameter_fits : bool
        True for single parameter fits
    pairwise_fits : bool
        True for pairwise parameter fits
    use_multiplicative_prescription : bool
        if True uses the multiplicative prescription for the |EFT| corrections
    live_points : int
        number of |NS| live points
    lepsilon : float
        sampling tollerance. Terminate when live point likelihoods are all the same, within Lepsilon tolerance.
        Increase this when your likelihood function is inaccurate,
    target_evidence_unc: float
        target evidence uncertainty.
    target_post_unc: float
        target posterior uncertainty.
    frac_remain: float
        integrate until this fraction of the integral is left in the remainder.
        Set to a higher number (0.5) if you know the posterior is simple.
    store_raw: bool
        if True store the result to eventually resume the job

    """

    print_rate = 5000

    def __init__(
        self,
        loaded_datasets,
        coefficients,
        result_path,
        use_quad,
        result_ID,
        single_parameter_fits,
        pairwise_fits,
        use_multiplicative_prescription,
        poly_mode,
        external_coefficients,
        live_points=500,
        lepsilon=0.001,
        target_evidence_unc=0.5,
        target_post_unc=0.5,
        frac_remain=0.01,
        store_raw=False,
        vectorized=False,
        float64=False,
        external_chi2=None,
    ):
        super().__init__(
            f"{result_path}/{result_ID}",
            loaded_datasets,
            coefficients,
            use_quad,
            single_parameter_fits,
            use_multiplicative_prescription,
            poly_mode,
            external_coefficients,
            external_chi2,
        )
        self.live_points = live_points
        self.lepsilon = lepsilon
        self.target_evidence_unc = target_evidence_unc
        self.target_post_unc = target_post_unc
        self.frac_remain = frac_remain
        self.vectorized = vectorized
        self.npar = self.free_parameters.shape[0]
        self.result_ID = result_ID
        self.pairwise_fits = pairwise_fits
        self.store_raw = store_raw

        # Set coefficients relevant quantities
        self.fixed_coeffs = self.coefficients._objlist[~self.coefficients.is_free]
        self.coeffs_index = self.coefficients._table.index

        # Ultranest requires float64 below 11 dof
        if float64 or self.npar < 11:
            jax.config.update("jax_enable_x64", True)

    @classmethod
    def from_dict(cls, config):
        """Create object from theory dictionary.

        Parameters
        ----------
        config : dict
            configuration dictionary

        Returns
        -------
        cls : Optimizer
            created object
        """

        if config.get("datasets") is not None:
            loaded_datasets = load_datasets(
                config["data_path"],
                config["datasets"],
                config["coefficients"],
                config["order"],
                config["use_quad"],
                config["use_theory_covmat"],
                config["use_t0"],
                config.get("use_multiplicative_prescription", False),
                config.get("theory_path", None),
                config.get("rot_to_fit_basis", None),
                config.get("uv_couplings", False),
                config.get("external_chi2", False),
                config.get("poly_mode", False),
                config.get("external_coefficients", {}),
            )
        elif config.get("external_chi2") is not None:
            loaded_datasets = None

        else:
            raise ValueError("No datasets or external chi2 provided")

        coefficients = CoefficientManager.from_dict(config["coefficients"])

        single_parameter_fits = config.get("single_parameter_fits", False)
        pairwise_fits = config.get("pairwise_fits", False)
        nlive = config.get("nlive", 500)

        if "nlive" not in config:
            _logger.warning(
                f"Number of live points (nlive) not set in the input card. Using default: {nlive}"
            )

        lepsilon = config.get("lepsilon", 0.05)
        if "lepsilon" not in config:
            _logger.warning(
                f"Sampling tollerance (Lepsilon) not set in the input card. Using default: {lepsilon}"
            )

        target_evidence_unc = config.get("target_evidence_unc", 0.5)
        if "target_evidence_unc" not in config:
            _logger.warning(
                f"Target Evidence uncertanty (target_evidence_unc) not set in the input card. Using default: {target_evidence_unc}"
            )

        target_post_unc = config.get("target_post_unc", 0.5)
        if "target_post_unc" not in config:
            _logger.warning(
                f"Target Posterior uncertanty (target_post_unc) not set in the input card. Using default: {target_post_unc}"
            )

        frac_remain = config.get("frac_remain", 0.01)
        if "frac_remain" not in config:
            _logger.warning(
                f"Remaining fraction (frac_remain) not set in the input card. Using default: {frac_remain}"
            )

        store_raw = config.get("store_raw", False)
        vectorized = config.get("vectorized", False)
        float64 = config.get("float64", False)

        use_multiplicative_prescription = config.get(
            "use_multiplicative_prescription", False
        )

        external_chi2 = config.get("external_chi2", None)

        return cls(
            loaded_datasets,
            coefficients,
            config["result_path"],
            config["use_quad"],
            config["result_ID"],
            single_parameter_fits,
            pairwise_fits,
            use_multiplicative_prescription,
            config["poly_mode"],
            config["external_coefficients"],
            live_points=nlive,
            lepsilon=lepsilon,
            target_evidence_unc=target_evidence_unc,
            target_post_unc=target_post_unc,
            frac_remain=frac_remain,
            store_raw=store_raw,
            vectorized=vectorized,
            float64=float64,
            external_chi2=external_chi2,
        )

    def chi2_func_ns(self, params):
        """Compute the chi2 function for |NS|.
        It is simplified with respect to the one in the Optimizer class,
        so that it can be compiled with jax.jit.
        Parameters
        ----------
        params : jnp.ndarray
            Wilson coefficients
        """

        if self.loaded_datasets is not None:
            chi2_tot = chi2.compute_chi2(
                self.loaded_datasets,
                params,
                self.use_quad,
                self.use_multiplicative_prescription,
                False,
                self.poly_mode,
            )
        else:
            chi2_tot = 0
        if self.chi2_ext is not None:
            for chi2_ext in self.chi2_ext:
                chi2_ext_i = chi2_ext(params)
                chi2_tot += chi2_ext_i
        return chi2_tot

    def compute_fixed_coeff(self, constrain, param_dict):
        """Compute the fixed coefficient."""
        temp = 0.0
        for add_factor_dict in constrain:
            free_dofs = jnp.array(
                [param_dict[fixed_name] for fixed_name in add_factor_dict]
            )
            fact_exp = jnp.array(list(add_factor_dict.values()), dtype=float)
            temp += jnp.prod(fact_exp[:, 0] * jnp.power(free_dofs, fact_exp[:, 1]))
        return temp

    def produce_all_params(self, params):
        """Produce all parameters from the free parameters.

        Parameters
        ----------
        params : jnp.ndarray
            free parameters

        Returns
        -------
        all_params : jnp.ndarray
            all parameters
        """
        is_free = self.coefficients.is_free
        num_params = self.coefficients.size

        if all(is_free):
            return params

        all_params = jnp.zeros(num_params)
        all_params = all_params.at[is_free].set(params)

        param_dict = dict(zip(self.coeffs_index, all_params))

        fixed_coefficients = [
            coeff for coeff in self.fixed_coeffs if coeff.constrain is not None
        ]

        for coefficient_fixed in fixed_coefficients:
            fixed_coeff = self.compute_fixed_coeff(
                coefficient_fixed.constrain, param_dict
            )
            fixed_index = self.coeffs_index.get_loc(coefficient_fixed.name)
            all_params = all_params.at[fixed_index].set(fixed_coeff)

        return all_params

    @partial(jax.jit, static_argnames=["self"])
    def gaussian_loglikelihood(self, params):
        """Multi gaussian log likelihood function.

        Parameters
        ----------
        params :  np.ndarray
            params prior

        Returns
        -------
        -0.5 * chi2 : np.ndarray
            multi gaussian log likelihood
        """

        all_params = self.produce_all_params(params)

        return -0.5 * self.chi2_func_ns(all_params)

    @partial(jax.jit, static_argnames=["self"])
    def flat_prior(self, hypercube):
        """Update the prior function.

        Parameters
        ----------
        hypercube : np.ndarray
            hypercube prior

        Returns
        -------
        flat_prior : np.ndarray
            updated hypercube prior
        """
        min_val = self.free_parameters.minimum.values
        max_val = self.free_parameters.maximum.values
        return hypercube * (max_val - min_val) + min_val

    def run_sampling(self):
        """Run the minimization with Ultra nest."""

        log_dir = None
        if self.store_raw:
            log_dir = self.results_path

        if self.vectorized:
            loglikelihood = jax.vmap(self.gaussian_loglikelihood)
            flat_prior = jax.vmap(self.flat_prior)
        else:
            loglikelihood = self.gaussian_loglikelihood
            flat_prior = self.flat_prior

        t1 = time.time()
        sampler = ultranest.ReactiveNestedSampler(
            self.free_parameters.index.tolist(),
            loglikelihood,
            flat_prior,
            log_dir=log_dir,
            resume=True,
            vectorized=self.vectorized,
        )
        if self.npar > 10:
            # set up step sampler. Here, we use a differential evolution slice sampler:
            sampler.stepsampler = stepsampler.SliceSampler(
                nsteps=100,
                generate_direction=stepsampler.generate_region_oriented_direction,
            )
        result = sampler.run(
            min_num_live_points=self.live_points,
            dlogz=self.target_evidence_unc,
            frac_remain=self.frac_remain,
            dKL=self.target_post_unc,
            Lepsilon=self.lepsilon,
            update_interval_volume_fraction=0.8 if self.npar > 20 else 0.2,
            max_num_improvement_loops=0,
        )

        t2 = time.time()
        rank = 0
        if run_parallel:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()

        if rank == 0:
            log.console.log(f"Time : {((t2 - t1) / 60.0):.3f} minutes")
            log.console.log(f"Number of samples: {result['samples'].shape[0]}")

            table = Table(
                style=Style(color="white"), title_style="bold cyan", title=None
            )
            table.add_column("Parameter", style="bold red", no_wrap=True)
            table.add_column("Best value")
            table.add_column("Error")
            for par, col in zip(self.free_parameters.index, result["samples"].T):
                table.add_row(f"{par}", f"{col.mean():.3f}", f"{col.std():.3f}")
            log.console.print(table)

            self.save(result)

    def save(self, result):
        """Save |NS| replicas to json inside a dictionary: {coff: [replicas values]}.

        Parameters
        ----------
        result : dict
            result dictionary

        """
        values = {}
        for c in self.coefficients.name:
            values[c] = []

        for sample in result["samples"]:
            self.coefficients.set_free_parameters(sample)
            self.coefficients.set_constraints()

            for c in self.coefficients.name:
                values[c].append(self.coefficients[c].value)

        # In single parameter fits the posteriors of different runs are merged
        # in the same file, so extra (non coefficient) keys would clash.
        if not self.single_parameter_fits:
            # posterior of the external coefficients (poly_mode)
            values.update(self.external_coefficients_values(values))

            # chi2 of each posterior sample
            values["chi2"] = self.samples_chi2(result["samples"]).tolist()

            # maximum likelihood point found by ultranest
            values["best_fit"] = self.best_fit_point(result["maximum_likelihood"])
            log.console.log(
                f"Best fit point (chi2 = {values['best_fit']['chi2']:.5g}): "
                + ", ".join(
                    f"{k} = {v:.4g}"
                    for k, v in values["best_fit"].items()
                    if k != "chi2"
                )
            )

        if self.pairwise_fits:
            posterior_file = (
                self.results_path
                / f"posterior_{self.coefficients.name[0]}_{self.coefficients.name[1]}.json"
            )
        else:
            posterior_file = self.results_path / "posterior.json"

        self.dump_posterior(posterior_file, values)

    def external_coefficients_values(self, coeff_values):
        """Evaluate the external coefficients (poly_mode) on the given coefficient values.

        Parameters
        ----------
        coeff_values : dict
            {coefficient name: value or list of values}

        Returns
        -------
        dict
            {external coefficient name: list of values (or float)}
        """
        if not self.poly_mode or not self.external_coefficients:
            return {}

        ext_values = evaluate_external_coefficients(
            self.external_coefficients, coeff_values
        )
        out = {}
        for name, val in ext_values.items():
            if name in coeff_values:
                _logger.warning(
                    f"External coefficient {name} has the same name as a fitted coefficient, "
                    f"storing it as ext_{name}"
                )
                name = f"ext_{name}"
            out[name] = val.tolist()
        return out

    def samples_chi2(self, samples, batch_size=1000):
        """Compute the total chi2 of each sample of free parameters.

        Parameters
        ----------
        samples : np.ndarray
            samples of free parameters, shape (n_samples, n_free_params)
        batch_size : int
            number of samples evaluated at once

        Returns
        -------
        np.ndarray
            chi2 = -2 log L for each sample
        """
        loglikelihood = jax.jit(jax.vmap(self.gaussian_loglikelihood))
        logl = [
            np.asarray(loglikelihood(jnp.asarray(samples[i : i + batch_size])))
            for i in range(0, samples.shape[0], batch_size)
        ]
        return -2.0 * np.concatenate(logl)

    def best_fit_point(self, maximum_likelihood):
        """Build the best fit point from the ultranest maximum likelihood info.

        Parameters
        ----------
        maximum_likelihood : dict
            ultranest ``result["maximum_likelihood"]``

        Returns
        -------
        dict
            {coefficient: value, external coefficient: value, "chi2": value}
        """
        self.coefficients.set_free_parameters(np.asarray(maximum_likelihood["point"]))
        self.coefficients.set_constraints()
        best_fit = {
            c: float(self.coefficients[c].value) for c in self.coefficients.name
        }
        best_fit.update(
            {
                k: float(v)
                for k, v in self.external_coefficients_values(best_fit).items()
            }
        )
        best_fit["chi2"] = float(-2.0 * maximum_likelihood["logl"])
        return best_fit
