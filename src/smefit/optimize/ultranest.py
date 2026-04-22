# -*- coding: utf-8 -*-
"""Fitting the Wilson coefficients with |NS|"""
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import ultranest
from jax.extend import backend as jbackend
from rich.style import Style
from rich.table import Table
from ultranest import stepsampler

from smefit.rge.rge import load_rge_matrix

from .. import chi2, log
from ..coefficients import CoefficientManager
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
    vectorized: bool
        if True use jax vectorization
    external_chi2: dict
        dict of external chi2
    rgemat: numpy.ndarray
        solution matrix of the RGE
    rge_dict: dict
        dictionary with the RGE input parameter options
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
        live_points=500,
        lepsilon=0.001,
        target_evidence_unc=0.5,
        target_post_unc=0.5,
        frac_remain=0.01,
        n_samples=10000,
        cluster_num_live_points=None,
        store_raw=False,
        vectorized=False,
        external_chi2=None,
        rgemat=None,
        rge_dict=None,
    ):
        super().__init__(
            results_path=f"{result_path}/{result_ID}",
            loaded_datasets=loaded_datasets,
            coefficients=coefficients,
            use_quad=use_quad,
            single_parameter_fits=single_parameter_fits,
            use_multiplicative_prescription=use_multiplicative_prescription,
            external_chi2=external_chi2,
            rgemat=rgemat,
            rge_dict=rge_dict,
        )
        self.live_points = live_points
        self.lepsilon = lepsilon
        self.target_evidence_unc = target_evidence_unc
        self.target_post_unc = target_post_unc
        self.frac_remain = frac_remain
        self.n_samples = n_samples
        self.vectorized = vectorized
        self.npar = self.free_parameters.shape[0]
        self.cluster_num_live_points = (
            cluster_num_live_points
            if cluster_num_live_points is not None
            else 3 * self.npar
        )
        self.result_ID = result_ID
        self.pairwise_fits = pairwise_fits
        self.store_raw = store_raw

        # Set coefficients relevant quantities
        self.fixed_coeffs = self.coefficients._objlist[~self.coefficients.is_free]
        self.coeffs_index = self.coefficients._table.index

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

        rge_dict = config.get("rge", None)
        operators_to_keep = config["coefficients"]
        cutoff_scale = config.get("cutoff_scale", None)

        if rge_dict is not None and config.get("datasets") is not None:
            _logger.info("Loading RGE matrix")
            save_path = Path(config["result_path"]) / config["result_ID"]
            rgemat, operators_to_keep = load_rge_matrix(
                rge_dict,
                list(operators_to_keep.keys()),
                config["datasets"],
                config.get("theory_path", None),
                cutoff_scale,
                save_path=save_path,
            )
            _logger.info("The operators generated by the RGE are: ")
            _logger.info(list(operators_to_keep.keys()))
        else:
            rgemat = None

        if config.get("datasets") is not None:
            loaded_datasets = load_datasets(
                config["data_path"],
                config["datasets"],
                operators_to_keep,
                config["use_quad"],
                config["use_theory_covmat"],
                config["use_t0"],
                config.get("use_multiplicative_prescription", False),
                config.get("default_order", "LO"),
                config.get("theory_path", None),
                config.get("rot_to_fit_basis", None),
                config.get("uv_couplings", False),
                config.get("external_chi2", False),
                rgemat=rgemat,
                cutoff_scale=cutoff_scale,
            )
        elif config.get("external_chi2") is not None:
            loaded_datasets = None

        else:
            raise ValueError("No datasets or external chi2 provided")

        coefficients = CoefficientManager.from_dict(config["coefficients"])

        single_parameter_fits = config.get("single_parameter_fits", False)
        pairwise_fits = config.get("pairwise_fits", False)
        nlive = config.get("nlive", 500)
        cluster_num_live_points = config.get("cluster_num_live_points", None)

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
        n_samples = config.get("n_samples", 10000)
        if "n_samples" not in config:
            _logger.warning(
                f"Number of samples (n_samples) not set in the input card. "
                f"Using default: {n_samples}"
            )

        store_raw = config.get("store_raw", False)
        vectorized = config.get("vectorized", False)

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
            live_points=nlive,
            cluster_num_live_points=cluster_num_live_points,
            lepsilon=lepsilon,
            target_evidence_unc=target_evidence_unc,
            target_post_unc=target_post_unc,
            frac_remain=frac_remain,
            n_samples=n_samples,
            store_raw=store_raw,
            vectorized=vectorized,
            external_chi2=external_chi2,
            rgemat=rgemat,
            rge_dict=rge_dict,
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
                use_replica=False,
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
    def gaussian_loglikelihood_jit(self, params):
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

    def gaussian_loglikelihood(self, params):
        """Gaussian log likelihood wrapper.

        This keeps the JIT path for pure SMEFiT χ², while allowing external
        likelihoods (e.g. smelli) to run in eager mode.
        """

        if self.chi2_ext is None:
            return self.gaussian_loglikelihood_jit(params)

        all_params = self.produce_all_params(params)
        return -0.5 * self.chi2_func_ext_ns(all_params)

    def chi2_func_ext_ns(self, params):
        """Compatibility χ² path for non-JAX external likelihood contributions."""
        if self.loaded_datasets is not None:
            chi2_tot = chi2.compute_chi2(
                self.loaded_datasets,
                params,
                self.use_quad,
                self.use_multiplicative_prescription,
                use_replica=False,
            )
        else:
            chi2_tot = 0

        for chi2_ext in self.chi2_ext:
            chi2_ext_i = chi2_ext(params)
            chi2_tot += chi2_ext_i

        return chi2_tot

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
            log_dir = self.results_path / "ultranest_log"

        if self.vectorized and self.chi2_ext is None:
            loglikelihood = jax.vmap(self.gaussian_loglikelihood)
            flat_prior = jax.vmap(self.flat_prior)
        else:
            loglikelihood = self.gaussian_loglikelihood
            flat_prior = self.flat_prior
        ns_vectorized = self.vectorized and self.chi2_ext is None

        _logger.info(f"Running fit with backend: {jbackend.get_backend().platform}")
        t1 = time.time()
        sampler = ultranest.ReactiveNestedSampler(
            self.free_parameters.index.tolist(),
            loglikelihood,
            flat_prior,
            log_dir=log_dir,
            resume=True,
            vectorized=ns_vectorized,
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
            cluster_num_live_points=self.cluster_num_live_points,
        )

        t2 = time.time()

        # Keep only the requested number of samples.
        # If they are less, it keeps all of them.
        if len(result["samples"]) < self.n_samples:
            _logger.warning(
                f"Only {len(result['samples'])} samples were produced, "
                f"less than the requested {self.n_samples}"
            )
        result["samples"] = result["samples"][: self.n_samples]

        rank = 0
        if run_parallel:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()

        if rank == 0:
            log.console.log(f"Time : {((t2 - t1) / 60.0):.3f} minutes")
            log.console.log(f"Number of samples: {result['samples'].shape[0]}")

            self.save(result)

            if self.store_raw:
                _logger.info("Ultranest plots being produced...")
                sampler.plot()
                _logger.info(f"Ultranest plots produced in {log_dir}")

            table = Table(
                style=Style(color="white"), title_style="bold cyan", title=None
            )
            table.add_column("Parameter", style="bold red", no_wrap=True)
            table.add_column("Best value")
            table.add_column("Error")
            for par, col in zip(self.free_parameters.index, result["samples"].T):
                table.add_row(f"{par}", f"{col.mean():.3f}", f"{col.std():.3f}")
            log.console.print(table)

    def save(self, result):
        """Save |NS| replicas to json inside a dictionary: {coff: [replicas values]}.
        Saving also some basic information about the fit.

        Parameters
        ----------
        result : dict
            result dictionary

        """
        posterior_samples = {}
        for c in self.coefficients.name:
            posterior_samples[c] = []

        for sample in result["samples"]:
            self.coefficients.set_free_parameters(sample)
            self.coefficients.set_constraints()

            for c in self.coefficients.name:
                posterior_samples[c].append(self.coefficients[c].value)

        if self.pairwise_fits:
            fit_results_file = (
                self.results_path
                / f"fit_results_{self.coefficients.name[0]}_{self.coefficients.name[1]}.json"
            )
        else:
            fit_results_file = self.results_path / "fit_results.json"

        # self.dump_posterior(posterior_file, values)

        # Writing the fit summary results
        logz = result["logz"]
        max_loglikelihood = result["maximum_likelihood"]["logl"]
        # get the best fit point, including the constrained coefficients
        self.coefficients.set_free_parameters(result["maximum_likelihood"]["point"])
        self.coefficients.set_constraints()
        best_fit_point = dict(zip(self.coefficients.name, self.coefficients.value))

        fit_result = {
            "samples": posterior_samples,
            "logz": logz,
            "max_loglikelihood": max_loglikelihood,
            "best_fit_point": best_fit_point,
            "free_parameters": self.coefficients.free_parameters.index.tolist(),
        }

        self.dump_fit_result(fit_results_file, fit_result)
