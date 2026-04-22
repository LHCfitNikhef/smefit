# -*- coding: utf-8 -*-
import json
import pathlib
from collections import namedtuple

import jax.numpy as jnp
import numpy as np
import pandas as pd
import scipy.linalg as la
import yaml

from .basis_rotation import rotate_to_fit_basis
from .covmat import construct_covmat, covmat_from_systematics
from .log import logging

_logger = logging.getLogger(__name__)

DataTuple = namedtuple(
    "DataTuple",
    (
        "Commondata",
        "SMTheory",
        "OperatorsNames",
        "LinearCorrections",
        "LinearCorrectionsNoRGE",
        "QuadraticCorrections",
        "ExpNames",
        "NdataExp",
        "InvCovMat",
        "ThCovMat",
        "Luminosity",
        "Replica",
    ),
)


def check_file(path):
    """Check if path exists."""
    if not path.exists():
        raise FileNotFoundError(f"File {path} does not exist.")


def check_missing_operators(loaded_corrections, coeff_config):
    """Check if all the coefficient in the runcard are also present inside the theory tables."""
    loaded_corrections = set(loaded_corrections)
    missing_operators = [k for k in coeff_config if k not in loaded_corrections]
    if missing_operators != []:
        raise ValueError(
            f"{missing_operators} not in the theory. Comment it out in setup script and restart.\n"
            "In case a cutoff was applied "
            f"the operator(s) {missing_operators} did not survive the mask."
        )


def check_condition_number(fit_covmat, critical_cond=15.5):
    condition_number = np.linalg.cond(fit_covmat)
    digits = round(np.log10(condition_number), 2)
    if digits > critical_cond:
        _logger.warning(
            f"Covariance matrix log10 condition number = {digits} "
            + f"larger than critical threshold = {critical_cond}"
        )
    return condition_number


def check_covmat_positivity(fit_covmat):
    eigvals = jnp.linalg.eigvalsh(fit_covmat)
    if min(eigvals) <= 0:
        _logger.warning(f"Covariance matrix not positive")


def check_covmat_invertibility(fit_covmat, inv_covmat, threshold=1e-3):
    if (
        jnp.max(np.abs(inv_covmat @ fit_covmat - np.eye(fit_covmat.shape[0])))
        > threshold
    ):
        _logger.warning(
            "Failed inverting fit covariance matrix. "
            + "Check matrix condition number and using float64 "
        )


class Loader:
    """Class to check, load commondata and corresponding theory predictions.

    Parameters
    ----------
    setname: str
        dataset name to load
    operators_to_keep : list
        list of operators for which corrections are loaded
    order: "LO", "NLO"
        EFT perturbative order
    use_quad : bool
        if True loads also |HO| corrections
    use_theory_covmat: str
        If specified, add the given theory covariance matrix to the experimental one.
    rot_to_fit_basis: dict, None
        matrix rotation to fit basis or None

    """

    commondata_path = pathlib.Path()
    """path to commondata folder, commondata excluded"""
    theory_path = pathlib.Path(commondata_path)
    """path to theory folder, theory excluded. Default it assumes to be the same as commondata_path"""

    def __init__(
        self,
        setname,
        operators_to_keep,
        order,
        use_quad,
        use_theory_covmat,
        use_multiplicative_prescription,
        rot_to_fit_basis,
        cutoff_scale,
    ):
        self._data_folder = self.commondata_path
        self._sys_folder = self.commondata_path / "systypes"
        self._theory_folder = self.theory_path
        self.setname = setname
        self.dataspec = {}
        (
            self.dataspec["SM_predictions"],
            self.dataspec["theory_covmat"],
            self.dataspec["lin_corrections"],
            self.dataspec["quad_corrections"],
            self.dataspec["scales"],
        ) = self.load_theory(
            self.setname,
            operators_to_keep,
            order,
            use_quad,
            use_theory_covmat,
            use_multiplicative_prescription,
            rot_to_fit_basis,
        )

        (
            self.dataspec["central_values"],
            self.dataspec["sys_error"],
            self.dataspec["sys_error_t0"],
            self.dataspec["stat_error"],
            self.dataspec["luminosity"],
        ) = self.load_experimental_data()

        # mask theory and data to ensure only data below the specified cutoff scale is included
        self.mask = np.array(
            [True] * self.n_data
        )  # initial mask retains all datapoints
        if cutoff_scale is not None:
            self.apply_cutoff_mask(cutoff_scale)

        if len(self.dataspec["central_values"]) != len(self.dataspec["SM_predictions"]):
            raise ValueError(
                f"Number of experimental data points and theory predictions does not match in dataset {self.setname}."
            )

    def apply_cutoff_mask(self, cutoff_scale):
        """
        Updates previously loaded theory and datasets by filtering out
        points with scales above the cutoff scale

        Parameters
        ----------
        cutoff_scale: flaot
            Value of the cutoff scale as specified in the runcard
        """
        self.mask = self.dataspec["scales"] < cutoff_scale

        # if all datapoints lie above the cutoff, return
        if np.all(~self.mask):
            return

        # Apply mask to all relevant theory entries in dataspec
        self.dataspec.update(
            {
                "SM_predictions": self.dataspec["SM_predictions"][self.mask],
                "theory_covmat": self.dataspec["theory_covmat"][self.mask, :][
                    :, self.mask
                ],
                "lin_corrections": {
                    k: v[self.mask] for k, v in self.dataspec["lin_corrections"].items()
                },
                "quad_corrections": {
                    k: v[self.mask]
                    for k, v in self.dataspec["quad_corrections"].items()
                },
                "scales": self.dataspec["scales"][self.mask],
            }
        )

        # Single data points satisfy the mask already at this point
        if self.n_data == 1:
            stat_error = self.dataspec["stat_error"]
        else:
            stat_error = self.dataspec["stat_error"][self.mask]

        self.dataspec.update(
            {
                "central_values": self.dataspec["central_values"][self.mask],
                "sys_error": self.dataspec["sys_error"].loc[self.mask],
                "sys_error_t0": self.dataspec["sys_error_t0"].loc[self.mask],
                "stat_error": stat_error,
            }
        )

    def load_experimental_data(self):
        """
        Load experimental data with corresponding uncertainties

        Returns
        -------
            cental_values: numpy.ndarray
                experimental central values
            covmat : numpy.ndarray
                experimental covariance matrix
        """
        data_file = self._data_folder / f"{self.setname}.yaml"
        check_file(data_file)

        _logger.info(f"Loading dataset : {self.setname}")
        with open(data_file, encoding="utf-8") as f:
            data_dict = yaml.safe_load(f)

        central_values = np.array(data_dict["data_central"])
        stat_error = np.array(data_dict["statistical_error"])
        luminosity = data_dict.get("luminosity", None)

        num_sys = data_dict["num_sys"]
        num_data = data_dict["num_data"]

        # Load systematics from commondata file.
        # Read values of sys first

        sys_add = np.array(data_dict["systematics"])

        # Read systype file
        if num_sys != 0:
            type_sys = np.array(data_dict["sys_type"])
            name_sys = data_dict["sys_names"]

            # express systematics as percentage values of the central values
            sys_mult = sys_add / central_values * 1e2

            # Identify add and mult systematics
            # and replace the mult ones with corresponding value computed
            # from data central value. Required for implementation of t0 prescription
            indx_add = np.flatnonzero(type_sys == "ADD")
            indx_mult = np.flatnonzero(type_sys == "MULT")
            sys_t0 = np.zeros((num_sys, num_data))
            sys_t0[indx_add] = sys_add[indx_add].reshape(sys_t0[indx_add].shape)

            sys_t0[indx_mult] = (
                sys_mult[indx_mult].reshape(sys_t0[indx_mult].shape)
                * self.dataspec["SM_predictions"]
                * 1e-2
            )

            # store also the sys without the t0 prescription
            sys = sys_add.reshape((num_sys, num_data))

            # limit case with 1 sys
            if num_sys == 1:
                name_sys = [name_sys]
        # limit case no sys
        else:
            name_sys = ["UNCORR"]
            sys = np.zeros((num_sys + 1, num_data))
            sys_t0 = sys

        # Build dataframe with shape (N_data * N_sys) and systematic name as the column headers
        df = pd.DataFrame(data=sys.T, columns=name_sys)
        df_t0 = pd.DataFrame(data=sys_t0.T, columns=name_sys)
        # limit case 1 data
        if num_data == 1:
            central_values = np.asarray([central_values])

        # here return both exp sys and t0 modified sys
        return central_values, df, df_t0, stat_error, luminosity

    @staticmethod
    def load_theory(
        setname,
        operators_to_keep,
        order,
        use_quad,
        use_theory_covmat,
        use_multiplicative_prescription,
        rotation_matrix=None,
    ):
        """
        Load theory predictions

        Parameters
        ----------
            operators_to_keep: list
                list of operators to keep
            order: "LO", "NLO"
                EFT perturbative order
            use_quad: bool
                if True returns also |HO| corrections
            use_theory_covmat: str
                If specified, add the given theory covariance matrix to the experimental one
            rotation_matrix: numpy.ndarray
                rotation matrix from tables basis to fitting basis

        Returns
        -------
            sm: numpy.ndarray
                |SM| predictions
            lin_dict: dict
                dictionary with |NHO| corrections
            quad_dict: dict
                dictionary with |HO| corrections, empty if not use_quad
            scales: list
                list of energy scales for the theory predictions
        """
        theory_file = Loader.theory_path / f"{setname}.json"
        check_file(theory_file)
        # load theory predictions
        with open(theory_file, encoding="utf-8") as f:
            raw_th_data = json.load(f)

        quad_dict = {}
        lin_dict = {}

        # save sm prediction at the chosen perturbative order

        sm = np.array(raw_th_data[order]["SM"])

        # split corrections into a linear and quadratic dict
        for key, value in raw_th_data[order].items():
            # quadratic terms
            if "*" in key and use_quad:
                quad_dict[key] = np.array(value)
                if use_multiplicative_prescription:
                    quad_dict[key] = np.divide(quad_dict[key], sm)

            # linear terms
            elif "SM" not in key and "*" not in key:
                lin_dict[key] = np.array(value)
                if use_multiplicative_prescription:
                    lin_dict[key] = np.divide(lin_dict[key], sm)

        # select corrections to keep
        def is_to_keep(op1, op2=None):
            if op2 is None:
                return op1 in operators_to_keep
            return op1 in operators_to_keep and op2 in operators_to_keep

        # rotate corrections to fitting basis
        if rotation_matrix is not None:
            lin_dict_fit_basis, quad_dict_fit_basis = rotate_to_fit_basis(
                lin_dict, quad_dict, rotation_matrix
            )

            lin_dict_to_keep = {
                k: val for k, val in lin_dict_fit_basis.items() if is_to_keep(k)
            }
            quad_dict_to_keep = {
                k: val
                for k, val in quad_dict_fit_basis.items()
                if is_to_keep(k.split("*")[0], k.split("*")[1])
            }
        else:
            lin_dict_to_keep = {k: val for k, val in lin_dict.items() if is_to_keep(k)}
            quad_dict_to_keep = {
                k: val
                for k, val in quad_dict.items()
                if is_to_keep(k.split("*")[0], k.split("*")[1])
            }
        best_sm = np.array(raw_th_data["best_sm"])
        if use_theory_covmat:
            theory_covmat_key = (
                use_theory_covmat if use_theory_covmat in raw_th_data else "theory_cov"
            )
            th_cov = np.array(raw_th_data[theory_covmat_key])
        else:
            th_cov = np.zeros((best_sm.size, best_sm.size))

        # check if scales are present in the theory file
        scales = np.array(raw_th_data.get("scales", [None] * len(best_sm)))

        return (
            best_sm,
            th_cov,
            lin_dict_to_keep,
            quad_dict_to_keep,
            scales,
        )

    @property
    def n_data(self):
        """
        Number of data

        Returns
        -------
            n_data: int
                number of experimental data
        """
        return self.dataspec["central_values"].size

    @property
    def lumi(self):
        """
        Integrated luminosity of the dataset in fb^-1

        Returns
        -------
            lumi: float
                Integrated luminosity of the dataset in fb^-1

        """
        return self.dataspec["luminosity"]

    @property
    def central_values(self):
        """
        Central values

        Returns
        -------
            central_values: numpy.ndarray
                experimental central values
        """
        return self.dataspec["central_values"]

    @property
    def covmat(self):
        """
        Experimental covariance matrix

        Returns
        -------
            covmat: numpy.ndarray
                experimental covariance matrix of a single dataset
        """
        return construct_covmat(self.dataspec["stat_error"], self.dataspec["sys_error"])

    @property
    def theory_covmat(self):
        """
        Theory covariance matrix

        Returns
        -------
            theory covmat: numpy.ndarray
                theory covariance matrix of a single dataset
        """
        return self.dataspec["theory_covmat"]

    @property
    def sys_error(self):
        """
        Systematic errors

        Returns
        -------
            sys_error: pd.DataFrame
                systematic errors of the dataset
        """
        return self.dataspec["sys_error"]

    @property
    def sys_error_t0(self):
        """
        Systematic errors modified according to t0 prescription

        Returns
        -------
            sys_error_t0: pd.DataFrame
                t0 systematic errors of the dataset
        """
        return self.dataspec["sys_error_t0"]

    @property
    def stat_error(self):
        """
        Statistical errors

        Returns
        -------
            stat_error: np.array
                statistical errors of the dataset
        """
        return self.dataspec["stat_error"]

    @property
    def sm_prediction(self):
        """
        |SM| prediction for the dataset

        Returns
        -------
            SM_predictions : numpy.ndarray
                best |SM| prediction
        """
        return self.dataspec["SM_predictions"]

    @property
    def lin_corrections(self):
        """
        |NHO| corrections

        Returns
        -------
            lin_corrections : dict
                dictionary with operator names and |NHO| correctsions
        """
        return self.dataspec["lin_corrections"]

    @property
    def quad_corrections(self):
        """
        |HO| corrections

        Returns
        -------
            quad_corrections : dict
                dictionary with operator names and |HO| correctsions
        """
        return self.dataspec["quad_corrections"]


def construct_corrections_matrix_linear(
    corrections_list, n_data_tot, sorted_keys, rgemat=None
):
    """
    Constructs the linear EFT corrections.

    Parameters
    ----------
        corrections_list : list(dict)
            list containing corrections per experiment
        n_data_tot : int
            total number of experimental data points
        sorted_keys: numpy.ndarray
            list of sorted operator corrections
        rgemat: numpy.ndarray, optional
            solution matrix of the RGE, shape (k, l, m) with k the number of datapoints,
            l the number of generated coefficients and m the number of
            original |EFT| coefficients specified in the runcard.

    Returns
    -------
        corr_values : np.ndarray
            matrix with correction values (n_data_tot, sorted_keys.size)
    """

    corr_values = np.zeros((n_data_tot, sorted_keys.size))
    cnt = 0
    # loop on experiments
    for n_dat, correction_dict in corrections_list:
        # loop on corrections
        for key, values in correction_dict.items():
            idx = np.where(sorted_keys == key)[0][0]
            corr_values[cnt : cnt + n_dat, idx] = values
        cnt += n_dat

    if rgemat is not None:
        corr_values = jnp.einsum("ij, ijk -> ik", corr_values, rgemat)

    return corr_values


def construct_corrections_matrix_quadratic(
    corrections_list, n_data_tot, sorted_keys, rgemat=None
):
    """
    Constructs quadratic EFT corrections.

    Parameters
    ----------
        corrections_list : list(dict)
            list containing per experiment the number of datapoints and corresponding corrections
        n_data_tot : int
            total number of experimental data points
        sorted_keys: numpy.ndarray
            list of sorted operator corrections, shape=(n rg generated coeff,)
            or shape=(n original coeff,) in the absence of rgemat
        rgemat: numpy.ndarray, optional
            solution matrix of the RGE, shape=(k, l, m) with k the number of datapoints,
            l the number of generated coefficients under the RG and m the number of
            original |EFT| coefficients specified in the runcard.

    Returns
    -------
        corr_values : np.ndarray
            matrix with correction values (n_data_tot, sorted_keys.size, sorted_keys.size)
    """

    corr_values = np.zeros((n_data_tot, sorted_keys.size, sorted_keys.size))
    cnt = 0

    # loop on experiments
    for n_dat, correction_dict in corrections_list:
        # loop on corrections
        for key, values in correction_dict.items():
            op1, op2 = key.split("*")
            idx1 = np.where(sorted_keys == op1)[0][0]
            idx2 = np.where(sorted_keys == op2)[0][0]

            # we want to store the values in the upper triangular part of the matrix
            if idx1 > idx2:
                idx1, idx2 = idx2, idx1

            corr_values[cnt : cnt + n_dat, idx1, idx2] = values
        cnt += n_dat

    if rgemat is not None:
        corr_values = jnp.einsum("ijk, ijl, ikr -> ilr", corr_values, rgemat, rgemat)

    return corr_values


def load_datasets(
    commondata_path,
    datasets,
    operators_to_keep,
    use_quad,
    use_theory_covmat,
    use_t0,
    use_multiplicative_prescription,
    default_order="LO",
    theory_path=None,
    rot_to_fit_basis=None,
    has_uv_couplings=False,
    has_external_chi2=False,
    rgemat=None,
    cutoff_scale=None,
):
    """
    Loads experimental data, theory and |SMEFT| corrections into a namedtuple

    Parameters
    ----------
        commondata_path : str, pathlib.Path
            path to commondata folder, commondata excluded
        datasets : list
            List of datasets to be loaded
        operators_to_keep: list
            list of operators for which corrections are loaded
        default_order: str
            Default perturbative order of the theory predictions
        use_quad: bool
            if True loads also |HO| corrections
        use_theory_covmat: bool
            if True add the theory covariance matrix to the experimental one
        theory_path : str, pathlib.Path, optional
            path to theory folder, theory excluded.
            Default it assumes to be the same as commondata_path
        rot_to_fit_basis: dict, optional
            matrix rotation to fit basis or None
        has_uv_couplings: bool, optional
            True for UV fits
        has_external_chi2: bool, optional
            True in the presence of external chi2 modules
        rgemat: numpy.ndarray, optional
            solution matrix of the RGE, shape=(k, l, m) with k the number of datapoints,
            l the number of generated coefficients under the RG and m the number of
            original |EFT| coefficients specified in the runcard.
        cutoff_scale: float, optional
            kinematic cutoff scale
    """

    exp_data = []
    sm_theory = []
    sys_error_t0 = []
    sys_error = []
    stat_error = []
    lin_corr_list = []
    quad_corr_list = []
    n_data_exp = []
    lumi_exp = []
    exp_name = []
    th_cov = []

    Loader.commondata_path = pathlib.Path(commondata_path)
    Loader.theory_path = pathlib.Path(theory_path or commondata_path)

    _logger.info(f"Applying cutoff scale: {cutoff_scale} GeV.")
    for sset in datasets:
        dataset_name = sset.get("name")

        # determine which theory covariance matrix to use
        theory_cov = (
            f"theory_cov_{sset.get('theory_cov', 'current')}"
            if use_theory_covmat
            else None
        )

        dataset = Loader(
            dataset_name,
            operators_to_keep,
            sset.get("order", default_order),
            use_quad,
            theory_cov,
            use_multiplicative_prescription,
            rot_to_fit_basis,
            cutoff_scale,
        )

        # skip dataset if all datapoints are above the cutoff scale
        if np.all(~dataset.mask):
            continue

        exp_name.append(dataset_name)
        n_data_exp.append(dataset.n_data)
        if np.isscalar(dataset.lumi):
            lumi_exp.extend([dataset.lumi] * dataset.n_data)
        elif dataset.lumi is None:
            lumi_exp.extend([np.nan] * dataset.n_data)
        else:
            if len(dataset.lumi) != dataset.n_data:
                raise ValueError(
                    f"Length of luminosity list does not match number "
                    f"of data points for dataset {dataset_name}."
                )
            lumi_exp.extend(dataset.lumi)
        exp_data.extend(dataset.central_values)
        sm_theory.extend(dataset.sm_prediction)
        lin_corr_list.append([dataset.n_data, dataset.lin_corrections])
        quad_corr_list.append([dataset.n_data, dataset.quad_corrections])
        sys_error_t0.append(dataset.sys_error_t0)
        sys_error.append(dataset.sys_error)
        stat_error.append(dataset.stat_error)
        th_cov.append(dataset.theory_covmat)

    exp_data = np.array(exp_data)
    n_data_tot = exp_data.size

    # if uv couplings are present allow for op which are not in the
    # theory files (same for external chi2 and rge)
    if has_uv_couplings or has_external_chi2 or rgemat is not None:
        sorted_keys = np.unique((*operators_to_keep,))
    else:
        # Construct unique list of operator names entering lin_corr_list
        operator_names = []
        for item in lin_corr_list:
            operator_names.extend(list(item[1].keys()))
        sorted_keys = np.unique(operator_names)

        # operators to keep contains the operators that are present in the runcard
        # sorted_keys contains the operators that are present in the theory files
        # we need to check that all operators in the runcard are also present in the theory files
        check_missing_operators(sorted_keys, operators_to_keep)

    lin_corr_values = construct_corrections_matrix_linear(
        lin_corr_list, n_data_tot, sorted_keys, rgemat
    )

    lin_corr_values_norge = construct_corrections_matrix_linear(
        lin_corr_list, n_data_tot, sorted_keys
    )

    if use_quad:
        quad_corr_values = construct_corrections_matrix_quadratic(
            quad_corr_list, n_data_tot, sorted_keys, rgemat
        )
    else:
        quad_corr_values = None

    # Construct unique large cov matrix accounting for correlations between different datasets
    # The theory covariance matrix, when used, will be different from zero.
    # At the moment it does not account for correlation between different datasets
    theory_covariance = la.block_diag(*th_cov)
    exp_covmat = covmat_from_systematics(stat_error, sys_error) + theory_covariance

    # replicas always generated using the experimental covmat, no t0
    replica = np.random.multivariate_normal(exp_data, exp_covmat)
    if use_t0:
        fit_covmat = (
            covmat_from_systematics(stat_error, sys_error_t0) + theory_covariance
        )
    else:
        fit_covmat = exp_covmat

    # Check fit_covmat condition number
    check_condition_number(fit_covmat)
    check_covmat_positivity(fit_covmat)
    inv_cov_mat = np.linalg.inv(fit_covmat)
    check_covmat_invertibility(fit_covmat, inv_cov_mat)
    # Make one large datatuple containing all data, SM theory, corrections, etc.
    return DataTuple(
        exp_data,
        np.array(sm_theory),
        sorted_keys,
        lin_corr_values,
        lin_corr_values_norge,
        quad_corr_values,
        np.array(exp_name),
        np.array(n_data_exp),
        inv_cov_mat,
        theory_covariance,
        np.array(lumi_exp),
        replica,
    )


def get_dataset(datasets, data_name):
    idx = np.where(datasets.ExpNames == data_name)[0][0]
    ndata = datasets.NdataExp[idx]
    lumi = datasets.Luminosity[idx]
    posix_in = datasets.NdataExp[:idx].sum()
    posix_out = posix_in + ndata

    return DataTuple(
        datasets.Commondata[posix_in:posix_out],
        datasets.SMTheory[posix_in:posix_out],
        datasets.OperatorsNames,
        datasets.LinearCorrections[posix_in:posix_out],
        (
            datasets.QuadraticCorrections[posix_in:posix_out]
            if datasets.QuadraticCorrections is not None
            else None
        ),
        data_name,
        ndata,
        datasets.InvCovMat[posix_in:posix_out].T[posix_in:posix_out],
        datasets.ThCovMat[posix_in:posix_out].T[posix_in:posix_out],
        lumi,
        datasets.Replica[posix_in:posix_out],
    )
