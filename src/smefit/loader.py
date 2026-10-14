# -*- coding: utf-8 -*-

import json
import pathlib
from collections import namedtuple

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
        "QuadraticCorrections",
        "OperatorsDictionary",
        "ExternalCoefficients",
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


def check_missing_oparators(loaded_corrections, coeff_config):
    """Check if all the coefficient in the runcard are also present inside the theory tables."""
    loaded_corrections = set(loaded_corrections)
    missing_operators = [k for k in coeff_config if k not in loaded_corrections]
    if missing_operators != []:
        raise ValueError(
            f"{missing_operators} not in the theory. Comment it out in setup script and restart."
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
    use_theory_covmat: bool
    if True add the theory covariance matrix to the experimental one
    rot_to_fit_basis: dict, None
        matrix rotation to fit basis or None
    poly_mode: bool
        run in polynomial mode
    external_coefficients:
        list of coefficients in the theory card that are going to be replaced with the runcard expression

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
        poly_mode,
        external_coefficients,
    ):
        self._data_folder = self.commondata_path
        self._sys_folder = self.commondata_path / "systypes"
        self._theory_folder = self.theory_path

        self.setname = setname

        self.dataspec = {}
        if poly_mode:
            (
                self.dataspec["SM_predictions"],
                self.dataspec["theory_covmat"],
                self.dataspec["operator_dict"],
            ) = self.load_theory_poly(
                order,
                use_theory_covmat,
                use_multiplicative_prescription,
            )
        else:
            (
                self.dataspec["SM_predictions"],
                self.dataspec["theory_covmat"],
                self.dataspec["lin_corrections"],
                self.dataspec["quad_corrections"],
            ) = self.load_theory(
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
        self.dataspec["external_coefficients"] = external_coefficients
        if len(self.dataspec["central_values"]) != len(self.dataspec["SM_predictions"]):
            raise ValueError(
                f"Number of experimental data points and theory predictions does not match in dataset {self.setname}."
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
            indx_add = np.where(type_sys == "ADD")[0]
            indx_mult = np.where(type_sys == "MULT")[0]
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

    def load_theory_poly(
        self, order, use_theory_covmat, use_multiplicative_prescription
    ):
        """
        Load theory predictions

        Parameters
        ----------
            operators_to_keep: list
                list of operators to keep
            order: "LO", "NLO"
                EFT perturbative order
            use_theory_covmat: bool
                if True add the theory covariance matrix to the experimental one



        Returns
        -------
            sm: numpy.ndarray
                |SM| predictions
            poly_dict: dict
                dictionary which labels are the polynomial expression of the correction and the values the contribution to the theory prediction


        """
        theory_file = self._theory_folder / f"{self.setname}.json"
        check_file(theory_file)
        with open(theory_file, encoding="utf-8") as f:
            raw_th_data = json.load(f)

        # save sm prediction at the chosen perturbative order
        sm = np.array(raw_th_data[order]["SM"])

        poly_dict = {}
        for key, value in raw_th_data[order].items():
            if "SM" not in key:
                poly_dict[key] = np.array(value)
                if use_multiplicative_prescription:
                    poly_dict[key] = np.divide(poly_dict[key], sm)
        best_sm = np.array(raw_th_data["best_sm"])
        th_cov = np.zeros((best_sm.size, best_sm.size))
        if use_theory_covmat:
            th_cov = raw_th_data["theory_cov"]
        return raw_th_data["best_sm"], th_cov, poly_dict

    def load_theory(
        self,
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
            use_theory_covmat: bool
                if True add the theory covariance matrix to the experimental one
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

        """
        theory_file = self._theory_folder / f"{self.setname}.json"
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
        th_cov = np.zeros((best_sm.size, best_sm.size))
        if use_theory_covmat:
            th_cov = raw_th_data["theory_cov"]
        return raw_th_data["best_sm"], th_cov, lin_dict_to_keep, quad_dict_to_keep

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

    @property
    def operator_dict(self):
        """
        Dictionary of operator and their coefficients in poly_mode

        Returns
        -------
            poly_dict : dict
                dictionary with operators formula and their coefficients
        """
        return self.dataspec["operator_dict"]

    @property
    def external_coefficients(self):
        """
        Dictionary of theory card operator that are going to be replaced with the expression in the runcard

        Returns
        -------
            external_coeficcients : dict
                as per above
        """
        return self.dataspec["external_coefficients"]


def construct_corrections_matrix(corrections_list, n_data_tot, sorted_keys=None):
    """
    Construct a unique list of correction name,
    with corresponding values.

    Parameters
    ----------
        corrections_list : list(dict)
            list containing corrections per experiment
        n_data_tot : int
            total number of experimental data points
        sorted_keys: numpy.ndarray
            list of sorted operator corrections

    Returns
    -------
        sorted_keys : np.ndarray
            unique list of operators for which at least one correction is present
        corr_values : np.ndarray
            matrix with correction values (n_data_tot, sorted_keys.size)
    """

    if sorted_keys is None:
        tmp = [
            [
                *c,
            ]
            for _, c in corrections_list
        ]
        sorted_keys = np.unique([item for sublist in tmp for item in sublist])
    corr_values = np.zeros((n_data_tot, sorted_keys.size))
    cnt = 0
    # loop on experiments
    for n_dat, correction_dict in corrections_list:
        # loop on corrections
        for key, values in correction_dict.items():
            if "*" in key:
                op1, op2 = key.split("*")
                if op2 < op1:
                    key = f"{op2}*{op1}"

            idx = np.where(sorted_keys == key)[0][0]
            corr_values[cnt : cnt + n_dat, idx] = values
        cnt += n_dat

    return sorted_keys, corr_values


def load_datasets(
    commondata_path,
    datasets,
    operators_to_keep,
    order,
    use_quad,
    use_theory_covmat,
    use_t0,
    use_multiplicative_prescription,
    theory_path=None,
    rot_to_fit_basis=None,
    has_uv_couplings=False,
    has_external_chi2=False,
    poly_mode=False,
    external_coefficients={},
):
    """
    Loads experimental data, theory and |SMEFT| corrections into a namedtuple

    Parameters
    ----------
        commondata_path : str, pathlib.Path
            path to commondata folder, commondata excluded
        datasets : list
            list of datasets to be loaded
        operators_to_keep: list
            list of operators for which corrections are loaded
        order: "LO", "NLO"
            EFT perturbative order
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
        poly_mode: bool
            True if running in polymode
        external_coefficients: dictionary
            coefficients in the theory card that are going to be replaced by runcard expressions

    """

    exp_data = []
    sm_theory = []
    sys_error_t0 = []
    sys_error = []
    stat_error = []
    lin_corr_list = []
    quad_corr_list = []
    operators_dict_list = []
    n_data_exp = []
    lumi_exp = []
    exp_name = []
    th_cov = []

    Loader.commondata_path = pathlib.Path(commondata_path)
    if theory_path is not None:
        Loader.theory_path = pathlib.Path(theory_path)
    else:
        Loader.theory_path = pathlib.Path(commondata_path)

    for sset in np.unique(datasets):
        dataset = Loader(
            sset,
            operators_to_keep,
            order,
            use_quad,
            use_theory_covmat,
            use_multiplicative_prescription,
            rot_to_fit_basis,
            poly_mode,
            external_coefficients,
        )
        exp_name.append(sset)
        n_data_exp.append(dataset.n_data)
        lumi_exp.append(dataset.lumi)
        exp_data.extend(dataset.central_values)
        sm_theory.extend(dataset.sm_prediction)
        if poly_mode:
            lin_corr_list.append([dataset.n_data, {}])
            quad_corr_list.append([dataset.n_data, {}])
            operators_dict_list.append(
                [
                    list(dataset.operator_dict.keys()),
                    list(dataset.operator_dict.values()),
                ]
            )
        else:
            lin_corr_list.append([dataset.n_data, dataset.lin_corrections])
            quad_corr_list.append([dataset.n_data, dataset.quad_corrections])
            operators_dict_list.append([[], []])
        sys_error_t0.append(dataset.sys_error_t0)
        sys_error.append(dataset.sys_error)
        stat_error.append(dataset.stat_error)
        th_cov.append(dataset.theory_covmat)

    exp_data = np.array(exp_data)
    n_data_tot = exp_data.size

    sorted_keys = None
    # if uv couplings are present allow for op which are not in the
    # theory files
    if has_uv_couplings or has_external_chi2:
        sorted_keys = np.unique((*operators_to_keep,))
    operators_names, lin_corr_values = construct_corrections_matrix(
        lin_corr_list, n_data_tot, sorted_keys
    )
    if poly_mode:
        operators_names = sorted(operators_to_keep.keys())
    if not (poly_mode):
        check_missing_oparators(operators_names, operators_to_keep)

    if use_quad:
        quad_corrections_names = []
        for op1 in operators_names:
            for op2 in operators_names:
                if (
                    f"{op1}*{op2}" not in quad_corrections_names
                    and f"{op2}*{op1}" not in quad_corrections_names
                ):
                    quad_corrections_names.append(f"{op1}*{op2}")
        _, quad_corr_values = construct_corrections_matrix(
            quad_corr_list, n_data_tot, np.array(quad_corrections_names)
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

    # Make one large datatuple containing all data, SM theory, corrections, etc.
    return DataTuple(
        exp_data,
        np.array(sm_theory),
        operators_names,
        lin_corr_values,
        quad_corr_values,
        operators_dict_list,
        external_coefficients,
        np.array(exp_name),
        np.array(n_data_exp),
        np.linalg.inv(fit_covmat),
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
        datasets.QuadraticCorrections[posix_in:posix_out]
        if datasets.QuadraticCorrections is not None
        else None,
        [datasets.OperatorsDictionary[idx]]
        if datasets.OperatorsDictionary is not None
        else None,
        datasets.ExternalCoefficients
        if datasets.ExternalCoefficients is not None
        else None,
        data_name,
        ndata,
        datasets.InvCovMat[posix_in:posix_out].T[posix_in:posix_out],
        datasets.ThCovMat[posix_in:posix_out].T[posix_in:posix_out],
        lumi,
        datasets.Replica[posix_in:posix_out],
    )
