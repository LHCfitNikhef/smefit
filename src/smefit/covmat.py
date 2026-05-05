# -*- coding: utf-8 -*-
"""
Module containing computation of covariance matrix.
Based on
https://github.com/NNPDF/nnpdf/tree/master/validphys2/src/validphys/covmats_utils.py
https://github.com/NNPDF/nnpdf/tree/master/validphys2/src/validphys/covmats.py
"""

import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import scipy.linalg as la
from scipy.linalg import block_diag

from smefit.log import logging

INTRA_DATASET_SYS_NAME = ("UNCORR", "CORR", "THEORYUNCORR", "THEORYCORR")
_logger = logging.getLogger(__name__)


def construct_covmat(stat_errors: np.array, sys_errors: pd.DataFrame):
    """
    Basic function to construct a covariance matrix (covmat), given the
    statistical error and a dataframe of systematics.
    Errors with name UNCORR or THEORYUNCORR are added in quadrature with
    the statistical error to the diagonal of the covmat.
    Other systematics are treated as correlated; their covmat contribution is
    found by multiplying them by their transpose.

    Parameters
    ----------
        stat_errors: numpy.ndarray
            a 1-D array of statistical uncertainties
        sys_errors: pandas.DataFrame
            a dataframe with shape (N_data * N_sys) and systematic name as the
            column headers. The uncertainties should be in the same units as the
            data.

    Returns
    -------
        cov_mat: numpy.ndarray
            Covariance matrix

    Notes
    -----
    This function doesn't contain any logic to ignore certain contributions to
    the covmat, if you wanted to not include a particular systematic/set of
    systematics i.e all uncertainties with MULT errors, then filter those out
    of ``sys_errors`` before passing that to this function.

    """
    diagonal = stat_errors**2

    is_uncorr = sys_errors.columns.isin(("UNCORR", "THEORYUNCORR"))
    diagonal += (sys_errors.loc[:, is_uncorr].to_numpy() ** 2).sum(axis=1)

    corr_sys_mat = sys_errors.loc[:, ~is_uncorr].to_numpy()
    return np.diag(diagonal) + corr_sys_mat @ corr_sys_mat.T


def covmat_from_systematics(stat_errors: list, sys_errors: list):
    """Given two lists containing the statistic and systematic errors,
    construct the full covariance matrix.

    This is similar to :py:meth:`construct_covmat`
    except that special corr systematics are concatenated across all datasets
    before being multiplied by their transpose to give off block-diagonal
    contributions. The other systematics contribute to the block diagonal in the
    same way as :py:meth:`construct_covmat`.

    Parameters
    ----------
    stat_errors : list[(stat_errors: np.array)]
        list of stat_errors for each dataset.

    sys_errors : list[(sys_errors: pd.DataFrame)]
        list of sys_errors for each dataset.

    Returns
    -------
    cov_mat : np.array
        Numpy array which is N_dat x N_dat (where N_dat is the number of data points)
        containing uncertainty and correlation information.
    """
    special_corrs = []
    block_diags = []

    for dataset_stat_errors, dataset_sys_errors in zip(stat_errors, sys_errors):
        # separate out the special uncertainties which can be correlated across
        # datasets
        is_intra_dataset_error = dataset_sys_errors.columns.isin(INTRA_DATASET_SYS_NAME)
        block_diags.append(
            construct_covmat(
                dataset_stat_errors, dataset_sys_errors.loc[:, is_intra_dataset_error]
            )
        )
        special_corrs.append(dataset_sys_errors.loc[:, ~is_intra_dataset_error])
    # concat systematics across datasets
    special_sys = pd.concat(special_corrs, axis=0, sort=False)
    # non-overlapping systematics are set to NaN by concat, fill with 0 instead.
    special_sys.fillna(0, inplace=True)

    diag = la.block_diag(*block_diags)
    covmat = diag + special_sys.to_numpy() @ special_sys.to_numpy().T
    return covmat


#################################################################
###Algorithm to invert covmat avoiding large condition number####
#################################################################


def compute_blocks_inverse(covmat, tol=1e-25):
    """
    Decomposes the covmatrix into diagonal blocks and inverts them.
    Parameters
    ----------
    covmat: covariance matrix
    tol: what is considered numerically zero in the covmat
    """
    n = covmat.shape[0]
    mask = jnp.abs(covmat) > tol
    col_indices = jnp.arange(n)
    # For each row, find the index of the furthest non-zero column.
    max_col_per_row = jnp.max(jnp.where(mask, col_indices, 0), axis=1)
    max_row_per_col = jnp.max(jnp.where(mask.T, col_indices, 0), axis=1)
    combined_reach = jnp.maximum(max_col_per_row, max_row_per_col)
    # covmat should be symmetric, however checking both row and columns
    # increase safety in case same asymmetry is present

    running_max = lax.cummax(combined_reach, axis=0)
    # normally running_max==combined_reach already, however interdataset
    # correlation can spoil this creating larger blocks connecting elements
    # that are far away in the covmat.
    # In this case I raise a warning that the dataset should be reordered to
    # have full optimization in the decomposition and inversion algo
    if jnp.any(running_max != combined_reach):
        _logger.warning(
            "In the runcard some intercorrelated datasets are not written next to each other."
            + "To increase efficiency and stability in covmat inversion\n please consider reordering"
            + " your runcard datasets entries."
        )

    # A block boundary exists where the furthest reach so far is <= the current index.
    boundaries = jnp.where(running_max <= col_indices)[0]
    sizes = np.diff(boundaries, prepend=-1).tolist()
    block_cond = []
    block_inverses = []
    start = 0
    for s in sizes:
        end = start + s
        block = covmat[start:end, start:end]
        block_cond.append(jnp.linalg.cond(block))
        block_inverses.append(jnp.linalg.inv(block))
        start = end
    return block_diag(*block_inverses), max(block_cond)
