# -*- coding: utf-8 -*-

"""
Module for the generation of theory predictions
"""
import jax.debug
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np
import scipy.special as sps


def _on_real(f):
    """Special functions don't accept complex input: apply f to the real part."""
    return lambda *args: f(
        *[np.real(a) if isinstance(a, np.ndarray) else jnp.real(a) for a in args]
    )


def _functions(xp, special):
    """Names usable in the runcard expressions of the external coefficients."""
    funcs = {
        "sqrt": xp.sqrt,
        "log": xp.log,
        "exp": xp.exp,
        "cos": xp.cos,
        "sin": xp.sin,
        "pi": np.pi,
    }
    for name in ("erfinv", "ndtri", "gammaincinv"):
        if hasattr(special, name):  # gammaincinv is missing in older jax versions
            funcs[name] = _on_real(getattr(special, name))
    return funcs


EXT_FUNCTIONS_JAX = _functions(jnp, jsp)  # used inside the fit (make_predictions)
EXT_FUNCTIONS_NP = _functions(np, sps)  # used for post-processing


def flatten(quad_mat, axis=0):
    """
    Delete lower triangular part of a quadratic matrix
    and flatten it into an array

    Parameters
    ----------
        quad_mat: numpy.ndarray
            tensor to flatten
        axis: int
            axis along which the triangular part is selected
    """
    size = quad_mat.shape[axis]
    return quad_mat[np.triu_indices(size)]


def evaluate_external_coefficients(external_coefficients, coefficients_dict):
    """Evaluate the runcard expressions of the external coefficients (poly_mode)
    on a set of samples, outside of jax.

    Expressions are evaluated in order, so later expressions can use earlier
    external coefficients, mirroring what ``make_predictions`` does.

    Parameters
    ----------
    external_coefficients: dict
        {name: expression} as given in the runcard
    coefficients_dict: dict
        {coefficient name: array of values (one per sample)}

    Returns
    -------
    dict
        {name: np.ndarray} with the real values of the external coefficients.
        Samples where the expression is complex (out of domain) are set to NaN.
    """
    point = {
        k: np.asarray(v, dtype=np.complex128) for k, v in coefficients_dict.items()
    }
    n_samples = np.broadcast(*point.values()).shape if point else ()
    ext_values = {}
    for name, expr in external_coefficients.items():
        res = np.broadcast_to(
            np.asarray(
                eval(expr, dict(EXT_FUNCTIONS_NP), dict(point)), dtype=np.complex128
            ),
            n_samples,
        )
        real = np.where(np.imag(res) != 0.0, np.nan, np.real(res))
        ext_values[name] = real
        point[name] = np.asarray(real, dtype=np.complex128)
    return ext_values


def string_to_function(expressions, param_dict):
    """Substitute to the theory card expression the numerical values, keeping also into account external
    defined parameters

    Parameters
    ----------
    expressions: the array of theory card dictionary keys,
    param_dict: dictionary with the values of the point in the parameter space tried and the external parameters numerical value
    """
    return [eval(e, param_dict) for e in expressions]


def make_predictions(
    dataset, coefficients_values, use_quad, use_multiplicative_prescription, poly_mode
):

    """
    Generate the corrected theory predictions for dataset
    given a set of |SMEFT| coefficients.

    Parameters
    ----------
        dataset : DataTuple
            dataset tuple
        coefficients_values : numpy.ndarray
            |EFT| coefficients values
        use_quad: bool
            if True include also |HO| corrections
    Returns
    -------
        corrected_theory : numpy.ndarray
            SM + EFT theory predictions
    """
    if poly_mode:
        summed_corrections = []
        # create dictionary of operator and the point tried
        dict_point = dict(zip(dataset.OperatorsNames, coefficients_values))

        # external coefficients could eval to imaginary, meaning they're not in the fitting domain
        out_of_domain = jnp.array(False)
        # size of the violation: largest |imaginary part| over all external coefficients.
        # |.| is needed because the sign of the imaginary part depends on the branch
        # (e.g. (-x - 0j)**0.5 = -i sqrt(x)), and the penalty must grow with the violation.
        violation = jnp.array(0.0, dtype=jnp.float32)

        for ext, val in dataset.ExternalCoefficients.items():
            # cast operands to complex so sqrt/log/power of negative args
            # land on the correct branch instead of returning nan
            dict_point_c = {
                k: jnp.asarray(v, dtype=jnp.complex64) for k, v in dict_point.items()
            }
            result = jnp.asarray(
                eval(val, dict(EXT_FUNCTIONS_JAX), dict_point_c), dtype=jnp.complex64
            )
            bad_here = jnp.any(jnp.imag(result) != 0.0)
            bad_here = bad_here | jnp.any(jnp.isnan(jnp.real(result)))
            out_of_domain = out_of_domain | bad_here
            violation = jnp.maximum(violation, jnp.max(jnp.abs(jnp.imag(result))))
            dict_point[ext] = jnp.real(result)

        # OperatorsDictionary[][1] are the labels, [][2] are the numerical coefficients
        for op_dict in dataset.OperatorsDictionary:
            op_dict_num_values = string_to_function(op_dict[0], dict_point)
            summed_corrections.extend(
                jnp.dot(jnp.array(op_dict_num_values), jnp.array(op_dict[1]))
            )

        # construct the theory prediction
        if use_multiplicative_prescription:
            corrected_theory = jnp.array(dataset.SMTheory) * (
                1.0 + jnp.array(summed_corrections)
            )
        else:
            corrected_theory = jnp.array(dataset.SMTheory) + jnp.array(
                summed_corrections
            )

        # if the point is out of domain, replace the (possibly NaN-poisoned)
        # theory with a clean, huge-but-finite offset built ONLY from
        # dataset.SMTheory -- never from corrected_theory/summed_corrections,
        # since NaN + anything is still NaN.
        safe_theory = jnp.array(dataset.SMTheory) + 10 ** (1 + violation)
        corrected_theory = jnp.where(out_of_domain, safe_theory, corrected_theory)

        return corrected_theory

    else:

        coefficients_values = jnp.array(coefficients_values)

        # Compute total linear correction
        # note @ is slower when running with mpiexec
        summed_corrections = jnp.einsum(
            "ij,j->i", dataset.LinearCorrections, coefficients_values
        )

        # Compute total quadratic correction
        if use_quad:
            coeff_outer_coeff = jnp.outer(coefficients_values, coefficients_values)
            # note @ is slower when running with mpiexec
            summed_quad_corrections = jnp.einsum(
                "ij,j->i", dataset.QuadraticCorrections, flatten(coeff_outer_coeff)
            )
            summed_corrections += summed_quad_corrections

        # Sum of SM theory + SMEFT corrections
        if use_multiplicative_prescription:
            corrected_theory = dataset.SMTheory * (1.0 + summed_corrections)
        else:
            corrected_theory = dataset.SMTheory + summed_corrections

        return corrected_theory
