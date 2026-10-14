# -*- coding: utf-8 -*-

"""
Module for the generation of theory predictions
"""
import jax.debug
import jax.numpy as jnp
import numpy as np


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
        # default: "no violation" placeholder, used only to keep safe_theory
        # well-defined below when ExternalCoefficients is empty (the for loop
        # is then a no-op and result/dict_point_c would otherwise never be set)
        result = jnp.array(0.0 + 0.0j, dtype=jnp.complex64)

        for ext, val in dataset.ExternalCoefficients.items():
            # cast operands to complex so sqrt/log/power of negative args
            # land on the correct branch instead of returning nan
            dict_point_c = {
                k: jnp.asarray(v, dtype=jnp.complex64) for k, v in dict_point.items()
            }
            result = jnp.asarray(eval(val, {}, dict_point_c), dtype=jnp.complex64)
            bad_here = jnp.any(jnp.imag(result) != 0.0)
            bad_here = bad_here | jnp.any(jnp.isnan(jnp.real(result)))
            out_of_domain = out_of_domain | bad_here
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
        safe_theory = jnp.array(dataset.SMTheory) + 10 ** (1 + jnp.imag(result))
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
