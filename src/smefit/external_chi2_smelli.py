"""External χ² interface for smelli."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np

from smefit.rge.rge import RGE


class SmelliChi2:
    """Compute a smelli likelihood contribution in a given SMEFiT fit."""

    def __init__(
        self,
        coefficients,
        rge_dict=None,
        include_likelihoods: Iterable[str] | None = None,
        initial_scale: float | None = None,
    ):
        self.names = list(coefficients.name)

        # Build smelli objects lazily to avoid hard import side effects.
        from smelli import GlobalLikelihood
        from wilson import Wilson

        self._Wilson = Wilson
        self.include_likelihoods = (
            ["likelihood_ewpt.yaml"]
            if include_likelihoods is None
            else list(include_likelihoods)
        )

        # Initial scale where SMEFT coefficients are defined.
        if rge_dict is not None and "init_scale" in rge_dict:
            self.scale = (
                float(initial_scale)
                if initial_scale is not None
                else float(rge_dict.get("init_scale", 1e3))
            )

            # Translate SMEFiT basis to Warsaw basis.
            self.translation_basis = RGE(
                self.names,
                self.scale,
                rge_dict.get("smeft_accuracy", "integrate"),
                rge_dict.get("adm_QCD", False),
                rge_dict.get("yukawa", "top"),
            ).RGEbasis
        else:
            self.scale = float(initial_scale) if initial_scale is not None else 1e3
            self.translation_basis = {op: {op: 1.0} for op in self.names}

        self.gl = GlobalLikelihood(
            eft="SMEFT", basis="Warsaw", include_likelihoods=self.include_likelihoods
        )

    def _build_wilson_dict(self, coefficient_values):
        coeff_values = np.asarray(coefficient_values, dtype=float)
        if coeff_values.shape[0] != len(self.names):
            raise ValueError(
                "Coefficient vector length does not match fitted coefficients "
                f"({coeff_values.shape[0]} != {len(self.names)})."
            )

        wc_dict = {}
        for op, c in zip(self.names, coeff_values):
            for key, val in self.translation_basis[op].items():
                wc_dict[key] = wc_dict.get(key, 0.0) + float(val) * c
        return {k: v for k, v in wc_dict.items() if v != 0.0}

    def compute_chi2(self, coefficient_values):
        wc_dict = self._build_wilson_dict(coefficient_values)
        w = self._Wilson(wc_dict, self.scale, eft="SMEFT", basis="Warsaw")
        pt = self.gl.parameter_point(w)
        return -2.0 * float(pt.log_likelihood_global())


# Backward-compatible class name for older runcards.
smelli_chi2 = SmelliChi2
