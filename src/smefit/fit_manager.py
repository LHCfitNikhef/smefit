# -*- coding: utf-8 -*-
import json

import numpy as np
import pandas as pd
import yaml
from rich.progress import track

from .coefficients import CoefficientManager
from .compute_theory import make_predictions
from .loader import load_datasets


class FitManager:
    """
    Class to collect all the fit information,
    load the results, compute best theory predictions.

    Attributes
    ----------
        path: pathlib.Path
            path to fit location
        name: srt
            fit name
        label: str, optional
            fit label if any otherwise guess it from the name
        config: dict
            configuration dictionary
        has_posterior: bool
            True if the fi contains the full posterrio distribution,
            False if only cl bounds are stored (external fits for benchmark)
        results: pandas.DataFrame
            fit results, they need to be loaded by `load_results`

    Parameters
    ----------
        path: pathlib.Path
            path to fit location
        name: srt
            fit name
        label: str, optional
            fit label if any otherwise guess it from the name
    """

    def __init__(self, path, name, label=None):
        self.path = path
        self.name = name
        self.label = (
            r"${\rm %s}$" % name.replace("_", r"\ ") if label is None else label
        )

        # load the configuration file
        self.config = self.load_configuration()
        self.has_posterior = self.config.get("has_posterior", True)
        self.results = None
        self.datasets = None

    def __repr__(self):
        return self.name

    def __eq__(self, comapre_name):
        return self.name == comapre_name

    def load_results(self):
        """
        Load posterior distribution of a fit.
        If the fit is produced by and external source it loads
        the results. Results are stored in a class attribute
        """
        file = "results"
        if self.has_posterior:
            file = "posterior"
        with open(f"{self.path}/{self.name}/{file}.json", encoding="utf-8") as f:
            results = json.load(f)

        # if the posterior is from single parameter fits
        # then each distribution might have a different number of samples
        is_single_param = results.get("single_parameter_fits", False)
        if is_single_param:

            del results["single_parameter_fits"]

            num_samples = []
            for key in results.keys():
                num_samples.append(len(results[key]))
            num_samples_min = min(num_samples)

            for key in results.keys():
                results[key] = np.random.choice(
                    results[key], num_samples_min, replace=False
                )

        # TODO: support pariwise posteriors

        # extra information not entering the theory predictions:
        # best fit point, chi2 of the samples, external coefficients (poly_mode)
        self.best_fit = results.pop("best_fit", None)
        extra_keys = [
            key
            for key in results
            if self.has_posterior and key not in self.config["coefficients"]
        ]
        self.extra_results = pd.DataFrame({key: results.pop(key) for key in extra_keys})

        # Be sure columns are sorted, otherwise can't compute theory...
        self.results = pd.DataFrame(results).sort_index(axis=1)

    def load_configuration(self):
        """Load configuration yaml card.

        Returns
        -------
        dict
            configuration card

        """
        with open(f"{self.path}/{self.name}/{self.name}.yaml", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config

    def load_datasets(self):
        """Load all datasets."""
        self.datasets = load_datasets(
            self.config["data_path"],
            self.config["datasets"],
            self.config["coefficients"],
            self.config["order"],
            self.config["use_quad"],
            self.config["use_theory_covmat"],
            False,  # t0 is not used here because in the report we look at the experimental chi2
            self.config.get("use_multiplicative_prescription", False),
            self.config.get("theory_path", None),
            self.config.get("rot_to_fit_basis", None),
            self.config.get("uv_couplings", False),
            self.config.get("external_chi2", False),
            self.config.get("poly_mode", False),
            self.config.get("external_coefficients", False),
        )

    @property
    def smeft_predictions(self):
        """Compute |SMEFT| predictions for each replica.

        Returns
        -------
        np.ndarray:
            |SMEFT| predictions for each replica
        """

        smeft = []
        for rep in track(
            range(self.n_replica),
            description=f"[green]Computing SMEFT predictions for each replica of {self.name}...",
        ):
            smeft.append(
                make_predictions(
                    self.datasets,
                    self.results.iloc[rep, :],
                    self.config["use_quad"],
                    self.config.get("use_multiplicative_prescription", False),
                    self.config.get("poly_mode", False),
                )
            )
        return np.array(smeft)

    @property
    def coefficients(self):
        """coefficient manager"""
        return CoefficientManager.from_dict(self.config["coefficients"])

    @property
    def n_replica(self):
        """Number of replicas"""
        return self.results.shape[0]
