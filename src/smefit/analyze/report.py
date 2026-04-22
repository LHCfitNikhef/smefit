# -*- coding: utf-8 -*-
import copy
import pathlib

import numpy as np
import pandas as pd

from ..fit_manager import FitManager
from ..log import logging
from .chi2_utils import Chi2tableCalculator
from .coefficients_utils import CoefficientsPlotter, compute_confidence_level
from .correlations import plot_correlations
from .fisher import FisherCalculator
from .html_utils import html_link, write_html_container
from .latex_tools import compile_tex
from .pca import PcaCalculator
from .summary import SummaryWriter

_logger = logging.getLogger(__name__)


class Report:
    r"""Report class manager.

    If :math:`\chi^2`, Fisher or Data vs Theory plots are produced it computes the
    best fit theory predictions.

    Attributes
    ----------
    report: str
        path to report folder
    fits: numpy.ndarray
        array with fits (instances of `smefit.fit_manager.FitManger`) included in the report
    data_info: pandas.DataFrame
        datasets information (references and data groups)
    coeff_info: pandas.DataFrame
        coefficients information (group and latex name)

    Parameters
    ----------
    report_path: pathlib.Path, str
        path to base folder, where the reports will be stored.
    result_path: pathlib.Path, str
        path to base folder, where the results are stored.
    report_config: dict
        dictionary with report configuration, see `/run_cards/analyze/report_runcard.yaml`
        for an example

    """

    def __init__(self, report_path, result_path, report_config):
        self.report = pathlib.Path(f"{report_path}/{report_config['name']}").absolute()
        self.fits = []
        # build the fits labels if needed
        if "fit_labels" not in report_config:
            fit_labels = [
                f"${{\\rm {fit}}}$".replace("_", r"\ ")
                for fit in report_config["result_IDs"]
            ]
        else:
            fit_labels = report_config["fit_labels"]
        # Loads fits
        for name, label in zip(report_config["result_IDs"], fit_labels):
            fit = FitManager(result_path, name, label)
            fit.load_results()

            if any(k in report_config for k in ["chi2_plots", "PCA", "fisher"]):
                fit.load_datasets()

            self.fits.append(fit)
        self.fits = np.array(self.fits)

        # Get names of datasets for each fit
        self.dataset_fits = []
        for fit in self.fits:
            self.dataset_fits.append([data["name"] for data in fit.config["datasets"]])

        # Loads useful information about data
        self.data_info = self._load_grouped_data_info(report_config["data_info"])
        # Loads coefficients grouped with latex name
        self.coeff_info = self._load_grouped_coeff_info(report_config["coeff_info"])
        self.html_index = ""
        self.html_content = ""

    def _load_grouped_data_info(self, raw_dict):
        """Load grouped info of datasets.

        Only elements appearing at least once in the fit configs are kept.

        Parameters
        ----------
        raw_dict: dict
            raw dictionary with relevant information

        Returns
        _______
        grouped_config: pandas.DataFrame
            table with information by group

        """
        out_dict = {}
        for group, entries in raw_dict.items():
            out_dict[group] = {}
            for val in entries:
                if np.any([val[0] in datasets for datasets in self.dataset_fits]):
                    out_dict[group][val[0]] = val[1]

            if len(out_dict[group]) == 0:
                out_dict.pop(group)
        return pd.DataFrame(out_dict).stack().swaplevel()

    def _load_grouped_coeff_info(self, raw_dict):
        """Load grouped info of coefficients.

        Only elements appearing at least once in the fit configs are kept.

        Parameters
        ----------
        raw_dict: dict
            raw dictionary with relevant information

        Returns
        _______
        grouped_config: pandas.DataFrame
            table with information by group

        """
        out_dict = {}
        for group, entries in raw_dict.items():
            out_dict[group] = {}
            for val in entries:
                if np.any([val[0] in fit.config["coefficients"] for fit in self.fits]):
                    out_dict[group][val[0]] = val[1]

            if len(out_dict[group]) == 0:
                out_dict.pop(group)
        return pd.DataFrame(out_dict).stack().swaplevel()

    def _append_section(self, title, links=None, figs=None, tables=None):
        self.html_index += html_link(f"#{title}", title, add_meta=False)
        self.html_content += write_html_container(
            title, links=links, figs=figs, dataFrame=tables
        )

    def summary(self):
        """Summary Table runner."""
        summary = SummaryWriter(self.fits, self.data_info, self.coeff_info)
        section_title = "Summary"
        coeff_tab = "coefficient_summary"
        data_tab = "dataset_summary"

        # write summary tables
        compile_tex(self.report, summary.write_coefficients_table(), coeff_tab)
        compile_tex(self.report, summary.write_dataset_table(), data_tab)

        self._append_section(
            section_title,
            links=[(data_tab, "Dataset summary"), (coeff_tab, "Coefficient summary")],
            tables=summary.fit_settings(),
        )

    def chi2(self, table=True, plot_experiment=None, plot_distribution=None):
        r""":math:`\chi^2` table and plots runner.

        Parameters
        ----------
        table: bool, optional
            write the latex :math:`\chi^2` table per dataset
        plot_experiment: bool, optional
            plot the :math:`\chi^2` per dataset
        plot_distribution: bool, optional
            plot the :math:`\chi^2` distribution per each replica

        """
        links_list = None
        figs_list = []
        chi2_cal = Chi2tableCalculator(self.data_info)

        # here we store the info for each fit
        chi2_dict = {}
        chi2_dict_group = {}
        chi2_replica = {}
        for fit in self.fits:
            # This computes the chi2 by taking the mean of the replicas
            _, chi2_total_rep = chi2_cal.compute(
                fit.datasets,
                fit.smeft_predictions,
            )

            chi2_df_best, _ = chi2_cal.compute(
                fit.datasets, fit.smeft_predictions_best_fit
            )

            chi2_replica[fit.label] = chi2_total_rep
            chi2_dict[fit.label] = chi2_cal.add_normalized_chi2(chi2_df_best)
            chi2_dict_group[fit.label] = chi2_cal.group_chi2_df(chi2_df_best)

        if table:
            lines = chi2_cal.write(chi2_dict, chi2_dict_group)
            compile_tex(self.report, lines, "chi2_tables")
            links_list = [("chi2_tables", "Tables")]

        if plot_experiment is not None:
            _logger.info("Plotting : chi^2 for each dataset")
            chi2_cal.plot_exp(chi2_dict, f"{self.report}/chi2_bar", **plot_experiment)
            figs_list.append("chi2_bar")

        if plot_distribution is not None:
            _logger.info("Plotting : chi^2 distribution for each replica")
            chi2_cal.plot_dist(
                chi2_replica, f"{self.report}/chi2_histo", **plot_distribution
            )
            figs_list.append("chi2_histo")

        self._append_section("Chi2", links=links_list, figs=figs_list)

    def coefficients(
        self,
        scatter_plot=None,
        confidence_level_bar=None,
        pull_bar=None,
        spider_plot=None,
        posterior_histograms=True,
        contours_2d=None,
        hide_dofs=None,
        show_only=None,
        logo=True,
        table=None,
        double_solution=None,
        ci_type="eti",
    ):
        """Coefficients plots and table runner.

        Parameters
        ----------
        hide_dofs: list
            list of operator not to display
        show_only: list
            list of all the operator to display, if None all the free dof are presented
        logo: bool
            if True add logo to the plots
        scatter_plot: None, dict
            kwarg confidence level bar plot or None
        confidence_level_bar: None, dict
            kwarg scatter plot or None
        posterior_histograms: bool
            if True plot the posterior distribution for each coefficient
        table: None, dict
            kwarg the latex confidence level table per coefficient or None
        double_solution: dict
            operator with double solution per fit
        ci_type: str
            type of confidence interval to compute, either 'eti', 'hdi' or 'hdi_mono'

        """
        links_list = None
        figs_list = []
        coeff_config = self.coeff_info
        if show_only is not None:
            coeff_config = coeff_config.loc[:, show_only]
        if hide_dofs is not None:
            coeff_config = coeff_config.drop(hide_dofs, level=1)

        coeff_plt = CoefficientsPlotter(
            self.report,
            coeff_config,
            logo=logo,
        )

        # compute confidence level bounds
        bounds_dict = {}
        for fit in self.fits:
            bounds_dict[fit.label] = compute_confidence_level(
                fit.results["samples"],
                coeff_plt.coeff_info,
                fit.has_posterior,
                (
                    double_solution.get(fit.name, None)
                    if double_solution is not None
                    else None
                ),
            )

        if scatter_plot is not None:
            _logger.info("Plotting : Central values and Confidence Level bounds")
            coeff_plt.plot_coeffs(bounds_dict, **scatter_plot, ci_type=ci_type)
            figs_list.append("coefficient_central")

        # when we plot the 95% CL we show the 95% CL for null solutions.
        # the error coming from a degenerate solution is not taken into account.
        if confidence_level_bar is not None:
            _logger.info("Plotting : Confidence Level error bars")
            bar_cl = confidence_level_bar["confidence_level"]
            confidence_level_bar.pop("confidence_level")
            zero_sol = 0
            if ci_type == "hdi":
                coeff_plt.plot_coeffs_bar(
                    {
                        name: bound_df.loc[zero_sol, f"hdi_{bar_cl}"]
                        for name, bound_df in bounds_dict.items()
                    },
                    **confidence_level_bar,
                )
            elif ci_type == "hdi_mono":
                coeff_plt.plot_coeffs_bar(
                    {
                        name: bound_df.loc[zero_sol, f"hdi_mono_{bar_cl}"]
                        for name, bound_df in bounds_dict.items()
                    },
                    **confidence_level_bar,
                )
            else:
                coeff_plt.plot_coeffs_bar(
                    {
                        name: -bound_df.loc[zero_sol, f"low{bar_cl}"]
                        + bound_df.loc[zero_sol, f"high{bar_cl}"]
                        for name, bound_df in bounds_dict.items()
                    },
                    **confidence_level_bar,
                )
            figs_list.append("coefficient_bar")

        # when we plot the 95% CL we show the 95% CL for null solutions.
        # the error coming from a degenerate solution is not taken into account.
        if pull_bar is not None:
            _logger.info("Plotting : Pull ")
            zero_sol = 0
            string_to_find = (
                "pull_hdi" if (ci_type == "hdi_mono" or ci_type == "hdi") else "pull"
            )
            coeff_plt.plot_pull(
                {
                    name: bound_df.loc[zero_sol, string_to_find]
                    for name, bound_df in bounds_dict.items()
                },
                **pull_bar,
            )
            figs_list.append("pull_bar")

        if spider_plot is not None:
            _logger.info("Plotting : spider plot")

            spider_cl = spider_plot["confidence_level"]
            spider_plot.pop("confidence_level")

            spider_bounds = {}
            for name, bound_df in bounds_dict.items():
                # dbl_solution = bound_df.index.get_level_values(0)
                # if dbl solution requested, add the confidence intervals, otherwise just
                # use the sum of the hdi intervals
                # if 1 in dbl_solution:
                #    dbl_op = double_solution.get(fit.name, None)
                #    idx = [
                #        np.argwhere(
                #            self.coeff_info.index.get_level_values(1) == op
                #        ).flatten()[0]
                #        for op in dbl_op
                #    ]
                #    bound_df_dbl = bound_df.iloc[:, idx]
                #    width_0 = bound_df_dbl.loc[0, f"hdi_{spider_cl}"]
                #    width_1 = bound_df_dbl.loc[1, f"hdi_{spider_cl}"]
                #    width_tot = width_0 + width_1

                # update bound df
                #    bound_df.loc[0, f"hdi_{spider_cl}"].iloc[idx] = width_tot

                #    spider_bounds[name] = bound_df.loc[0, f"hdi_{spider_cl}"]
                if ci_type == "hdi":
                    spider_bounds[name] = bound_df.loc[0, f"hdi_{spider_cl}"]
                elif ci_type == "hdi_mono":
                    spider_bounds[name] = bound_df.loc[0, f"hdi_mono_{spider_cl}"]
                else:
                    spider_bounds[name] = bound_df.loc[0, f"mean_err{spider_cl}"] * 2.0

            coeff_plt.plot_spider(
                spider_bounds,
                labels=[fit.label for fit in self.fits],
                **spider_plot,
            )
            figs_list.append("spider_plot")

        if posterior_histograms:
            _logger.info("Plotting : Posterior histograms")
            disjointed_lists = [
                (
                    double_solution.get(fit.name, None)
                    if double_solution is not None
                    else None
                )
                for fit in self.fits
            ]
            posterior_histograms["disjointed_lists"] = disjointed_lists

            coeff_plt.plot_posteriors(
                [fit.results["samples"] for fit in self.fits],
                labels=[fit.label for fit in self.fits],
                **posterior_histograms,
            )
            figs_list.append("coefficient_histo")

        if table is not None:
            _logger.info("Writing : Confidence level table")
            lines = coeff_plt.write_cl_table(bounds_dict, **table, ci_type=ci_type)
            compile_tex(self.report, lines, "coefficients_table")
            links_list = [("coefficients_table", "CL table")]

        if contours_2d:
            _logger.info("Plotting : 2D confidence level projections")
            coeff_plt.plot_contours_2d(
                [
                    (
                        fit.results["samples"][fit.coefficients.free_parameters.index],
                        fit.config["use_quad"],
                    )
                    for fit in self.fits
                ],
                labels=[fit.label for fit in self.fits],
                confidence_level=contours_2d["confidence_level"],
                dofs_show=contours_2d["dofs_show"],
                double_solution=double_solution,
            )
            figs_list.append("contours_2d")

        self._append_section("Coefficients", links=links_list, figs=figs_list)

    def correlations(
        self, hide_dofs=None, thr_show=0.1, title=True, fit_list=None, figsize=(10, 10)
    ):
        """Plot coefficients correlation matrix.

        Parameters
        ----------
        hide_dofs: list
            list of operator not to display.
        thr_show: float, None
            minimum threshold value to show.
            If None the full correlation matrix is displayed.
        title: bool
            if True display fit label name as title
        fit_list: list, optional
            list of fit names for which the correlation is computed.
            By default all the fits included in the report
        """
        figs_list = []

        if fit_list is not None:
            fit_list = [fit for fit in self.fits if fit in fit_list]
        else:
            fit_list = self.fits

        for fit in fit_list:
            _logger.info(f"Plotting correlations for: {fit.name}")
            coeff_to_keep = fit.coefficients.free_parameters.index
            plot_correlations(
                fit.results["samples"][coeff_to_keep],
                latex_names=self.coeff_info.droplevel(0),
                fig_name=f"{self.report}/correlations_{fit.name}",
                title=fit.label if title else None,
                hide_dofs=hide_dofs,
                thr_show=thr_show,
                figsize=figsize,
            )
            figs_list.append(f"correlations_{fit.name}")

        self._append_section("Correlations", figs=figs_list)

    def pca(
        self,
        table=True,
        plot=None,
        thr_show=1e-2,
        fit_list=None,
    ):
        """Principal Components Analysis runner.

        Parameters
        ----------
        table: bool, optional
            if True writes the PC directions in a latex list
        plot: bool, optional
            if True produces a PC heatmap
        thr_show: float
            minimum threshold value to show
        fit_list: list, optional
            list of fit names for which the PCA is computed.
            By default all the fits included in the report
        """
        figs_list, links_list = [], []
        if fit_list is not None:
            fit_list = [fit for fit in self.fits if fit in fit_list]
        else:
            fit_list = self.fits
        for fit in fit_list:
            _logger.info(f"Computing PCA for fit {fit.name}")
            pca_cal = PcaCalculator(
                fit.datasets,
                fit.coefficients,
                self.coeff_info.droplevel(0),
            )
            pca_cal.compute()

            if table:
                compile_tex(
                    self.report,
                    pca_cal.write(fit.label, thr_show),
                    f"pca_table_{fit.name}",
                )
                links_list.append((f"pca_table_{fit.name}", f"Table {fit.label}"))
            if plot is not None:
                title = fit.name

                # TODO: check why **fit_plot got removed (see PR)
                pca_cal.plot_heatmap(
                    f"{self.report}/pca_heatmap_{fit.name}",
                    title=title,
                    figsize=plot["figsize"],
                )
                figs_list.append(f"pca_heatmap_{fit.name}")
        self._append_section("PCA", figs=figs_list, links=links_list)

    def fisher(
        self, norm="coeff", summary_only=True, plot=None, fit_list=None, log=False
    ):
        """Fisher information table and plots runner.

        Summary table and plots are the default

        Parameters
        ----------
        norm: "coeff", "dataset"
            fisher information normalization: per coefficient, or per dataset
        summary_only: bool, optional
            if False writes the fine grained fisher tables per dataset and group
            if True only the summary table with grouped a datsets is written
        plot: None, dict
            plot options
        fit_list: list, optional
            list of fit names for which the fisher information is computed.
            By default all the fits included in the report
        log: bool, optional
            if True shows the log of the Fisher informaltion

        """
        figs_list, links_list = [], []
        if fit_list is not None:
            fit_list = [fit for fit in self.fits if fit in fit_list]
        else:
            fit_list = self.fits

        fishers = {}
        for fit in fit_list:
            compute_quad = fit.config["use_quad"]
            fisher_cal = FisherCalculator(fit.coefficients, fit.datasets, compute_quad)
            fisher_cal.compute_linear()
            fisher_cal.lin_fisher = fisher_cal.normalize(
                fisher_cal.lin_fisher, norm=norm, log=log
            )
            fisher_cal.summary_table = fisher_cal.groupby_data(
                fisher_cal.lin_fisher, self.data_info, norm, log
            )
            fishers[fit.name] = fisher_cal

            # if necessary compute the quadratic Fisher
            if compute_quad:
                fisher_cal.compute_quadratic(
                    fit.results["samples"], fit.smeft_predictions
                )
                fisher_cal.quad_fisher = fisher_cal.normalize(
                    fisher_cal.quad_fisher, norm=norm, log=log
                )
                fisher_cal.summary_HOtable = fisher_cal.groupby_data(
                    fisher_cal.quad_fisher, self.data_info, norm, log
                )

            compile_tex(
                self.report,
                fisher_cal.write_grouped(self.coeff_info, self.data_info, summary_only),
                f"fisher_{fit.name}",
            )
            links_list.append((f"fisher_{fit.name}", f"Table {fit.label}"))

            if plot is not None:
                fit_plot = copy.deepcopy(plot)
                fit_plot.pop("together", None)
                title = fit.label if fit_plot.pop("title") else None
                fisher_cal.plot_heatmap(
                    self.coeff_info,
                    f"{self.report}/fisher_heatmap_{fit.name}",
                    title=title,
                    **fit_plot,
                )
                figs_list.append(f"fisher_heatmap_{fit.name}")

        # plot both fishers
        if plot.get("together", False):
            fisher_1 = fishers[plot["together"][0]]
            fisher_2 = fishers[plot["together"][1]]
            fit_plot = copy.deepcopy(plot)
            fit_plot.pop("together")

            # show title of last fit
            title = fit.label if fit_plot.pop("title") else None

            # make heatmap of fisher_1 and fisher_2
            fisher_2.plot_heatmap(
                self.coeff_info,
                f"{self.report}/fisher_heatmap_both",
                title=title,
                other=fisher_1,
                labels=[fit.label for fit in self.fits],
                **fit_plot,
            )
            figs_list.append(f"fisher_heatmap_both")

        self._append_section("Fisher", figs=figs_list, links=links_list)
