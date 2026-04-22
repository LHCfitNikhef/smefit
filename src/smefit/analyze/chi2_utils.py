# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .latex_tools import chi2table_header, latex_packages


class Chi2tableCalculator:
    r"""Compute the :math:`\chi^2` for each replica and produce:

        * Tables with :math:`\chi^2` for each dataset and datagroup.
        * Plot of :math:`\chi^2` for each dataset.
        * Plot of :math:`\chi^2` for each replica

    Parameters
    ----------
    data_info: pandas.DataFrame
        datasets information (references and data groups)

    """

    def __init__(self, data_info):
        self.data_info = data_info
        self.chi2_df_sm = pd.DataFrame()
        self.chi2_df_sm_grouped = pd.DataFrame()

    @staticmethod
    def compute(datasets, smeft_predictions):
        r"""Compute the :math:`\chi^2` for each replica and dataset.

        Parameters
        ----------
            datasets: smefit.loader.DataTuple
                loaded datasets
            smeft_predictions: np.ndarray
                array with all the predictions for each replica

        Returns
        -------
        pd.DataFrame:
            :math:`\chi^2` for each dataset
        np.ndarray:
            :math:`\chi^2/n_{pts}` for each replica

        """
        chi2 = []
        chi2_sm = []
        chi2_rep = []

        diff = datasets.Commondata - np.mean(smeft_predictions, axis=0)
        covmat_diff = datasets.InvCovMat @ diff

        diff_sm = datasets.Commondata - datasets.SMTheory
        covmat_diff_sm = datasets.InvCovMat @ diff_sm

        # do the difference replica by replica and multiply by cov inverse
        diff_rep = (
            np.tile(datasets.Commondata, (smeft_predictions.shape[0], 1))
            - smeft_predictions
        )
        covmat_diff_rep = datasets.InvCovMat @ diff_rep.T

        # Compute per experiment
        cnt = 0
        for ndat_exp in datasets.NdataExp:
            chi2.append(
                np.dot(
                    diff[cnt : cnt + ndat_exp],
                    covmat_diff[cnt : cnt + ndat_exp],
                )
            )
            chi2_sm.append(
                np.dot(
                    diff_sm[cnt : cnt + ndat_exp],
                    covmat_diff_sm[cnt : cnt + ndat_exp],
                )
            )

            # Compute chi2 by replica
            # multiply the second term of the chi2 and
            # take the diagonal (replica left = replica right).
            # here np.einsum is faster than np.diag(a @ b), since
            # a and b are large usually
            chi2_rep.append(
                np.einsum(
                    "ij,ji->i",
                    diff_rep[:, cnt : cnt + ndat_exp],
                    covmat_diff_rep[cnt : cnt + ndat_exp],
                )
            )
            cnt += ndat_exp

        # compute chi2 for the whole dataset
        total_chi2_rep = np.einsum("ij,ji->i", diff_rep, covmat_diff_rep)

        return (
            pd.DataFrame(
                {
                    "ndat": datasets.NdataExp,
                    "chi2": np.array(chi2),
                    "chi2_std": np.std(chi2_rep, axis=1),
                    "chi2_sm": np.array(chi2_sm),
                },
                index=datasets.ExpNames,
            ),
            total_chi2_rep / datasets.Commondata.size,
        )

    @staticmethod
    def compute_ext_chi2(external_chi2, best_fit):
        r"""Compute the external likelihood for each dataset.

        Parameters
        ----------
            external_chi2: dict
                dictionary whose keys are the different datasets and the values
                the compute_chi2 function of the external likelihood module. Obtained
                by fit.external_chi2
            best_fit: pd.core.series.Series
                best fit value for all the Wilson coefficients. Obtained by
                the fit.best_fit property

        Returns
        -------
        pd.DataFrame:
            External chi2 for each dataset
        """
        ext_chi2_values = [ext_chi2_func(best_fit) for ext_chi2_func in external_chi2]

        # The SM ext. likelihood can be obtained through the ext. likelihood module
        # by setting all the WCs to zero
        sm_coeff = pd.Series(0, index=best_fit.index, dtype=best_fit.dtype)
        ext_chi2_sm_values = [
            ext_chi2_func(sm_coeff) for ext_chi2_func in external_chi2
        ]

        indices = [ext_func.__self__.__class__.__name__ for ext_func in external_chi2]
        return pd.DataFrame(
            {
                "sm_chi2": np.array(ext_chi2_sm_values),
                "ext_chi2": np.array(ext_chi2_values),
            },
            index=indices,
        )

    @staticmethod
    def add_normalized_chi2(chi2_df):
        r"""Add the normalized :math:`\chi^2` to the table.

        Parameters
        ----------
        chi2_df : pd.DataFrame
            :math:`\chi^2` table for each dataset

        Returns
        -------
        pd.DataFrame:
            :math:`\chi^2` table for each dataset with normalization
        """
        # reduced chi2
        chi2_df["chi2/ndat"] = chi2_df["chi2"] / chi2_df["ndat"]
        chi2_df["chi2_sm/ndat"] = chi2_df["chi2_sm"] / chi2_df["ndat"]
        return chi2_df

    @staticmethod
    def _add_chi2_df_colors(chi2_df):
        r"""Values higer than one std are labelled with blue.
        Values lowe  than one std are labelled with red.
        """
        chi2_upper = chi2_df["chi2"] + chi2_df["chi2_std"]
        chi2_lower = chi2_df["chi2"] - chi2_df["chi2_std"]
        chi2_df["color"] = "black"
        chi2_df.loc[chi2_df["chi2_sm"] > chi2_upper, "color"] = "blue"
        chi2_df.loc[chi2_df["chi2_sm"] < chi2_lower, "color"] = "red"
        return chi2_df

    def group_chi2_df(self, chi2_df):
        r"""Group the :math:`\chi^2` according to the data type.

        Parameters
        ----------
        chi2_df : pd.DataFrame
            :math:`\chi^2` table for each dataset

        Returns
        -------
        pd.DataFrame:
            :math:`\chi^2` table with deviation info
        """
        chi2_df_grouped = pd.merge(
            self.data_info.reset_index(), chi2_df, left_on="level_1", right_index=True
        ).drop([0, "chi2_std"], axis=1)
        chi2_df_grouped = chi2_df_grouped.groupby("level_0").sum(numeric_only=True)
        chi2_df_grouped.index.name = "data_group"

        # add total values
        chi2_df_grouped.loc["Total"] = chi2_df_grouped.sum()
        chi2_df_grouped["chi2/ndat"] = chi2_df_grouped["chi2"] / chi2_df_grouped["ndat"]
        chi2_df_grouped["chi2_sm/ndat"] = (
            chi2_df_grouped["chi2_sm"] / chi2_df_grouped["ndat"]
        )
        return chi2_df_grouped

    def _split_table_entries_sm(self, chi2_dict, chi2_dict_group):
        """Update the chi2_df dict for all included datasets."""
        labels_to_include = ["ndat", "chi2_sm/ndat"]
        for chi2_df, chi2_df_grouped in zip(
            chi2_dict.values(), chi2_dict_group.values()
        ):
            self.chi2_df_sm = pd.concat(
                [self.chi2_df_sm, chi2_df[labels_to_include]],
                axis=0,
            )
            self.chi2_df_sm_grouped = pd.concat(
                [self.chi2_df_sm_grouped, chi2_df_grouped[labels_to_include]],
                axis=0,
            )
        # TODO: is this woking with different element in each group?
        self.chi2_df_sm = self.chi2_df_sm[
            ~self.chi2_df_sm.index.duplicated(keep="first")
        ]
        self.chi2_df_sm_grouped = self.chi2_df_sm_grouped.drop_duplicates()

    def write(self, chi2_dict, chi2_dict_group, chi2_ext_dict):
        r"""Write the :math:`\chi^2` latex tables.

        Parameters
        ----------
        chi2_dict : dict
            tables computed with compute() method for each fit
        chi2_dict_group: dict
            tables obtained with group_chi2_df() method for each fit
        chi2_ext_dict: dict
            tables computed with compute_ext_chi2() method for each fit

        Returns
        -------
        list(str)
            list with the latex commands
        """
        self._split_table_entries_sm(chi2_dict, chi2_dict_group)
        L = latex_packages()
        L.extend([r"\usepackage{underscore}", r"\begin{document}"])
        L.extend(self.write_chi2_grouped(chi2_dict, chi2_dict_group))
        L.extend(["\n", "\n"])
        L.extend(self.write_chi2_summary(chi2_dict_group))

        if chi2_ext_dict != {}:
            L.extend(["\n", "\n"])
            L.extend(self.write_external_chi2(chi2_ext_dict))

        return L

    def write_external_chi2(self, ext_chi2_dict):
        r"""Write the external likelihood latex tables for each dataset.

        Parameters
        ----------
        ext_chi2_dict : dict
            tables computed with compute_ext_chi2() method per each fit

        Returns
        -------
        list(str)
            list with the latex commands
        """
        L = [
            r"\begin{table}[H]",
            r"\centering",
            r"\begin{tabular}{|l|" + "c|c|" * len(ext_chi2_dict) + "}",
            r"\hline",
        ]

        temp = r""
        for label in ext_chi2_dict.keys():
            temp += f"& \\multicolumn{{2}}{{c|}}{{{label}}}"
        temp += r"\\ \hline"
        L.append(temp)
        L.append(
            r"Process " + r" & SM & Best fit" * len(ext_chi2_dict) + r"\\ \hline",
        )

        # Extract unique ext. likelihood datasets
        datasets = set()
        for df in ext_chi2_dict.values():
            datasets.update(df.index)
        datasets = list(datasets)

        total_chi2 = {group: 0 for group in ext_chi2_dict.keys()}
        total_chi2_sm = {group: 0 for group in ext_chi2_dict.keys()}
        for dataset in datasets:
            temp = f"{dataset}"
            for group, ext_chi2_df in ext_chi2_dict.items():
                if dataset in ext_chi2_df.index:
                    temp += f" & {ext_chi2_df.loc[dataset, 'sm_chi2']:.3f}"
                    temp += f" & {ext_chi2_df.loc[dataset, 'ext_chi2']:.3f}"
                    total_chi2[group] += ext_chi2_df.loc[dataset, "ext_chi2"]
                    total_chi2_sm[group] += ext_chi2_df.loc[dataset, "sm_chi2"]
                else:
                    temp += " & &"
            temp += r" \\ \hline"
            L.append(temp)

        temp = r" \hline Total"
        for group in ext_chi2_dict.keys():
            temp += f" & {total_chi2_sm[group]:.3f} & {total_chi2[group]:.3f}"
        temp += r" \\ \hline"

        L.extend(
            [
                temp,
                r"\end{tabular}",
                r"\caption{External likelihood table for each dataset.}",
                r"\end{table}",
            ]
        )
        return L

    def write_chi2_grouped(self, chi2_dict, chi2_dict_group):
        r"""Write the :math:`\chi^2` latex tables for each data group.

        Parameters
        ----------
        chi2_dict : dict
            tables computed with compute() method per each fit

        Returns
        -------
        list(str)
            list with the latex commands
        """
        L = [
            r"$\chi^2$ table. Blue color text represents a value that is lower than the SM $\chi^2$ \
            by more than one standard deviation of the $\chi^2$ distribution.\
            Similarly, red color text represents values that are higher than the SM $\chi^2$ by more than one standard deviation.\
            In parenthesis is the total SM $\chi^2$ for the dataset included in the fit. \\"
        ]
        for group, datasets in self.data_info.groupby(level=0):
            L.extend(
                [
                    r"\begin{table}[H]",
                    r"\centering",
                    r"\begin{tabular}{|l|c|c|" + "c|" * len(chi2_dict) + "}",
                ]
            )
            L = chi2table_header(L, chi2_dict.keys())

            # loop over datasets
            for dataset, link in datasets.droplevel(0).items():
                temp = (
                    f"\\href{{{link}}}{{{dataset}}}"
                    + f" & {int(self.chi2_df_sm.loc[dataset,'ndat'])}"
                    + f" & {self.chi2_df_sm.loc[dataset,'chi2_sm/ndat'].round(3)}"
                )
                for chi2_df in chi2_dict.values():
                    temp += " & "
                    chi2_df = self._add_chi2_df_colors(chi2_df)
                    if dataset in chi2_df.index:
                        temp += f"\\textcolor{{{chi2_df.loc[dataset, 'color']}}}\
                            {{{chi2_df.loc[dataset, 'chi2/ndat']:.3f}}}"
                temp += r" \\ \hline"
                L.append(temp)

            closing_line = r"\hline Total & & "
            for chi2_df_grouped in chi2_dict_group.values():
                closing_line += " & "
                if group in chi2_df_grouped.index:
                    closing_line += f"{chi2_df_grouped.loc[group, 'chi2/ndat']:.3f} ({chi2_df_grouped.loc[group, 'chi2_sm/ndat']:.3f})"
            closing_line += r" \\ \hline"
            L.extend(
                [
                    closing_line,
                    r"\end{tabular}",
                    f"\\caption{{$\\chi^2$ table for {group} data}}",
                    r"\end{table}",
                ]
            )
        return L

    def write_chi2_summary(self, chi2_dict_group):
        r"""Write the summary :math:`\chi^2` table for grouped data.

        Parameters
        ----------
        chi2_dict_group : dict
            tables obtained with group_chi2_df() method for each fit

        Returns
        -------
        list(str)
           list with the latex commands

        """
        L = [
            r"\begin{table}[H]",
            r"\centering",
            r"\begin{tabular}{|l|" + "c|c|" * len(chi2_dict_group) + "}",
            r"\hline",
        ]
        temp_parts = [
            f"& \\multicolumn{{2}}{{c|}}{{{label}}}" for label in chi2_dict_group
        ]
        temp = "".join(temp_parts)
        temp += r"\\ \hline"
        L.append(temp)

        L.append(
            r"Process "
            + " ".join(
                [r"& $N_{\rm data}$ & $\chi^2/N_{\rm data}$" for _ in chi2_dict_group]
            )
            + r"\\ \hline"
        )

        for group in self.data_info.index.levels[0]:
            temp = f"{group}"
            for chi2_df_grouped in chi2_dict_group.values():
                if group in chi2_df_grouped.index:
                    temp += f" & {chi2_df_grouped.loc[group, 'ndat']} \
                        & {chi2_df_grouped.loc[group, 'chi2/ndat']:.3f} \
                            ({chi2_df_grouped.loc[group, 'chi2_sm/ndat']:.3f})"
                else:
                    temp += " & & "
            temp += r" \\ \hline"
            L.append(temp)

        temp = r" \hline Total"
        for chi2_df_grouped in chi2_dict_group.values():
            temp += f" & {chi2_df_grouped.loc['Total', 'ndat']} \
                & {chi2_df_grouped.loc['Total', 'chi2/ndat']:.3f} \
                    ({chi2_df_grouped.loc['Total', 'chi2_sm/ndat']:.3f})"
        temp += r" \\ \hline"
        L.extend(
            [
                temp,
                r"\end{tabular}",
                r"\caption{$\chi^2$ table for grouped data. In parenthesis is the total SM $\chi^2$ for the dataset included in the fit.\
                    The SM column refers to all the datasets available in the group}",
                r"\end{table}",
            ]
        )
        return L

    def plot_exp(
        self,
        chi2_dict,
        fig_name,
        figsize=(10, 15),
    ):
        r"""Plots a bar plot of the :math:`\chi^2` values per experiment"""

        chi2_bar = pd.DataFrame()
        chi2_bar[r"${\rm SM}$"] = self.chi2_df_sm["chi2_sm/ndat"]
        for name, chi2_df in chi2_dict.items():
            chi2_bar[name] = chi2_df["chi2/ndat"]
        chi2_bar.index = [
            f"\\rm{{{name}}}".replace("_", r"\_") for name in chi2_bar.index
        ]
        chi2_bar.plot(kind="barh", width=0.7, figsize=figsize)

        plt.vlines(1, -1, chi2_bar.shape[0] * 10, ls="dashed", color="black", alpha=0.5)
        x_max = chi2_bar.max().max()
        plt.vlines(
            np.arange(2, int(x_max) + 1),
            -1,
            chi2_bar.shape[0] * 10,
            ls="dashed",
            color="grey",
            lw=0.5,
        )
        plt.tick_params(axis="x", direction="in", labelsize=15)
        plt.xlabel(r"$\chi^2$", fontsize=20)
        plt.xlim(-0.1, chi2_bar.max().max() + 0.2)
        plt.legend(loc="upper right", frameon=False, prop={"size": 11})
        plt.tight_layout()
        plt.savefig(f"{fig_name}.pdf")
        plt.savefig(f"{fig_name}.png")

    def plot_dist(self, chi2_hist, fig_name, figsize=(7, 5)):
        r"""Plots the :math:`\chi^2` distribution."""
        plt.figure(figsize=figsize)
        ax = plt.subplot(111)

        for label, chi2_list in chi2_hist.items():
            ax.hist(
                chi2_list,
                bins="fd",
                density=True,
                edgecolor="k",
                alpha=0.3,
                label=label,
            )

        plt.tick_params(axis="x", direction="in", labelsize=15)
        plt.tick_params(axis="y", direction="in", labelsize=15)
        plt.xlabel(r"\rm $\chi^2$\ distribution", fontsize=20)
        plt.legend(loc="best", frameon=False, prop={"size": 11})
        plt.tight_layout()
        plt.savefig(f"{fig_name}.pdf")
        plt.savefig(f"{fig_name}.png")
