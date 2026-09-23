import numpy as np
import arviz as az
import pathlib
import json
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.lines import Line2D
import pandas as pd
from matplotlib.patches import Patch

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 20})
rc('text', usetex=True)

compute_bounds = True

result_dir = pathlib.Path(
    "/data/theorie/jthoeve/physics_projects/new_smefit/results/smefit_esppu_rebuttal")

runs_ind_lin = ["260921_jth_HLLHC_250GEV_LIN_IND_aggressive",
                "260921_jth_LEP3_250GEV_LIN_IND_aggressive",
                "260921_jth_FCCee_250GEV_LIN_IND_aggressive",
                "260921_jth_LCF500_250GEV_LIN_IND_aggressive",
                "260921_jth_LCF1000_250GEV_LIN_IND_aggressive"]

runs_ind_quad = ["260921_jth_HLLHC_250GEV_QUAD_IND_aggressive",
                 "260921_jth_LEP3_250GEV_QUAD_IND_aggressive",
                 "260921_jth_FCCee_250GEV_QUAD_IND_aggressive",
                 "260921_jth_LCF500_250GEV_QUAD_IND_aggressive",
                 "260921_jth_LCF1000_250GEV_QUAD_IND_aggressive"]

runs_glob_lin = ["260921_jth_HLLHC_250GEV_LIN_GLOB_aggressive",
                 "260921_jth_LEP3_250GEV_LIN_GLOB_aggressive",
                 "260921_jth_FCCee_250GEV_LIN_GLOB_aggressive",
                 "260921_jth_LCF500_250GEV_LIN_GLOB_aggressive",
                 "260921_jth_LCF1000_250GEV_LIN_GLOB_aggressive", ]

runs_glob_quad = ["260921_jth_HLLHC_250GEV_QUAD_GLOB_aggressive",
                  "260921_jth_LEP3_250GEV_QUAD_GLOB_aggressive",
                  "260921_jth_FCCee_250GEV_QUAD_GLOB_aggressive",
                  "260921_jth_LCF500_250GEV_QUAD_GLOB_aggressive",
                  "260921_jth_LCF1000_250GEV_QUAD_GLOB_aggressive"]

runs_all = runs_ind_lin + runs_ind_quad + runs_glob_lin + runs_glob_quad
vSM = 0.24622
mH = 0.125
corr_factor_Op = -2.0 * (vSM ** 4) / (mH ** 2)
corr_factor_OpBox = 3.0 * vSM ** 2
corr_factor_OpD = -(3.0 / 4.0) * vSM ** 2

posterior_dict = {}


def find_bounds(runs, hdi=True, th_scenario="aggressive"):
    q_low = 16.
    q_high = 84.

    bounds_dict = {}
    for run in runs:

        collider = run.split("_")[2]
        eft_order = run.split("_")[4]
        ind_or_glob = run.split("_")[5]
        idx = (collider, eft_order, ind_or_glob)

        with open(result_dir / run.replace("aggressive", th_scenario) / "fit_results.json", "r") as f:
            fit_results = json.load(f)

        samples_op = np.array(fit_results["samples"]["Op"])
        individual = True if "IND" in run else False

        if not individual:
            samples_OpBox = np.array(fit_results["samples"]["OpBox"])
            samples_OpD = np.array(fit_results["samples"]["OpD"])
            samples_kappa = corr_factor_Op * samples_op + corr_factor_OpBox * samples_OpBox + corr_factor_OpD * samples_OpD
        else:
            samples_kappa = corr_factor_Op * samples_op

        if not hdi:
            low_Op, high_Op = np.percentile(samples_op, [q_low, q_high])
            low_kappa, high_kappa = np.percentile(samples_kappa, [q_low, q_high])
        else:
            # if "HLLHC_250GEV_QUAD" in run:
            #     low_Op, high_Op = az.hdi(samples_op, hdi_prob=.68, multimodal=True)[0]
            #     low_kappa, high_kappa = az.hdi(samples_kappa, hdi_prob=.68, multimodal=True)[0]
            # else:
            low_Op, high_Op = az.hdi(samples_op, hdi_prob=.68, multimodal=False)
            low_kappa, high_kappa = az.hdi(samples_kappa, hdi_prob=.68, multimodal=False)

            low_Op = float(low_Op)
            high_Op = float(high_Op)
            low_kappa = float(low_kappa)
            high_kappa = float(high_kappa)
            half_width_kappa = (high_kappa - low_kappa) / 2

        # symmetrise linear bounds
        if "LIN" in run:
            half_width_Op = (high_Op - low_Op) / 2
            low_Op = - half_width_Op
            high_Op = half_width_Op

            low_kappa = - half_width_kappa
            high_kappa = half_width_kappa

        # round everything to 2 decimals using string formatting

        bounds_dict[idx] = [low_Op, high_Op, low_kappa, high_kappa, half_width_kappa]
        posterior_dict[idx] = samples_kappa

    bounds_df = pd.DataFrame(bounds_dict)
    bounds_df = bounds_df.T
    bounds_df.columns = ["low_Op", "high_Op", "low_kappa", "high_kappa", "half_width_kappa"]
    return bounds_df


if compute_bounds:
    bounds_trilinear_aggressive = find_bounds(runs_all, hdi=True, th_scenario="aggressive")
    bounds_trilinear_conservative = find_bounds(runs_all, hdi=True, th_scenario="conservative")

    # save to csv
    bounds_trilinear_aggressive.to_csv("trilinear_bounds_aggressive.csv")
    bounds_trilinear_conservative.to_csv("trilinear_bounds_conservative.csv")

    #
    # for fit_name, posterior in posterior_dict.items():
    #
    #     if "aggressive" in fit_name:
    #         fig, ax = plt.subplots()
    #
    #         ax.hist(posterior, bins="fd", alpha=0.4, label="aggressive", density=True)
    #         ax.hist(posterior_dict[(fit_name[0], fit_name[1], fit_name[2], "conservative")], bins="fd", alpha=0.4, label="conservative", density=True)
    #         ax.legend()
    #         fig.savefig("plots/kappa_posterior_{}.pdf".format("_".join(fit_name)))
    #


else:
    bounds_trilinear_aggressive = pd.read_csv(f"trilinear_bounds_aggressive.csv", index_col=[0, 1, 2])
    bounds_trilinear_conservative = pd.read_csv(f"trilinear_bounds_conservative.csv", index_col=[0, 1, 2])

import math


def _fmt_interval(width, sig_figs=2):
    """Format number with given significant figures."""
    if width == 0:
        return "$0$"

    exponent = math.floor(math.log10(abs(width)))
    decimals = sig_figs - 1 - exponent

    return f"${width:.{max(0, decimals)}f}$"


def make_latex_table_dkappa3(bounds_aggr: pd.DataFrame,
                             bounds_cons: pd.DataFrame,
                             ndp: int = 3,
                             collider_map=None,
                             fit_map=None,
                             table_caption=r"$\delta\kappa_3$ (68\% C.I.)",
                             table_label="tab:dkappa3_bounds"):
    """
    bounds_* must have:
      - MultiIndex rows: (Collider, Fit, Scenario) where Scenario in {"IND","GLOB"}
      - columns: low_kappa, high_kappa
    Produces LaTeX matching the screenshot layout using booktabs + multirow.
    """
    if collider_map is None:
        collider_map = {
            "HLLHC": "HL-LHC",
            "LEP3": "LEP3",
            "FCCee": "FCC-ee",
            "LCF500": "LCF550",  # change if you want LCF500 instead
            "LCF1000": "LCF1000",
        }
    if fit_map is None:
        fit_map = {"LIN": "Linear", "QUAD": "Quad."}

    # Ensure we can index like df.loc[(collider, fit, scen), ...]
    for df in (bounds_aggr, bounds_cons):
        if not isinstance(df.index, pd.MultiIndex) or df.index.nlevels != 3:
            raise ValueError("Expected a 3-level MultiIndex: (Collider, LIN/QUAD, IND/GLOB).")
        for col in ("low_kappa", "high_kappa"):
            if col not in df.columns:
                raise ValueError(f"Missing column {col} in dataframe.")

    colliders_order = ["HLLHC", "LEP3", "FCCee", "LCF500", "LCF1000"]
    fits_order = ["LIN", "QUAD"]

    # Build LaTeX
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\renewcommand{\arraystretch}{1.25}")
    lines.append(r"\setlength{\tabcolsep}{8pt}")
    lines.append(r"\begin{tabular}{llcccc}")
    lines.append(r"\toprule")
    # Header row 1 (caption in left block, Aggressive/Conservative groups)
    lines.append(
        r"\multicolumn{2}{l}{" + table_caption + r"}"
                                                 r" & \multicolumn{2}{c}{Aggressive}"
                                                 r" & \multicolumn{2}{c}{Conservative}\\"
    )
    # Header row 2 (subcolumns)
    lines.append(
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}"
        r" & & Individual & Marginalised & Individual & Marginalised\\"
    )
    lines.append(r"\midrule")

    for collider in colliders_order:
        collider_tex = collider_map.get(collider, collider)
        # Start multirow for each collider (2 rows: Linear + Quad)
        for j, fit in enumerate(fits_order):
            fit_tex = fit_map.get(fit, fit)

            # Pull intervals:
            # IND = Individual, GLOB = Marginalised in your dataframe

            a_ind = bounds_aggr.loc[(collider, fit, "IND"), ["half_width_kappa"]]
            a_glb = bounds_aggr.loc[(collider, fit, "GLOB"), ["half_width_kappa"]]
            c_ind = bounds_cons.loc[(collider, fit, "IND"), ["half_width_kappa"]]
            c_glb = bounds_cons.loc[(collider, fit, "GLOB"), ["half_width_kappa"]]

            a_ind_s = _fmt_interval(a_ind["half_width_kappa"])
            a_glb_s = _fmt_interval(a_glb["half_width_kappa"])
            c_ind_s = _fmt_interval(c_ind["half_width_kappa"])
            c_glb_s = _fmt_interval(c_glb["half_width_kappa"])

            if j == 0:
                left = rf"\multirow{{2}}{{*}}{{{collider_tex}}} & {fit_tex}"
            else:
                left = rf" & {fit_tex}"

            lines.append(
                left + rf" & {a_ind_s} & {a_glb_s} & {c_ind_s} & {c_glb_s}\\"
            )

        lines.append(r"\hline")

    lines[-1] = r"\bottomrule"  # replace last midrule with bottomrule
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{" + table_caption + r"}")
    lines.append(r"\label{" + table_label + r"}")
    lines.append(r"\end{table}")

    # Required LaTeX packages for this table
    preamble_hint = "\n".join([
        "% Preamble requirements:",
        r"% \usepackage{booktabs}",
        r"% \usepackage{multirow}",
    ])

    return preamble_hint + "\n\n" + "\n".join(lines)


latex_str = make_latex_table_dkappa3(bounds_trilinear_aggressive, bounds_trilinear_conservative)
with open("trilinear_bounds_table.tex", "w") as f:
    f.write(latex_str)

# PLOTTING


# Styling
colors = {
    ("LIN", "IND"): "#e41a1c",  # red
    ("LIN", "GLOB"): "#4daf4a",  # green
    ("QUAD", "IND"): "#ff7f00",  # orange
    ("QUAD", "GLOB"): "#377eb8"  # blue
}

labels = {
    ("LIN", "IND"): r"${\rm Ind.}, \mathcal{O}(\Lambda^{-2})$",
    ("LIN", "GLOB"): r"${\rm Marg.}, \mathcal{O}(\Lambda^{-2})$",
    ("QUAD", "IND"): r"${\rm Ind.}, \mathcal{O}(\Lambda^{-4})$",
    ("QUAD", "GLOB"): r"${\rm Marg.}, \mathcal{O}(\Lambda^{-4})$",
}

colliders = bounds_trilinear_aggressive.index.get_level_values(0).unique()

fig, ax = plt.subplots(figsize=(12, 10))

# vertical spacing
y_base = np.arange(len(colliders))[::-1]  # top to bottom
offsets = {
    ("LIN", "IND"): 0.25,
    ("LIN", "GLOB"): 0.08,
    ("QUAD", "IND"): -0.08,
    ("QUAD", "GLOB"): -0.25,
}

bar_height = 0.1

for i, collider in enumerate(colliders):
    for key in offsets:
        order, fit_type = key

        try:
            row_aggressive = bounds_trilinear_aggressive.loc[(collider, order, fit_type)]
            row_conservative = bounds_trilinear_conservative.loc[(collider, order, fit_type)]
        except KeyError:
            continue

        width_conservative = row_conservative["half_width_kappa"]
        width_aggressive = row_aggressive["half_width_kappa"]

        y = y_base[i] + offsets[key]

        ax.barh(
            y,
            width_aggressive,
            height=bar_height,
            color=colors[key],
            alpha=0.85,
            label=labels[key] if i == 0 else None
        )

        ax.barh(
            y,
            width_conservative,
            height=bar_height,
            color=colors[key],
            alpha=0.3,
            linewidth=0.2,
            edgecolor='k',
            label=None
        )

# Formatting
ax.set_yticks(y_base)
ax.set_xticks(np.arange(0, 0.51, 0.1))
ax.set_yticklabels([r"${\rm HL}\textnormal{-}{\rm LHC}$",
                    r"${\rm LEP3}$",
                    r"${\rm FCC}\textnormal{-}{\rm ee}$",
                    r"${\rm LCF550}$",
                    r"${\rm LCF1000}$", ])
# ax.set_xticklabels([f"{x:.1f}" for x in ax.get_xticks()])


ax.set_xlabel(r"$\delta \kappa_3$")

ax.grid(axis="x", which="major", linestyle=":", linewidth=0.8, alpha=0.6)
ax.xaxis.minorticks_on()
ax.grid(axis="x", which="minor", linestyle=":", linewidth=0.5, alpha=0.35)
ax.tick_params(
    axis='both',
    which='both',
    direction='in',
    top=True,
    right=True
)

# Clean legend (unique entries only)
handles, legend_labels = ax.get_legend_handles_labels()
unique = dict(zip(legend_labels, handles))
leg1 = ax.legend(unique.values(), unique.keys(), frameon=False, loc='lower right')
ax.set_xlim(0, 0.35)

# Custom legend elements
ci_handles = [
    Patch(facecolor='lightgray', edgecolor='black', label=r"${\rm Conservative}$"),
    Patch(facecolor='black', edgecolor='black', label=r"${\rm Aggressive}$")
]

# Add legend (position as needed)
leg2 = ax.legend(handles=ci_handles, loc="upper right", bbox_to_anchor=(0.975, 0.4), frameon=False)

ax.add_artist(leg1)
# add smefit logo to bottom right
logo = plt.imread("logo.png")
new_ax = fig.add_axes([0.12, 0.882, 0.2, 0.2], anchor='SE', zorder=10)
new_ax.imshow(logo)
new_ax.axis('off')

ax.set_title(r"${\rm 68\%\,C.I., \;}\mu_0=250\,{\rm GeV}$", y=1.02)

# plt.tight_layout()
plt.savefig("smefit_trilinear_bounds_agg_cons_bar_rebuttal.pdf")
