import numpy as np
import arviz as az
import pathlib
import json
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.lines import Line2D
import pandas as pd

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 20})
rc('text', usetex=True)

compute_bounds = True

result_dir = pathlib.Path(
    "/data/theorie/jthoeve/physics_projects/new_smefit/results/descoped_fccee")

runs_ind_lin = [
    "260324_jth_FCCee_250GEV_LIN_IND_aggressive",
    "260702_jth_FCCee_no_top_250GEV_LIN_IND_aggressive",
    "260702_jth_FCCee_descoped_4IPs_staged_top_250GEV_LIN_IND_aggressive",
    "260702_jth_FCCee_descoped_4IPs_250GEV_LIN_IND_aggressive",
    "260702_jth_FCCee_descoped_2IPs_staged_top_250GEV_LIN_IND_aggressive",
    "260702_jth_FCCee_descoped_2IPs_250GEV_LIN_IND_aggressive",
    "260324_jth_LEP3_250GEV_LIN_IND_aggressive",
]

runs_glob_lin = [
    "260326_jth_FCCee_250GEV_LIN_GLOB_aggressive",
    "260702_jth_FCCee_no_top_250GEV_LIN_GLOB_aggressive",
    "260702_jth_FCCee_descoped_4IPs_staged_top_250GEV_LIN_GLOB_aggressive",
    "260702_jth_FCCee_descoped_4IPs_250GEV_LIN_GLOB_aggressive",
    "260702_jth_FCCee_descoped_2IPs_staged_top_250GEV_LIN_GLOB_aggressive",
    "260702_jth_FCCee_descoped_2IPs_250GEV_LIN_GLOB_aggressive",
    "260326_jth_LEP3_250GEV_LIN_GLOB_aggressive",
]

runs_all = runs_ind_lin + runs_glob_lin

vSM = 0.24622
mH = 0.125
corr_factor_Op = -2.0 * (vSM ** 4) / (mH ** 2)
corr_factor_OpBox = 3.0 * vSM ** 2
corr_factor_OpD = -(3.0 / 4.0) * vSM ** 2


def parse_run(run):
    """Extract (scope, eft_order, ind_or_glob, has_top) from a run directory name.

    Run names don't have a fixed number of underscore-separated tokens (the
    descoping scenarios add variable-length infixes like "descoped_4IPs" or
    "no_top"), so fields are identified by substring rather than position.
    """
    eft_order = "QUAD" if "QUAD" in run else "LIN"
    ind_or_glob = "GLOB" if "GLOB" in run else "IND"

    if "LEP3" in run:
        scope = "LEP3"
    elif "descoped_4IPs" in run:
        scope = "FCCee_4IPs"
    elif "descoped_2IPs" in run:
        scope = "FCCee_2IPs"
    else:
        scope = "FCCee"

    if "no_top" in run:
        has_top = False
    elif "descoped" in run and "staged_top" not in run:
        has_top = False
    else:
        has_top = True

    return scope, eft_order, ind_or_glob, has_top


def find_bounds(runs, hdi=True):

    q_low = 16.
    q_high = 84.

    bounds_dict = {}
    for run in runs:

        idx = parse_run(run)

        with open(result_dir / run / "fit_results.json", "r") as f:
            fit_results = json.load(f)

        samples_op = np.array(fit_results["samples"]["Op"])
        individual = idx[2] == "IND"

        samples_OpBox = np.array(fit_results["samples"]["OpBox"])
        samples_OpD = np.array(fit_results["samples"]["OpD"])

        if not individual:
            samples_kappa = corr_factor_Op * samples_op + corr_factor_OpBox * samples_OpBox + corr_factor_OpD * samples_OpD
        else:
            samples_kappa = corr_factor_Op * samples_op

        if not hdi:
            low_Op, high_Op = np.percentile(samples_op, [q_low, q_high])
            low_kappa, high_kappa = np.percentile(samples_kappa, [q_low, q_high])
        else:
            low_Op, high_Op = az.hdi(samples_op, hdi_prob=.68)
            low_kappa, high_kappa = az.hdi(samples_kappa, hdi_prob=.68)

            low_Op = float(low_Op)
            high_Op = float(high_Op)
            low_kappa = float(low_kappa)
            high_kappa = float(high_kappa)

        # symmetrise linear bounds
        half_width_Op = (high_Op - low_Op) / 2
        low_Op = - half_width_Op
        high_Op = half_width_Op

        half_width_kappa = (high_kappa - low_kappa) / 2
        low_kappa = - half_width_kappa
        high_kappa = half_width_kappa

        bounds_dict[idx] = [low_Op, high_Op, low_kappa, high_kappa]

    bounds_df = pd.DataFrame(bounds_dict)
    bounds_df = bounds_df.T
    bounds_df.columns = ["low_Op", "high_Op", "low_kappa", "high_kappa"]
    bounds_df.index.names = ["scope", "order", "fit", "has_top"]
    return bounds_df


def build_latex_bounds_table(df, output_path="trilinear_bounds_scoping_table.tex"):
    """Write a grouped LaTeX table of kappa bounds for staged-top/no-top scenarios."""
    scope_order = ["LEP3", "FCCee", "FCCee_4IPs", "FCCee_2IPs"]
    eft_order = ["LIN", "QUAD"]
    fit_order = ["IND", "GLOB"]

    scope_labels = {
        "LEP3": r"${\rm LEP3}$",
        "FCCee": r"${\rm FCC}\textnormal{-}{\rm ee}$",
        "FCCee_4IPs": r"${\rm FCC}\textnormal{-}{\rm ee}\;{\rm descoped\;4\,IPs}$",
        "FCCee_2IPs": r"${\rm FCC}\textnormal{-}{\rm ee}\;{\rm descoped\;2\,IPs}$",
    }
    eft_labels = {
        "LIN": r"$\mathcal{O}(\Lambda^{-2})$",
        "QUAD": r"$\mathcal{O}(\Lambda^{-4})$",
    }
    fit_labels = {
        "IND": r"${\rm Individual}$",
        "GLOB": r"${\rm Marginalised}$",
    }

    rows = []
    idx_rows = []
    for scope in scope_order:
        for order in eft_order:
            for fit in fit_order:
                idx_top = (scope, order, fit, True)
                idx_no_top = (scope, order, fit, False)

                has_top_row = idx_top in df.index
                has_no_top_row = idx_no_top in df.index
                if not has_top_row and not has_no_top_row:
                    continue

                def fmt(idx):
                    if idx not in df.index:
                        return "--"
                    row = df.loc[idx]
                    return rf"$[{row['low_kappa']:.3f},\,{row['high_kappa']:.3f}]$"

                idx_rows.append((scope_labels.get(scope, scope), eft_labels[order], fit_labels[fit]))
                rows.append({
                    r"Staged top": fmt(idx_top),
                    r"No top": fmt(idx_no_top),
                })

    table_df = pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_tuples(idx_rows, names=["Scope", "EFT order", "Fit"]),
    )

    latex_table = table_df.to_latex(
        escape=False,
        multirow=True,
        column_format="lllcc",
        caption=r"Bounds on $\delta\kappa_3$ grouped by FCC-ee descoping scenario and fit strategy.",
        label="tab:trilinear-bounds-scoping",
    )
    pathlib.Path(output_path).write_text(latex_table)
    return latex_table


if compute_bounds:
    bounds_trilinear_scoping = find_bounds(runs_all, hdi=True)

    # save to csv
    bounds_trilinear_scoping.to_csv("trilinear_bounds_scoping.csv")

else:
    bounds_trilinear_scoping = pd.read_csv(
        "trilinear_bounds_scoping.csv", index_col=[0, 1, 2, 3])


latex_table = build_latex_bounds_table(
    bounds_trilinear_scoping,
    output_path="trilinear_bounds_scoping_table.tex",
)


# Styling
colors = {
    ("LIN", "IND"): "#e41a1c",   # red
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

scope_order = ["LEP3", "FCCee", "FCCee_4IPs", "FCCee_2IPs"]
scope_labels = {
    "LEP3": r"${\rm LEP3}$",
    "FCCee": r"${\rm FCC}\textnormal{-}{\rm ee}$",
    "FCCee_4IPs": r"${\rm FCC}\textnormal{-}{\rm ee}\;{\rm descoped\;4\,IPs}$",
    "FCCee_2IPs": r"${\rm FCC}\textnormal{-}{\rm ee}\;{\rm descoped\;2\,IPs}$",
}
scopes = [s for s in scope_order if s in bounds_trilinear_scoping.index.get_level_values(0).unique()]

fig, ax = plt.subplots(figsize=(12, 8))
fig.subplots_adjust(left=0.22)

# vertical spacing
y_base = np.arange(len(scopes))[::-1]  # top to bottom
offsets = {
    ("LIN", "IND"): 0.12,
    ("LIN", "GLOB"): -0.12,
}

bar_height = 0.15

for i, scope in enumerate(scopes):
    for key in offsets:
        order, fit_type = key

        try:
            row_top = bounds_trilinear_scoping.loc[(scope, order, fit_type, True)]
        except KeyError:
            row_top = None

        try:
            row_no_top = bounds_trilinear_scoping.loc[(scope, order, fit_type, False)]
        except KeyError:
            row_no_top = None

        y = y_base[i] + offsets[key]

        if row_top is not None:
            ax.barh(
                y,
                row_top["high_kappa"],
                left=0,
                height=bar_height,
                color=colors[key],
                alpha=0.85,
                label=labels[key] if i == 0 else None  # avoid duplicate legend
            )

        if row_no_top is not None:
            ax.barh(
                y,
                row_no_top["high_kappa"],
                left=0,
                height=bar_height,
                color=colors[key],
                alpha=0.3,
                label=None  # avoid duplicate legend
            )


# Formatting
ax.set_yticks(y_base)
ax.set_xticks(np.arange(0, 0.31, 0.05))
ax.set_yticklabels([scope_labels[s] for s in scopes])
ax.set_ylim(y_base[-1] - 1.8, y_base[0] + 0.6)

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
leg1 = ax.legend(unique.values(), unique.keys(), frameon=False, loc='lower left')
ax.set_xlim(0, 0.3)


# Custom legend elements
top_handles = [
    Line2D([0], [0], color='lightgray', lw=3, label=r"${\rm No\;top\;run}$"),
    Line2D([0], [0], color='black', lw=3, label=r"${\rm Staged\;top\;run}$")
]

# Add legend (position as needed)
leg2 = ax.legend(handles=top_handles, loc='center right', bbox_to_anchor=(0.98, 0.2), frameon=False)
ax.add_artist(leg1)
# add smefit logo to bottom right
logo = plt.imread("logo.png")
new_ax = fig.add_axes([0.68, 0.12, 0.2, 0.2], anchor='SE', zorder=10)
new_ax.imshow(logo)
new_ax.axis('off')

ax.set_title(r"${\rm 68\%\,C.I., \;}\mu_0=250\,{\rm GeV}$")

plt.savefig("smefit_trilinear_bounds_scoping.pdf")
