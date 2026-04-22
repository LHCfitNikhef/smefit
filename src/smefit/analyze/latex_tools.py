# -*- coding: utf-8 -*-
import os
import pathlib
import subprocess

from .html_utils import run_htlatex


def latex_packages():
    L = [
        r"\documentclass{article}",
        r"\usepackage{float}",
        r"\usepackage{amsmath}",
        r"\usepackage{amssymb}",
        r"\usepackage{booktabs}",
        r"\usepackage[a4paper]{geometry}",
        r"\usepackage{array}",
        r"\usepackage{hyperref}",
        r"\usepackage{xcolor}",
        r"\usepackage{multirow}",
        r"\usepackage{pdflscape}",
        r"\usepackage{longtable}",
        r"\allowdisplaybreaks",
        r"\newcolumntype{C}[1]{>{\centering\let\newline\\\arraybackslash\hspace{0pt}}m{#1}}",
        r"\usepackage{graphicx}",
        r"\usepackage{tabularx}",
        r"\geometry{verbose, tmargin = 1.5cm, bmargin = 1.5cm, lmargin = 1cm, rmargin = 1cm}",
    ]
    return L


def multicolum_table_header(fit_labels, ncolumn=2):
    """Append the multicolumn table header"""
    L = [r"\hline"]
    temp = " & "
    for label in fit_labels:
        temp += f" & \\multicolumn{{{ncolumn}}}{{c|}}{{{label}}}"
    temp += r" \\ \hline"
    L.append(temp)
    return L


def chi2table_header(L, fit_labels):
    L.append(r"\hline")
    temp = r" \multicolumn{2}{|c|}{} & SM"
    for label in fit_labels:
        temp += f"& {label}"
    temp += r"\\ \hline"
    L.append(temp)
    L.append(
        r"Process & $N_{\rm data}$ & $\chi^ 2/N_{\rm data}$"
        + r"& $\chi^ 2/N_{data}$" * len(fit_labels)
        + r"\\ \hline"
    )
    return L


def dump_to_tex(tex_file, L):
    """Dump a string to a tex file.

    Parameters
    ----------
    tex_file: pathlib.Path
        path to tex file
    L : list(str)
        latex lines
    """
    L.append(r"\end{document}")
    L = [l + "\n" for l in L]
    with open(tex_file, "w", encoding="utf-8") as file:
        file.writelines(L)


def run_pdflatex(report, filename):
    """Run pdflatex.

    Parameters
    ----------
    report: str
        report path
    filename : str
        file name
    """
    subprocess.call(
        f"pdflatex -halt-on-error -output-directory {report} {filename}.tex > {report}/pdflatex.log",
        shell=True,
    )
    for ext in ["*.log", "*.aux", "*.out"]:
        for f in report.glob(ext):
            os.remove(f)


def compile_tex(report, L, filename):
    """Compile tex file.

    Parameters
    ----------
    report: str
        report path
    L : list(str)
        latex lines
    filename : str
        file name
    """
    tex_file = pathlib.Path(f"{report}/{filename}.tex")
    dump_to_tex(tex_file, L)
    run_pdflatex(report, filename)
    run_htlatex(report, tex_file)
