# -*- coding: utf-8 -*-
import pathlib
import shutil
import subprocess
import sys
import os
import yaml
from matplotlib import rc, use

from ..log import logging
from .html_utils import dump_html_index
from .report import Report

_logger = logging.getLogger(__name__)


# global mathplotlib settings
use("PDF")
rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text", **{"usetex": True, "latex.preamble": r"\usepackage{amssymb}"})


def run_report(report_card_file):
    """
    Run the analysis given a report card name

    Parameters
    ----------
        report_card_file : pathlib:Path
            report configuration dictionary name
    """
    with open(report_card_file, encoding="utf-8") as f:
        report_config = yaml.safe_load(f)

    _logger.info(f"Analyzing : {report_config['result_IDs']}")

    report_name = report_config["name"]
    report_path = pathlib.Path(report_config["report_path"]).absolute()
    report_folder = report_path.joinpath(f"{report_name}")

    # Clean output folder if exists
    try:
        shutil.rmtree(report_folder)
    except FileNotFoundError:
        pass
    report_folder.mkdir(exist_ok=True, parents=True)

    # Initialize ANALYZE class
    report = Report(report_path, report_config["result_path"], report_config)

    # Things to include in report
    if report_config["summary"]:
        report.summary()

    if "chi2_plots" in report_config:
        report.chi2(**report_config["chi2_plots"])

    if "coefficients_plots" in report_config:
        report.coefficients(**report_config["coefficients_plots"])

    if "correlations" in report_config:
        report.correlations(**report_config["correlations"])

    if "PCA" in report_config:
        report.pca(**report_config["PCA"])

    if "fisher" in report_config:
        report.fisher(**report_config["fisher"])

    # Move all files to a meta folder
    meta_path = pathlib.Path(f"{report_folder}/meta").absolute()
    meta_path.mkdir()
    for f in report_folder.glob("*.*"):
        shutil.move(str(f), meta_path)

    # Combine PDF files together into raw pdf report
    pdf_files = sorted(meta_path.glob("*.pdf"), key=os.path.getmtime)
    gs_cmd = "gswin64c" if sys.platform == "win32" else "gs"
    subprocess.call([
        gs_cmd, "-q", "-dNOPAUSE", "-dBATCH", "-sDEVICE=pdfwrite",
        f"-sOutputFile={report_folder}/report_{report_name}.pdf",
        *[str(f) for f in pdf_files],
        ])

    # dump html index
    dump_html_index(
        report.html_content, report.html_index, report_folder, report_config["title"]
    )
