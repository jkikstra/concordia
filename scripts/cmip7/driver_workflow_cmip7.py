"""
Run CMIP7 fast-track/extensions workflow.

We use this to avoid having to run notebooks
by hand too often.

Structure: the "To edit" block below holds every value you change by hand. The
dataclasses hold only names and types -- they are the contract against each
notebook's first `parameters` cell, and deliberately carry no defaults, so a
forgotten parameter is a loud TypeError rather than a silent fallback.
`build_parameters` maps the one onto the other.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any
import tqdm.auto as tqdm

from concordia.cmip7.utils_papermill import run_notebook

####### To edit - Shared ##################
WORKFLOW: str = "fast-track"                # "fast-track" or "extensions"
MARKERS: list[str] = ["h"]                  # lowercase; all: ["vl","ln","l","ml","m","h","hl"]
VERSION_ESGF: str = "1-1-1"
SETTINGS_FILES: dict[str, str] = {
    "fast-track": "config_cmip7_v0-4-0.yaml",
    "extensions": "config_cmip7_v0-4-0-EXT.yaml",
}
run_main: bool = True
run_main_gridding: bool = True               # produce BC-*, ..., VOC-* .nc files (AIR, anthro, openburning)
run_openburning_h2: bool = False              # produce H2-em-openburning_*.nc, requires CO openburning to already have been run
run_anthro_supplemental_voc: bool = False     # produce VOC01, ..., VOC25 .nc files (anthro VOC speciation), requires VOC bulk to already have been run
run_openburning_supplemental_voc: bool = False # produce C2H2, ..., Toluenelump .nc files (openburning VOC speciation), requires VOC bulk to already have been run
run_anthro_timeseries_correction: bool = False
run_AIR_anthro_timeseries_correction: bool = True
run_openburning_timeseries_correction: bool = False
SKIP_EXISTING_MAIN_WORKFLOW_FILES: bool = True # overwrite existing files? (only main workflow, not supplemental)
# main: files to produce
DO_GRIDDING_ONLY_FOR_THESE_SPECIES: list[str] | None = ["SO2"] # all, or e.g. ["BC", "SO2", "NMVOC", "NMVOCbulk"]
DO_GRIDDING_ONLY_FOR_THESE_SECTORS: list[str] | None = ['AIR_anthro'] # all, or a selection of ['anthro', 'openburning', 'AIR_anthro']
# supplemental: VOC files to produce
# - anthro
DO_VOC_SPECIATION_ANTHRO_ONLY_FOR_THESE_SPECIES: list[str] | None = None # e.g. ["VOC01_alcohols_em_speciated_VOC_anthro"]
# - openburning
DO_VOC_SPECIATION_OPENBURNING_ONLY_FOR_THESE_SPECIES: list[str] | None = None # e.g. ["C10H16"]
####### To edit - Fast-track ##################
HISTORY_FILE: str = "country-history_202511261223_202511040855_202512032146_202512021030_7e32405ade790677a6022ff498395bff00d9792d.csv"
run_spatial_harmonisation: bool = True       # spatial harmonisation with CEDS anthro in 2023 (needs raw CEDS files locally)
####### To edit - Extensions ##################
EXTENSIONS_INPUT_PATH: str = "/home/zecchetto/ECE-climate/extensions/input"
EXTENSIONS_LOCATION_DOWNSCALED: str = EXTENSIONS_INPUT_PATH + "/all_downscaled_markers_1-1-1"
EXTENSIONS_LOCATION_CMIP7_HISTORY: str = EXTENSIONS_INPUT_PATH + "/workflow_history_files"
# Fast-track gridded outputs (already-final files ending at 2100-12), used to anchor the
# extension at 2100. Must be the *parent* of the "{marker}_{VERSION_ESGF}" folders, i.e. it
# must match the fast-track config's out_path.
EXTENSIONS_LOCATION_FASTTRACK_GRIDDED: str = "/home/zecchetto/ECE-climate/extensions"
run_2100_alignment_to_fasttrack: bool = True
run_2100_alignment_diagnostic: bool = True
FADE_ANCHOR_YEAR: int = 2100                 # year where extension is forced to equal fast-track
FADE_CONVERGENCE_YEAR: int = 2150            # year at which the additive correction has decayed to zero
DROP_ANCHOR_TIMESTEP: bool = True            # drop the FADE_ANCHOR_YEAR (=2100) timestep from output
###### To edit - Note ###########
# Every editable value lives in the blocks above. The dataclasses below carry no
# defaults on purpose: they only declare the names and types the notebooks accept.
##################################

HERE = Path(__file__).resolve().parent.parent.parent
NOTEBOOKS_DIR = HERE / "notebooks" / "cmip7"
RUN_NOTEBOOKS_DIR = HERE / "notebooks-papermill"

NOTEBOOK = {
 "fast-track": "workflow_cmip7-fast-track",
 "extensions": "workflow_cmip7-extensions",
}


@dataclass
class SharedParameters:
    """Declared by the first `parameters` cell of BOTH notebooks (17 names)."""

    marker_to_run: str
    SETTINGS_FILE: str
    GRIDDING_VERSION: str | None
    VERSION_ESGF: str
    run_main: bool
    run_main_gridding: bool
    run_openburning_h2: bool
    run_anthro_supplemental_voc: bool
    run_openburning_supplemental_voc: bool
    run_anthro_timeseries_correction: bool
    run_AIR_anthro_timeseries_correction: bool
    run_openburning_timeseries_correction: bool
    DO_GRIDDING_ONLY_FOR_THESE_SPECIES: list[str] | None
    DO_GRIDDING_ONLY_FOR_THESE_SECTORS: list[str] | None
    DO_VOC_SPECIATION_ANTHRO_ONLY_FOR_THESE_SPECIES: list[str] | None
    DO_VOC_SPECIATION_OPENBURNING_ONLY_FOR_THESE_SPECIES: list[str] | None
    SKIP_EXISTING_MAIN_WORKFLOW_FILES: bool


@dataclass
class FastTrackParameters:
    """Declared by workflow_cmip7-fast-track.py only (2 names)."""

    HISTORY_FILE: str
    run_spatial_harmonisation: bool


@dataclass
class ExtensionsParameters:
    """Declared by workflow_cmip7-extensions.py only (8 names).

    The LOCATION_* names must match the notebook exactly -- they are not the
    EXTENSIONS_* constants above, which only supply their values.
    """

    LOCATION_DOWNSCALED: str
    LOCATION_CMIP7_HISTORY: str
    LOCATION_FASTTRACK_GRIDDED: str
    run_2100_alignment_to_fasttrack: bool
    run_2100_alignment_diagnostic: bool
    FADE_ANCHOR_YEAR: int
    FADE_CONVERGENCE_YEAR: int
    DROP_ANCHOR_TIMESTEP: bool


def build_parameters(workflow: str, marker: str, gridding_version: str) -> dict[str, Any]:
    """Map the editable constants above onto the notebook parameter payload."""
    shared = SharedParameters(
        marker_to_run=marker,
        SETTINGS_FILE=SETTINGS_FILES[workflow],
        GRIDDING_VERSION=gridding_version,
        VERSION_ESGF=VERSION_ESGF,
        run_main=run_main,
        run_main_gridding=run_main_gridding,
        run_openburning_h2=run_openburning_h2,
        run_anthro_supplemental_voc=run_anthro_supplemental_voc,
        run_openburning_supplemental_voc=run_openburning_supplemental_voc,
        run_anthro_timeseries_correction=run_anthro_timeseries_correction,
        run_AIR_anthro_timeseries_correction=run_AIR_anthro_timeseries_correction,
        run_openburning_timeseries_correction=run_openburning_timeseries_correction,
        DO_GRIDDING_ONLY_FOR_THESE_SPECIES=DO_GRIDDING_ONLY_FOR_THESE_SPECIES,
        DO_GRIDDING_ONLY_FOR_THESE_SECTORS=DO_GRIDDING_ONLY_FOR_THESE_SECTORS,
        DO_VOC_SPECIATION_ANTHRO_ONLY_FOR_THESE_SPECIES=DO_VOC_SPECIATION_ANTHRO_ONLY_FOR_THESE_SPECIES,
        DO_VOC_SPECIATION_OPENBURNING_ONLY_FOR_THESE_SPECIES=DO_VOC_SPECIATION_OPENBURNING_ONLY_FOR_THESE_SPECIES,
        SKIP_EXISTING_MAIN_WORKFLOW_FILES=SKIP_EXISTING_MAIN_WORKFLOW_FILES,
    )

    if workflow == "fast-track":
        specific = FastTrackParameters(
            HISTORY_FILE=HISTORY_FILE,
            run_spatial_harmonisation=run_spatial_harmonisation,
        )
    else:
        specific = ExtensionsParameters(
            LOCATION_DOWNSCALED=EXTENSIONS_LOCATION_DOWNSCALED,
            LOCATION_CMIP7_HISTORY=EXTENSIONS_LOCATION_CMIP7_HISTORY,
            LOCATION_FASTTRACK_GRIDDED=EXTENSIONS_LOCATION_FASTTRACK_GRIDDED,
            run_2100_alignment_to_fasttrack=run_2100_alignment_to_fasttrack,
            run_2100_alignment_diagnostic=run_2100_alignment_diagnostic,
            FADE_ANCHOR_YEAR=FADE_ANCHOR_YEAR,
            FADE_CONVERGENCE_YEAR=FADE_CONVERGENCE_YEAR,
            DROP_ANCHOR_TIMESTEP=DROP_ANCHOR_TIMESTEP,
        )

    return asdict(shared) | asdict(specific)


def main():
    for marker in tqdm.tqdm(MARKERS, desc=f"Running {WORKFLOW} workflow"):
        print(f"Gridding the {marker} scenario")
        # derived per marker, so a multi-marker run cannot collide on one output folder
        gridding_version = f"{marker}_{VERSION_ESGF}" if WORKFLOW == "fast-track" else f"{marker}-ext_{VERSION_ESGF}"
        parameters = build_parameters(workflow=WORKFLOW, marker=marker, gridding_version=gridding_version)

        run_notebook(
                run_notebooks_dir=RUN_NOTEBOOKS_DIR,
                notebook=NOTEBOOKS_DIR / f"{NOTEBOOK[WORKFLOW]}.py",
                parameters=parameters,
                idn=gridding_version,
                )


if __name__ == "__main__":
    main()
