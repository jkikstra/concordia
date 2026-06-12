"""
Run CMIP7 fast-track workflow.

We use this to avoid having to run notebooks
by hand too often.
"""

from __future__ import annotations

from pathlib import Path
import tqdm.auto as tqdm

from concordia.cmip7.utils_papermill import run_notebook

def get_notebook_parameters(notebook_name: str,
                            marker_to_run: str, # options: H, HL, M, ML, L, LN, VL
                            # VERSIONING 
                            GRIDDING_VERSION: str,
                            # Files
                            SETTINGS_FILE: str = "config_cmip7_v0-4-0-EXT.yaml", 
                            #HISTORY_FILE: str = "country-history_202511261223_202511040855_202512032146_202512021030_7e32405ade790677a6022ff498395bff00d9792d.csv",
                            # WORKFLOW ELEMENTS: What elements of the workflow 
                            run_main: bool = True, # argument not currently a used
                            run_main_gridding: bool =True, # produce BC-*, ..., VOC-* .nc files (AIR, anthro, openburning)
                            run_anthro_timeseries_correction:bool = True,
                            run_AIR_anthro_timeseries_correction:bool = True,
                            run_openburning_timeseries_correction:bool = True,
                            run_2100_alignment_to_fasttrack:bool = True,
                            # SUPPLEMENTAL WORKFLOWS
                            run_openburning_h2:bool = True, # produce H2-em-openburning_*.nc, requires CO openburning to already have been run
                            run_anthro_supplemental_voc:bool = True, # produce VOC01, ..., VOC25 .nc files (anthro VOC speciation), requires VOC bulk to already have been run
                            run_openburning_supplemental_voc:bool = True, # produce C2H2, ..., Toluenelump .nc files (openburning VOC speciation), requires VOC bulk to already have been run
                            # Parameters
                            FADE_ANCHOR_YEAR:int = 2100,        # year where extension is forced to equal fast-track
                            FADE_CONVERGENCE_YEAR:int = 2150,   # year at which the additive correction has decayed to zero
                            DROP_ANCHOR_TIMESTEP:bool = True,   # drop the FADE_ANCHOR_YEAR (=2100) timestep from output
                            run_2100_alignment_diagnostic:bool = True,
                            # SPECIES: specify if you only want to run a selected set of emissions species
                            DO_GRIDDING_ONLY_FOR_THESE_SPECIES: list[str] | None = None,
                            # SECTORS: specify if you only want to run a selected set of sectors (anthro, openburning, AIR_anthro)
                            DO_GRIDDING_ONLY_FOR_THESE_SECTORS: list[str] | None = None,
                            # supplemental: VOC files to produce
                            # - anthro
                            DO_VOC_SPECIATION_ANTHRO_ONLY_FOR_THESE_SPECIES: list[str] | None = None, # e.g. ["VOC01_alcohols_em_speciated_VOC_anthro"]
                            # - openburning
                            DO_VOC_SPECIATION_OPENBURNING_ONLY_FOR_THESE_SPECIES: list[str] | None = None, # e.g. ["C10H16"]
                            # want to change the default filenamestructure? NOTE: leave unchanged unless you know what you're doing and passing e.g. {{name}} correctly
                            FILE_NAME_ENDING: str | None = None, # normally defined in the workflow itself
                            # overwrite existing files? (only main workflow, not supplemental)
                            SKIP_EXISTING_MAIN_WORKFLOW_FILES: bool = True
                            ) -> dict[str, str]:
    """
    Get parameters for a given notebook
    """
    res = {
            "GRIDDING_VERSION": GRIDDING_VERSION, # defaults to `marker_to_run`, but here you can give another option
            "marker_to_run": marker_to_run,
            "SETTINGS_FILE": SETTINGS_FILE,
            #"HISTORY_FILE": HISTORY_FILE,
            # WORKFLOW ELEMENTS: What elements of the workflow 
            "run_main": run_main, # currently does nothing
            "run_main_gridding": run_main_gridding,
            "run_anthro_timeseries_correction":run_anthro_timeseries_correction,
            "run_AIR_anthro_timeseries_correction" : run_AIR_anthro_timeseries_correction,
            "run_openburning_timeseries_correction": run_openburning_timeseries_correction,
            "run_anthro_supplemental_voc": run_anthro_supplemental_voc,
            "run_openburning_supplemental_voc": run_openburning_supplemental_voc,
            "run_2100_alignment_to_fasttrack":run_2100_alignment_to_fasttrack,
            # SUPPLEMENTAL WORKFLOWS
            "run_openburning_h2":run_openburning_h2,
            "run_openburning_timeseries_correction":run_openburning_timeseries_correction,
            "run_openburning_supplemental_voc":run_openburning_supplemental_voc,
            # Parameters 
            "FADE_ANCHOR_YEAR":FADE_ANCHOR_YEAR,
            "FADE_CONVERGENCE_YEAR":FADE_CONVERGENCE_YEAR,
            "DROP_ANCHOR_TIMESTEP":DROP_ANCHOR_TIMESTEP,
            "run_2100_alignment_diagnostic":run_2100_alignment_diagnostic,
            # "run_anthro_supplemental_solidbiofuel": run_anthro_supplemental_solidbiofuel, # not yet implemented, for the future
            "DO_GRIDDING_ONLY_FOR_THESE_SPECIES": DO_GRIDDING_ONLY_FOR_THESE_SPECIES,
            "DO_GRIDDING_ONLY_FOR_THESE_SECTORS": DO_GRIDDING_ONLY_FOR_THESE_SECTORS,
            "DO_VOC_SPECIATION_ANTHRO_ONLY_FOR_THESE_SPECIES":DO_VOC_SPECIATION_ANTHRO_ONLY_FOR_THESE_SPECIES,
            "DO_VOC_SPECIATION_OPENBURNING_ONLY_FOR_THESE_SPECIES":DO_VOC_SPECIATION_OPENBURNING_ONLY_FOR_THESE_SPECIES,
            "FILE_NAME_ENDING": FILE_NAME_ENDING,
            "SKIP_EXISTING_MAIN_WORKFLOW_FILES": SKIP_EXISTING_MAIN_WORKFLOW_FILES,
        }

    return res


def main():  # noqa: PLR0912
    """
    Run the cmip7-fast-track workflow(s) through calling all notebooks in order.
    """
    HERE = Path(__file__).parent.parent.parent
    print(f"HERE: {HERE}")
    DEFAULT_NOTEBOOKS_DIR = HERE / "notebooks" / "cmip7"
    RUN_NOTEBOOKS_DIR = HERE / "notebooks-papermill"

    notebooks_dir = DEFAULT_NOTEBOOKS_DIR
    all_notebooks = tuple(sorted(notebooks_dir.glob("*.py")))

    GRIDDING_VERSION_PREFIX = "run_20260608_" # appended with the marker
    
    DO_GRIDDING_ONLY_FOR_THESE_SPECIES = None # all species
    # DO_GRIDDING_ONLY_FOR_THESE_SPECIES = ["BC", "SO2", "NMVOC", "NMVOCbulk"] # test just one/some species

    DO_GRIDDING_ONLY_FOR_THESE_SECTORS = None # all: i.e, same as doing ['anthro', 'openburning', 'AIR_anthro']
    # DO_GRIDDING_ONLY_FOR_THESE_SECTORS = ["openburning", "anthro"] # test just one/some sectors

    SKIP_EXISTING_MAIN_WORKFLOW_FILES = True
    FILE_NAME_ENDING = None # specify in the workflow notebook file itself

    # All
    markers = [
        # "vl",
        # "ln",
        # "l",
        # "ml",
       #  "m",
       #  "h",
         "hl",
    ]
    # processing: run the notebook
    notebook_prefixes = [
        "workflow_cmip7-extensions"
    ]
    # # Skip this step
    # notebook_prefixes = []


    for marker in tqdm.tqdm(markers,
                            desc="Running full workflow"):
        
        GRIDDING_VERSION = f"{GRIDDING_VERSION_PREFIX}{marker}" # folder name of outputs in results folder

        for notebook in all_notebooks:
            if any(notebook.name.startswith(np) for np in notebook_prefixes):

                print(f"RUNNING: {markers}")
                GRIDDING_VERSION = f"{marker}-ext_1-1-1"

                parameters = get_notebook_parameters(notebook.name,
                                                     
                                                     # SCENARIO: Which marker to run
                                                     marker_to_run=marker,
                                                     
                                                     # WORKFLOW ELEMENTS: What elements of the workflow 
                                                     run_main=True, # argument not currently a used
                                                     run_main_gridding=True, # produce BC-*, ..., VOC-* .nc files (AIR, anthro, openburning)
                                                     run_anthro_timeseries_correction = True,
                                                     run_AIR_anthro_timeseries_correction = True,
                                                     run_openburning_timeseries_correction = True,
                                                     run_2100_alignment_to_fasttrack = True,
                                                     FADE_ANCHOR_YEAR = 2100,        # year where extension is forced to equal fast-track
                                                     FADE_CONVERGENCE_YEAR = 2150,   # year at which the additive correction has decayed to zero
                                                     DROP_ANCHOR_TIMESTEP = True,   # drop the FADE_ANCHOR_YEAR (=2100) timestep from output
                                                     run_2100_alignment_diagnostic = True,

                                                     # SUPPLEMENTAL WORKFLOWS
                                                     run_openburning_h2 = True, # produce H2-em-openburning_*.nc, requires CO openburning to already have been run
                                                     run_anthro_supplemental_voc = True, # produce VOC01, ..., VOC25 .nc files (anthro VOC speciation), requires VOC bulk to already have been run
                                                     run_openburning_supplemental_voc = True, # produce C2H2, ..., Toluenelump .nc files (openburning VOC speciation), requires VOC bulk to already have been run
                                                     
                                                     # VERSIONING 
                                                     GRIDDING_VERSION=GRIDDING_VERSION,
                                                     
                                                     # SPECIES: specify if you only want to run a selected set of emissions species
                                                     DO_GRIDDING_ONLY_FOR_THESE_SPECIES=DO_GRIDDING_ONLY_FOR_THESE_SPECIES,
                                                     # SECTORS: specify if you only want to run a selected set of sectors (anthro, openburning, AIR_anthro)
                                                     DO_GRIDDING_ONLY_FOR_THESE_SECTORS=DO_GRIDDING_ONLY_FOR_THESE_SECTORS,
                                                     # supplemental: VOC files to produce
                                                     # - anthro
                                                     DO_VOC_SPECIATION_ANTHRO_ONLY_FOR_THESE_SPECIES = None, # e.g. ["VOC01_alcohols_em_speciated_VOC_anthro"]
                                                     # - openburning
                                                     DO_VOC_SPECIATION_OPENBURNING_ONLY_FOR_THESE_SPECIES = None, # e.g. ["C10H16"]
                                                     
                                                     # overwrite existing files? (only main workflow, not supplemental)
                                                     SKIP_EXISTING_MAIN_WORKFLOW_FILES=SKIP_EXISTING_MAIN_WORKFLOW_FILES,

                                                     # want to change the default filenamestructure? NOTE: leave unchanged unless you know what you're doing and passing e.g. {{name}} correctly
                                                     FILE_NAME_ENDING=FILE_NAME_ENDING,
                                                     
                                                     #  ... add here other parameters that you might like to change
                                                     SETTINGS_FILE = "config_cmip7_v0-4-0-EXT.yaml",
                                                     )

                # how to identify this run in the papermill notebook save folder
                if GRIDDING_VERSION is None:
                    notebook_identification = f"{marker}"
                else:
                    notebook_identification = f"{GRIDDING_VERSION}"

                print(notebook.name)

                run_notebook(notebook=notebook,
                             run_notebooks_dir=RUN_NOTEBOOKS_DIR,
                             parameters=parameters,
                             idn=notebook_identification
                             )

    


if __name__ == "__main__":
    main()
