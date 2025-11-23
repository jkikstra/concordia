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
                            GRIDDING_VERSION: str | None = None, # defaults to `marker_to_run`, but here you can give another option
                            SETTINGS_FILE: str = "config_cmip7_v0-4-0.yaml", # preparing for first upload to ESGF 
                            run_main: bool = True,
                            run_main_gridding: bool = True, # if false, we'll stop at only running the downscaling of main
                            run_anthro_supplemental_voc: bool = False,
                            run_openburning_supplemental_voc: bool = False,
                            # run_anthro_supplemental_solidbiofuel: bool = False, # not yet implemented, for the future
                            HISTORY_FILE: str = "cmip7_history_countrylevel_251024.csv",
                            DO_GRIDDING_ONLY_FOR_THESE_SPECIES: list[str] | None = None
                            ) -> dict[str, str]:
    """
    Get parameters for a given notebook
    """
    if notebook_name == "workflow_cmip7-fast-track.py":
        res = {
            "GRIDDING_VERSION": GRIDDING_VERSION, # defaults to `marker_to_run`, but here you can give another option
            "marker_to_run": marker_to_run,
            "SETTINGS_FILE": SETTINGS_FILE,
            "run_main": run_main, # currently does nothing
            "run_main_gridding": run_main_gridding,
            "run_anthro_supplemental_voc": run_anthro_supplemental_voc,
            "run_openburning_supplemental_voc": run_openburning_supplemental_voc,
            # "run_anthro_supplemental_solidbiofuel": run_anthro_supplemental_solidbiofuel, # not yet implemented, for the future
            "HISTORY_FILE": HISTORY_FILE,
            "DO_GRIDDING_ONLY_FOR_THESE_SPECIES": DO_GRIDDING_ONLY_FOR_THESE_SPECIES
        }
    
    # if notebook_name == "check_plot-global-total-timeseries.py":
    #     res = {
    #         "marker_to_run": marker,
    #         "path_output_plot_results": output_path,
    #         "filetypes": filetypes,
    #         "species": species
    #     }

    else:
        raise NotImplementedError(notebook_name)

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

    GRIDDING_VERSION_PREFIX = "prehandover_test_"
    
    DO_GRIDDING_ONLY_FOR_THESE_SPECIES = None # all species
    # DO_GRIDDING_ONLY_FOR_THESE_SPECIES = ["CO2"] # test just one species

    # All
    markers = [
        # "VL",
        # "LN",
        # "L",
        # "ML",
        # "M",
        "H",
        # "HL",
    ]
    # # High priority markers:
    # markers = [
    #     "VL",
    #     "H"
    # ]


    # -----------------
    # 0. CREATE PROXIES
    # -----------------

    # tbd.
    # creates the inputs that are placed under "proxy_path"
    # Should not need to be run every time.

    # -----------------
    # 1A. MAIN WORKFLOW
    # -----------------

    # processing: run the notebook
    notebook_prefixes = [
        "workflow_cmip7-fast-track"
    ]
    # # Skip this step
    # notebook_prefixes = []


    for marker in tqdm.tqdm(markers,
                            desc="Running full workflow"):
        
        GRIDDING_VERSION = f"{GRIDDING_VERSION_PREFIX}{marker}" # folder name of outputs in results folder

        for notebook in all_notebooks:
            if any(notebook.name.startswith(np) for np in notebook_prefixes):

                parameters = get_notebook_parameters(notebook.name,
                                                     
                                                     # SCENARIO: Which marker to run
                                                     marker_to_run=marker,
                                                     
                                                     # WORKFLOW ELEMENTS: What elements of the workflow 
                                                     run_main=True, # argument not currently a used
                                                     run_main_gridding=True, # produce BC_*, ..., VOC_* .nc files (AIR, anthro, openburning)
                                                     run_anthro_supplemental_voc=True, # produce VOC01, ..., VOC25 .nc files (anthro VOC speciation)
                                                     run_openburning_supplemental_voc=True, # produce C2H2, ..., Toluenelump .nc files (openburning VOC speciation)
                                                     
                                                     # VERSIONING 
                                                     GRIDDING_VERSION=GRIDDING_VERSION,
                                                     
                                                     # SPECIES: specify if you only want to run a selected set of emissions species
                                                     DO_GRIDDING_ONLY_FOR_THESE_SPECIES=DO_GRIDDING_ONLY_FOR_THESE_SPECIES
                                                     
                                                     #  ... add here other parameters that you might like to change
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

    # --------------------------------------------------------------
    # 1B. SUPPLEMENTAL WORKFLOW (where not covered in MAIN WORKFLOW)
    # --------------------------------------------------------------

    # tbd.
    # currently, nothing necessary here.
    

    # --------------------------------
    # 2A. POST-PROCESSING (data fixes)
    # --------------------------------

    # tbd.

    # ------------------------------------
    # 2B. POST-PROCESSING (metadata fixes)
    # ------------------------------------

    # tbd.

    # ------------------------------------
    # 3. PLOTTING
    # ------------------------------------

    # tbd.



    # TODO: run checks&plots automatically as well.
    # - [ ] add parameters to the check notebooks
    # - [ ] ensure no bits of the check notebooks should not be run
    # - [ ] add those notebooks to this runner

    


if __name__ == "__main__":
    main()
