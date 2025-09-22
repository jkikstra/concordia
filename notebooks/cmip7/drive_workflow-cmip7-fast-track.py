"""
Run CMIP7 fast-track workflow. 

We use this to avoid having to run every marker
by hand.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import jupytext
import papermill as pm
import tqdm.auto as tqdm


def get_notebook_parameters(notebook_name: str, marker: str) -> dict[str, str]:
    """
    Get parameters for a given notebook
    """
    if notebook_name == "workflow-cmip7-fast-track.py":
        res = {"marker_to_run": marker}

    else:
        raise NotImplementedError(notebook_name)

    return res


def run_notebook(notebook: Path, run_notebooks_dir: Path, parameters: dict[str, Any], idn: str) -> None:
    """
    Run a notebook
    """
    notebook_jupytext = jupytext.read(notebook)

    # Write the .py file as .ipynb
    in_notebook = run_notebooks_dir / f"{notebook.stem}_{idn}_unexecuted.ipynb"
    in_notebook.parent.mkdir(exist_ok=True, parents=True)
    jupytext.write(notebook_jupytext, in_notebook, fmt="ipynb")

    output_notebook = run_notebooks_dir / f"{notebook.stem}_{idn}.ipynb"
    output_notebook.parent.mkdir(exist_ok=True, parents=True)

    print(f"Executing {notebook.name=} with {parameters=} from {in_notebook=}. " f"Writing to {output_notebook=}")
    # Execute to specific directory
    pm.execute_notebook(in_notebook, output_notebook, parameters=parameters)


def run_notebook_marker(notebook: Path, run_notebooks_dir: Path, marker: str) -> None:
    """
    Run a notebook that only needs marker information
    """
    parameters = get_notebook_parameters(notebook.name, marker=marker)

    run_notebook(notebook=notebook, run_notebooks_dir=run_notebooks_dir, parameters=parameters, idn=marker)


def main():  # noqa: PLR0912
    """
    Run the 500x series of notebooks
    """
    HERE = Path(__file__).parent
    DEFAULT_NOTEBOOKS_DIR = HERE
    RUN_NOTEBOOKS_DIR = HERE / "notebooks-papermill"

    notebooks_dir = DEFAULT_NOTEBOOKS_DIR
    all_notebooks = tuple(sorted(notebooks_dir.glob("*.py")))

    # All
    # markers = [
    #     "VLLO",
    #     "VLHO",
    #     "L",
    #     "ML",
    #     "M",
    #     "H",
    #     "HL",
    # ]
    markers = [
        "VLLO",
        "H"
    ]

    # processing: run the norebook
    notebook_prefixes = [
        "workflow-cmip7-fast-track"
    ]
    # # Skip this step
    # notebook_prefixes = []

    for marker in tqdm.tqdm(markers, desc="Running full workflow"):
        for notebook in all_notebooks:
            if any(notebook.name.startswith(np) for np in notebook_prefixes):
                run_notebook_marker(
                    notebook=notebook,
                    run_notebooks_dir=RUN_NOTEBOOKS_DIR,
                    marker=marker,
                )

    # TODO: enable separate running of MAIN downscaling and SUPPLEMENTAL downscaling.
    # - [ ] add parameters on the running of main workflow vs supplemental (VOC speciation) workflow
    
    # TODO: run checks automatically as well.
    # - [ ] add parameters to the check notebooks
    # - [ ] ensure no bits of the check notebooks should not be run
    # - [ ] add those notebooks to this runner


if __name__ == "__main__":
    main()
