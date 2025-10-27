from __future__ import annotations
from pathlib import Path
import jupytext
from typing import Any

import papermill as pm

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