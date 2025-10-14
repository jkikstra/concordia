
Make sure you're on the correct branch.
I.e., https://github.com/jkikstra/concordia/tree/rescue-update-2025 (very similar to `main` on https://github.com/IAMconsortium/concordia) 

Use conda/mamba to create a virtual environment.
See [instructions](https://github.com/IAMconsortium/concordia?tab=readme-ov-file#getting-started).

1. Create a config that points to where we have the IIASA Sharepoint data; i.e., $data_path should point to "RESCUE - WP 1/data_2025_10_14 - similar to 2024_09_16"
1. start jupyter lab from your conda concordia environment
1. Change the version to "RERUN-{DATE}" in `notebooks/workflow-rescue.py` 
1. Run `notebooks/workflow-rescue.py` (it is jupytext)
    1. For each scenario file, change it where `model = (... settings.scenario_path / "REMIND-MAgPIE-CEDS-RESCUE-Tier2-2025-09-15.csv" ...)`
1. Upload the files to a new folder under RESCUE - WP1, and share this (with PIK colleagues)