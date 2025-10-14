# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from pathlib import Path

import concordia.cmip7.utils_futureproxy_ceds_bb4cmip as uprox

# %% 
# set up 
from concordia.cmip7.CONSTANTS import CONFIG, return_marker_information
from concordia.settings import Settings

FIXED_METADATA = True

# TODO: 
# - make the marker a parameter
GRIDDING_VERSION, MODEL_SELECTION, SCENARIO_SELECTION, SCENARIO_SELECTION_GRIDDED_AFTER_METADATA = return_marker_information(
    m="H", fixed_metadata=FIXED_METADATA
)

VERSION = CONFIG
try:
    # when running the script from a terminal or otherwise
    cmip7_dir = Path(__file__).resolve()
    settings = uprox.get_settings(base_path=cmip7_dir, file = CONFIG)
except (FileNotFoundError, NameError):
    try:
        # when running the script from a terminal or otherwise
        cmip7_dir = Path(__file__).resolve().parent
        settings = uprox.get_settings(base_path=cmip7_dir, file = CONFIG)
    except (FileNotFoundError, NameError):
        try:
            # Fallback for interactive/Jupyter mode, where 'file location' does not exist
            cmip7_dir = Path().resolve()  # one up
            settings = uprox.get_settings(base_path=cmip7_dir, file = CONFIG)
        except (FileNotFoundError, NameError):
            # if Path().resolve somehow goes to the root of this repository
            cmip7_dir = Path().resolve() / "notebooks" / "cmip7"  # one up
            settings = uprox.get_settings(base_path=cmip7_dir, file = CONFIG)

# %%
# locate files to upload
tree_root = settings.out_path / GRIDDING_VERSION / "final"

list(tree_root.rglob("*.nc"))
nc_files = list(tree_root.rglob("*.nc"))

nc_files_main = [f for f in nc_files if "speciated" not in f.name]
nc_files_supplemental = [f for f in nc_files if "speciated" in f.name]


# %%
# try to follow the docs: 
# point --cv-source to --cv-source https://raw.githubusercontent.com/jkikstra/input4MIPs_CVs/source_id_scenariomip/CVs/

# try uploading with:
# !input4mips-validation --logging-level DEBUG upload-ftp . --password "kikstra@iiasa.ac.at" --cv-source https://raw.githubusercontent.com/jkikstra/input4MIPs_CVs/source_id_scenariomip/CVs/ --ftp-dir-rel-to-root "IIASA-IAMC" --n-threads 10 --dry-run
# ERROR (in powershell):
#  !input4mips-validation --logging-level DEBUG upload-ftp . --password "kikstra@iiasa.ac.at" --cv-source https://raw.githubusercontent.com/jkikstra/input4MIPs_CVs/source_id_scenariomip/CVs/ --ftp-dir-rel-to-root "IIASA-IAMC" --n-threads 10 --dry-run
# !input4mips-validation : The term '!input4mips-validation' is not recognized as the name of a cmdlet,
# function, script file, or operable program. Check the spelling of the name, or if a path was included,
# verify that the path is correct and try again.
# At line:1 char:1
# + !input4mips-validation --logging-level DEBUG upload-ftp . --password  ...
# + ~~~~~~~~~~~~~~~~~~~~~~
#     + CategoryInfo          : ObjectNotFound: (!input4mips-validation:String) [], CommandNotFoundExcep
#    tion
#     + FullyQualifiedErrorId : CommandNotFoundException
# also tried, with same effect:
# !input4mips-validation upload-ftp "ftp.llnl.gov" --username "anonymous" --password "kikstra@iiasa.ac.at" --cv-source https://raw.githubusercontent.com/jkikstra/input4MIPs_CVs/source_id_scenariomip/CVs/ --ftp-dir-rel-to-root "incoming" --n-threads 10 --dry-run


# %%
# (only if necessary)
# # batch rename all files, renaming "IIASA" in file.name with "IIASA-IAMC" 
# 
# renamed_count = 0

# for file_path in nc_files:
#     if "IIASA" in file_path.name:
#         # Create new filename by replacing "IIASA" with "IIASA-IAMC"
#         new_name = file_path.name.replace("IIASA", "IIASA-IAMC")
#         new_path = file_path.parent / new_name
        
#         # Rename the file
#         try:
#             file_path.rename(new_path)
#             print(f"Renamed: {file_path.name} -> {new_name}")
#             renamed_count += 1
#         except FileExistsError:
#             print(f"Warning: {new_name} already exists, skipping {file_path.name}")
#         except Exception as e:
#             print(f"Error renaming {file_path.name}: {e}")

# print(f"\nTotal files renamed: {renamed_count}")

# %%
