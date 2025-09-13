# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# # Download CEDS emissions files

# %% [markdown]
# ## Specify input scenario data and project settings

# %% [markdown]
# Specify which scenario file to read in

# %%
download = {
    # proxies final sectors
    "proxies": [
        "https://rcdtn1.pnl.gov/data/CEDS/Jarmo_files/non-point_source_proxy_final_sector/", # url
        "C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/concordia_cmip7_v0_2/input/gridding/ceds_input/Jarmo_files/non-point_source_proxy_final_sector" # local_save_folder
    ],
    # seasonality files
    "seasonality": [
        "https://rcdtn1.pnl.gov/data/CEDS/Jarmo_files/seasonality/", # url
        "C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/concordia_cmip7_v0_2/input/gridding/ceds_input/seasonality" # local_save_folder
    ],
    # proxies intermediary sectors
    "proxies-interemdiary": [
        "https://rcdtn1.pnl.gov/data/CEDS/Jarmo_files/non-point_source_proxy_intermediate_sector/", # url
        "C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/concordia_cmip7_v0_2/input/gridding/ceds_input/non-point_source_proxy_intermediate_sector" # local_save_folder
    ],
    # point-source
    # not sure whether we would ever need to also download and add 'point-source emissions
}
# To: KIKSTRA Jarmo <kikstra@iiasa.ac.at>; ssmith-pnnl.gov <ssmith@pnnl.gov>; Gidden, Matthew J <matthew.gidden@pnnl.gov>; Hoesly, Rachel M <rachel.hoesly@pnnl.gov>
# Cc: Zebedee Nicholls <zebedee.nicholls@climate-resource.com>; HOEGNER Annika <hoegner@iiasa.ac.at>; Jonas Hörsch <jonas.hoersch@climateanalytics.org>
# Subject: RE: Updating CEDS proxy data for creating CMIP7 scenarios in line with CEDS ESGF products

# Steve

# %%
import os
import requests
from bs4 import BeautifulSoup # needs to be installed (`pip install bs4`)
from urllib.parse import urljoin

def download_CEDS_proxy_data(
    URL,
    local_save_folder
):

    # Base URL of the directory
    base_url = URL
    download_dir = local_save_folder
    
    # Create download directory
    os.makedirs(download_dir, exist_ok=True)
    
    # Get HTML content of directory listing
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all links
    for link in soup.find_all('a'):
        file_name = link.get('href')
        if not file_name or file_name.startswith('?') or file_name.startswith('/'):
            continue  # Skip navigation or malformed links
        file_url = urljoin(base_url, file_name)
        
        # Optional: skip directories (you can add more logic here)
        if file_name.endswith('/'):
            continue
    
        print(f"Downloading {file_name}...")
        file_path = os.path.join(download_dir, file_name)
        
        if os.path.exists(file_path):
            print(f"Skipping {file_name}, already exists.")
            continue
    
        # Download and save
        print(f"Downloading from: {file_url}")
        with requests.get(file_url, stream=True) as r:
            if r.status_code != 200:
                print(f"Failed to download {file_name}, status: {r.status_code}\nResponse headers: {r.headers}")
                continue
            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)


# %%
for key, (URL, local_save_folder) in download.items():
    print(f"\nStarting download for: {key}")
    download_CEDS_proxy_data(URL=URL, local_save_folder=local_save_folder)

# %%
