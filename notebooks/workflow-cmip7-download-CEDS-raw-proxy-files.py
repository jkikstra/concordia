# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: concordia
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
URL = "https://rcdtn1.pnl.gov/data/CEDS/Jarmo_files/" # provided by (Ahsan, Hamza<hamza.ahsan@pnnl.gov>) on Friday, May 23, 2025 05:41 
# To: KIKSTRA Jarmo <kikstra@iiasa.ac.at>; ssmith-pnnl.gov <ssmith@pnnl.gov>; Gidden, Matthew J <matthew.gidden@pnnl.gov>; Hoesly, Rachel M <rachel.hoesly@pnnl.gov>
# Cc: Zebedee Nicholls <zebedee.nicholls@climate-resource.com>; HOEGNER Annika <hoegner@iiasa.ac.at>; Jonas Hörsch <jonas.hoersch@climateanalytics.org>
# Subject: RE: Updating CEDS proxy data for creating CMIP7 scenarios in line with CEDS ESGF products

# Steve

# %%
import os
import requests
from bs4 import BeautifulSoup # needs to be installed (`pip install bs4`)
from urllib.parse import urljoin

# Base URL of the directory
base_url = URL
download_dir = "Jarmo_files"

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

    # Download and save
    with requests.get(file_url, stream=True) as r:
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
