# -*- coding: utf-8 -*-
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
        "C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/concordia_cmip7_v0_testing/input/gridding/20250523/Jarmo_files/non-point_source_proxy_final_sector" # local_save_folder
    ],
    # 
    "seasonality": [
        "https://rcdtn1.pnl.gov/data/CEDS/Jarmo_files/seasonality/", # url
        "C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/concordia_cmip7_v0_testing/input/gridding/20250523/Jarmo_files/seasonality" # local_save_folder
    ],
}

# %%
# URL = "https://rcdtn1.pnl.gov/data/CEDS/Jarmo_files/non-point_source_proxy_final_sector" # provided by (Ahsan, Hamza<hamza.ahsan@pnnl.gov>) on Friday, May 23, 2025 05:41 
# To: KIKSTRA Jarmo <kikstra@iiasa.ac.at>; ssmith-pnnl.gov <ssmith@pnnl.gov>; Gidden, Matthew J <matthew.gidden@pnnl.gov>; Hoesly, Rachel M <rachel.hoesly@pnnl.gov>
# Cc: Zebedee Nicholls <zebedee.nicholls@climate-resource.com>; HOEGNER Annika <hoegner@iiasa.ac.at>; Jonas Hörsch <jonas.hoersch@climateanalytics.org>
# Subject: RE: Updating CEDS proxy data for creating CMIP7 scenarios in line with CEDS ESGF products

# From: Ahsan, Hamza <hamza.ahsan@pnnl.gov>
# Sent: Friday, May 23, 2025 05:41
# To: KIKSTRA Jarmo <kikstra@iiasa.ac.at>; ssmith-pnnl.gov <ssmith@pnnl.gov>; Gidden, Matthew J <matthew.gidden@pnnl.gov>; Hoesly, Rachel M <rachel.hoesly@pnnl.gov>
# Cc: Zebedee Nicholls <zebedee.nicholls@climate-resource.com>; HOEGNER Annika <hoegner@iiasa.ac.at>; Jonas Hörsch <jonas.hoersch@climateanalytics.org>
# Subject: RE: Updating CEDS proxy data for creating CMIP7 scenarios in line with CEDS ESGF products
 
# Hi Jarmo,

 

# You may find the files here: https://rcdtn1.pnl.gov/data/CEDS/Jarmo_files

 

# Let me know if you have any questions.

 

# Best,

 

# Hamza

 

# From: KIKSTRA Jarmo <kikstra@iiasa.ac.at>
# Sent: Wednesday, May 21, 2025 10:03 AM
# To: Ahsan, Hamza <hamza.ahsan@pnnl.gov>; Smith, Steve J <ssmith@pnnl.gov>; Gidden, Matthew J <matthew.gidden@pnnl.gov>; Hoesly, Rachel M <rachel.hoesly@pnnl.gov>
# Cc: Zebedee Nicholls <zebedee.nicholls@climate-resource.com>; HOEGNER Annika <hoegner@iiasa.ac.at>; Jonas Hörsch <jonas.hoersch@climateanalytics.org>
# Subject: Re: Updating CEDS proxy data for creating CMIP7 scenarios in line with CEDS ESGF products

 

# Dear Hamza,

 

# So sorry for the long delay in responding (was on leave for 1.5 weeks and didn't get back to all emails until now). 

 

# Yes, please - if you could upload them and send me the link that would be great. 

# Assuming it is not a hassle, maybe all years 2010-2023? I don't think I will use more than that, but I think having some history will allow me to get a feeling for how these grids change over time. 

 

# Let's start from that and see if I can create the proxies from that myself. If I somehow struggle, I may come back.  

 

# Re: "We can also send you total emissions by iso, sector, year as a check" - I think I already have these, from the 18 March Zenodo release? 

 

# All the best,

 

# Jarmo

 

# From: Ahsan, Hamza <hamza.ahsan@pnnl.gov>
# Sent: Thursday, May 01, 2025 05:50
# To: KIKSTRA Jarmo <kikstra@iiasa.ac.at>; ssmith-pnnl.gov <ssmith@pnnl.gov>; Gidden, Matthew J <matthew.gidden@pnnl.gov>; Hoesly, Rachel M <rachel.hoesly@pnnl.gov>
# Cc: Zebedee Nicholls <zebedee.nicholls@climate-resource.com>; HOEGNER Annika <hoegner@iiasa.ac.at>; Jonas Hörsch <jonas.hoersch@climateanalytics.org>
# Subject: RE: Updating CEDS proxy data for creating CMIP7 scenarios in line with CEDS ESGF products

 

# Hi Jarmo,

 

# I would be happy to send you the files that Steve outlined. To clarify further, we have:

# Non-point source emissions by iso, sector, fuel, and year in csv format
# Non-point source proxy for each sector and year as R data objects (Note that these are global proxies. However, we have iso masks that you can apply to create iso specific proxies)
# Database of point source emissions in yml format (Each yml file contains iso, sector, lat, lon and corresponding time series values)
 

# It would probably be convenient to upload these to our dtn server for you to download.

 

# Hamza

 

# From: KIKSTRA Jarmo <kikstra@iiasa.ac.at>
# Sent: Wednesday, April 30, 2025 3:12 AM
# To: Smith, Steve J <ssmith@pnnl.gov>; Gidden, Matthew J <matthew.gidden@pnnl.gov>; Hoesly, Rachel M <rachel.hoesly@pnnl.gov>; Ahsan, Hamza <hamza.ahsan@pnnl.gov>
# Cc: Zebedee Nicholls <zebedee.nicholls@climate-resource.com>; HOEGNER Annika <hoegner@iiasa.ac.at>; Jonas Hörsch <jonas.hoersch@climateanalytics.org>
# Subject: Re: Updating CEDS proxy data for creating CMIP7 scenarios in line with CEDS ESGF products

 

# Not sure if my previous email went through - my Outlook says no, so apologies if it did.

# But since Steve is on leave until the 19th of May, I suppose the question is rather to @Hoesly, Rachel M and @Ahsan, Hamza directly.

 

# Any update on this? 

# IAM emissions reporting completeness is starting to get there, and we'd like to start the 'testing iteration' with UKESM soon. I can continue to use the old proxy files for now, which might be OK for us to check the format of the files I produce, but as it will give a different grid I'd expect grid-level jumps compared to with what you have done on the historical. 

 

# Best,

 

# Jarmo

 

# From: KIKSTRA Jarmo
# Sent: Saturday, April 19, 2025 16:37
# To: ssmith-pnnl.gov; Gidden, Matthew J
# Cc: Hoesly, Rachel M; Zebedee Nicholls; HOEGNER Annika; Jonas Hörsch; Ahsan, Hamza
# Subject: Re: Updating CEDS proxy data for creating CMIP7 scenarios in line with CEDS ESGF products

 

# Hi Steve,

 

# Thanks!

 

# Which years

# Before deciding that, I want to play around with it a bit to see what the results look like. I don't have enough experience in this to already judge what's best before implementation.

 

# Originally, I was thinking that we'd just use CEDS until 2023, and use the CEDS 2023 proxy for the future.  However, if there's good reasons to believe that 2023 is an outlier, we can also try 2021-2023 average. I'm not sure what effect COVID would have on spatial grids, but of course I'd very much expect differences e.g., in aircraft patterns (some countries opening air space much later) - if that's captured? So not sure whether longer averages would make sense - but a first step I think would be to compare how different they are - and then use that to understand what way is better.

 

# The main reason I always assumed that taking the last year is the best is continuity at the grid-cell level. Wouldn't there be more discontinuity if we don't take the 2023 grid? 

 

# In other words, we'd have the greatest flexibility if you could provide them at least for 2021, 2022, and 2023, and maybe average over the last 3 (2021-2023), 5 (2019-2023), and 10 years (2014-2023) - if that's easy to do? 

 

# If you or Hamza has the time to create the proxy_total(iso,lat,lon,sector, year) files directly, of course that would be an amazing help and great contribution!   

 

# Biomass burning

# Since Matt mentions biomass burning: is that something you can also provide as proxies? Or do I need to derive them myself from the CMIP7-BB4CMIP data?

 

# All the best,

 

# Jarmo 

 

 

# From: Smith, Steve J
# Sent: Friday, April 18, 2025 23:36
# To: Gidden, Matthew J
# Cc: KIKSTRA Jarmo; Hoesly, Rachel M; Zebedee Nicholls; HOEGNER Annika; Jonas Hörsch; Ahsan, Hamza
# Subject: Re: Updating CEDS proxy data for creating CMIP7 scenarios in line with CEDS ESGF products

 

# Might depend a bit as well on how you’re treating post 2021 data.

 

# Will you use CEDS for some of those years?

 

# Or generate those from the harmonization system?

 

# On Apr 18, 2025, at 5:02 PM, Gidden, Matthew J <matthew.gidden@pnnl.gov> wrote:

 

# Thanks a lot - that's really useful, Steve. 

 

# @KIKSTRA Jarmo I think we should probably take something like a 5-year average (or 2-3 year average @Smith, Steve J?) for most data and 10-year average for biomass burning - what do you think? Leyang's paper states the 10-year approach for BB but I couldn't find specifics on how patterns were developed for other sectors (e.g., last year, average over last N years, etc.).

 

# Best

# Matt

# From: Smith, Steve J <ssmith@pnnl.gov>
# Sent: Friday, April 18, 2025 4:53 PM
# To: Gidden, Matthew J <matthew.gidden@pnnl.gov>
# Cc: KIKSTRA Jarmo <kikstra@iiasa.ac.at>; Hoesly, Rachel M <rachel.hoesly@pnnl.gov>; Zebedee Nicholls <zebedee.nicholls@climate-resource.com>; HOEGNER Annika <hoegner@iiasa.ac.at>; Jonas Hörsch <jonas.hoersch@climateanalytics.org>; Ahsan, Hamza <hamza.ahsan@pnnl.gov>
# Subject: Re: Updating CEDS proxy data for creating CMIP7 scenarios in line with CEDS ESGF products

 

# (Ops, adding Hamza.)

 

# So let me back up a bit and write down some equations that describe our current CEDS gridding process.

 

# We first split emissions into two parts, point sources and everything else.

 

# So our total emissions are:

 

# E(iso,sector,year) = E_nonpt(iso, sector,year) + SUM: E_pt(n,iso,lat,lon,year)

 

 

# Our spatial allocation has two parts, the non-point source (which is the same as before) and point sources (which are directly placed on the grid). (N point sources, which have a spatial location, and an iso).

 

# The non-point sources are done as before, with a spatial template that has any point sources we are tracking removed from it.

 

# So the non-point source grid, is calculated as:

 

# Grid(lat,lon,sector,year) = SUM-over-isos-of. E_nonpt(iso, sector,year) * proxy_non_point(iso,sector,year)

 

# So the final grid is just:

 

# Grid(lat,lon,sector,year) = SUM-over-isos-of: E_nonpt(iso, sector,year) * proxy_non_point(iso,sector,year) + E_pt(n,iso,lat,lon,year)

 

# Where proxy(iso,sector,year) is the normalized spatial proxy by calendar year, iso, and gridding sector. (Same as before except with point sources removed.)

 

# (As before, our normalization splits cells that are in more than one country across countries using finer-scale country grid maps, such that the individual iso maps generated by the emissions * proxy values can simply be added together spatially without any further scaling).

 

 

# We, therefore, don’t’ generate an aggregate proxy anymore, but how one would do that is as follows:

 

# The gridded proxy for any specific species, for a specific iso would be:

 

# proxy_total(iso,lat,lon,sector, year) = [ E_nonpt(iso, sector,year) * proxy_non_point(iso,sector,year) + E_pt(n,iso,lat,lon,year) ] / E(iso,sector,year) 

 

# {Where the proxy for each iso, sector, year combination is generated by combining the above elements for that iso}

 

 

# So these are the components we can send you:

 

# E_nonpt(iso, sector,year) = non-point source emissions by iso, sector, and year (.csv)

# proxy_non_point(iso,sector,year) = normalized non-point source proxy for each iso, sector, and year (R data object)

# E_pt(n,iso,lat,lon,year) ] = database of point source emissions (by iso, location, year) (I think this is .csv, but maybe not.)

 

# We can also send you total emissions by iso, sector, year as a check.

 

# Hamza can send you all these components (their native data format in our system is in parenthesis).

 

# Would you want this for just the last historical year (2021), for 2021 - 2023, something else?

 

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
