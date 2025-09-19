CONFIG = "config_cmip7_esgf_v0_alpha.yaml"

CMIP_ERA = "CMIP6Plus" # for file-naming; note - currently doubles with DS ATTRS from cmip7 utils 

GASES = ["BC", "CH4", "CO", "CO2", "N2O", "NH3", "NOx", "OC", "Sulfur", "VOC"]
GASES_ESGF_CEDS = ["BC", "CH4", "CO", "CO2", "N2O", "NH3", "NOx", "OC", "SO2", "NMVOC"]
GASES_ESGF_BB4CMIP = ["BC", "CH4", "CO", "CO2", "N2O", "NH3", "NOx", "OC", "SO2", "NMVOCbulk"]
PROXY_YEARS =  [2022, 2023, 2024, 2025, 2030, 2035, 2040, 2045, 2050, 2055, 2060, 2065, 2070, 2075, 2080, 2085, 2090, 2095, 2100]

def return_marker_information(m, v="cmip7_esgf_v0_alpha"):
    if m == "H":
        GRIDDING_VERSION = f"{v}_h" # high scenario gcam test
        MODEL_SELECTION = "GCAM 7.1 scenarioMIP"
        SCENARIO_SELECTION = "SSP3 - High Emissions"
    elif m == "VLLO":
        GRIDDING_VERSION = f"{v}_vllo" # vllo scenario remind test
        MODEL_SELECTION = "REMIND-MAgPIE 3.5-4.11"
        SCENARIO_SELECTION = "SSP1 - Very Low Emissions"

    return GRIDDING_VERSION, MODEL_SELECTION, SCENARIO_SELECTION
