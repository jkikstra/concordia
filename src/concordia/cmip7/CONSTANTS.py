CONFIG = "config_cmip7_v0-4-0.yaml"
# CONFIG = "config_cmip7_esgf_v0_alpha.yaml"

CMIP_ERA = "CMIP6Plus" # for file-naming; note - currently doubles with DS ATTRS from cmip7 utils 

GASES = ["BC", "CH4", "CO", "CO2", "N2O", "NH3", "NOx", "OC", "Sulfur", "VOC"]
GASES_ESGF_CEDS = ["BC", "CH4", "CO", "CO2", "N2O", "NH3", "NOx", "OC", "SO2", "NMVOC"]
GASES_ESGF_BB4CMIP = ["BC", "CH4", "CO", "CO2", "N2O", "NH3", "NOx", "OC", "SO2", "NMVOCbulk"]

GASES_ESGF_CEDS_VOC = ["VOC01_alcohols_em_speciated_VOC_anthro",
                       "VOC02_ethane_em_speciated_VOC_anthro",
                       "VOC03_propane_em_speciated_VOC_anthro",
                       "VOC04_butanes_em_speciated_VOC_anthro",
                       "VOC05_pentanes_em_speciated_VOC_anthro",
                       "VOC06_hexanes_pl_em_speciated_VOC_anthro",
                       "VOC07_ethene_em_speciated_VOC_anthro",
                       "VOC08_propene_em_speciated_VOC_anthro",
                       "VOC09_ethyne_em_speciated_VOC_anthro",
                       "VOC12_other_alke_em_speciated_VOC_anthro",
                       "VOC13_benzene_em_speciated_VOC_anthro",
                       "VOC14_toluene_em_speciated_VOC_anthro",
                       "VOC15_xylene_em_speciated_VOC_anthro",
                       "VOC16_trimethylb_em_speciated_VOC_anthro",
                       "VOC17_other_arom_em_speciated_VOC_anthro",
                       "VOC18_esters_em_speciated_VOC_anthro",
                       "VOC19_ethers_em_speciated_VOC_anthro",
                       "VOC20_chlorinate_em_speciated_VOC_anthro",
                       "VOC21_methanal_em_speciated_VOC_anthro",
                       "VOC22_other_alka_em_speciated_VOC_anthro",
                       "VOC23_ketones_em_speciated_VOC_anthro",
                       "VOC24_acids_em_speciated_VOC_anthro",
                       "VOC25_other_voc_em_speciated_VOC_anthro"]

GASES_ESGF_BB4CMIP_VOC = ["C10H16",
                          "C2H2",
                          "C2H4", 
                          "C2H4O", 
                          "C2H5OH",
                          "C2H6", 
                          "C2H6S", 
                          "C3H6", 
                          "C3H6O", 
                          "C3H8", 
                          "C5H8", 
                          "C6H6", 
                          "C7H8", 
                          "C8H10", 
                          "CH2O", 
                          "CH3COCHO", 
                          "CH3COOH", 
                          "CH3OH", 
                          "HCN", 
                          "HCOOH", 
                          "HigherAlkanes", 
                          "HigherAlkenes", 
                          "HOCH2CHO", 
                          "MEK", 
                          "Toluenelump"]

def find_voc_data_variable_string(voc_code, voc_list=GASES_ESGF_CEDS_VOC):
    matching = [s for s in voc_list if voc_code in s]
    
    if len(matching) == 1:
        return matching[0]
    elif len(matching) == 0:
        raise ValueError(f"No string found containing '{voc_code}'")
    else:
        raise ValueError(f"Multiple strings found containing '{voc_code}': {matching}")


PROXY_YEARS =  [2022, 2023, 2024, 2025, 2030, 2035, 2040, 2045, 2050, 2055, 2060, 2065, 2070, 2075, 2080, 2085, 2090, 2095, 2100]

def return_marker_information(m, v="cmip7_esgf_v0_alpha", fixed_metadata=False, GRIDDING_VERSION=None, SCENARIO_SELECTION_GRIDDED_AFTER_METADATA=None):

    if v=="cmip7_esgf_v0_alpha":
        # outdated naming,
        # was only used for 0-3-0 release, when naming wasn't yet determined 
        if m == "VLLO":
            GRIDDING_VERSION = f"{v}_vllo" # vllo scenario remind test
            MODEL_SELECTION = "REMIND-MAgPIE 3.5-4.11"
            SCENARIO_SELECTION = "SSP1 - Very Low Emissions"
            if fixed_metadata:
                SCENARIO_SELECTION_GRIDDED_AFTER_METADATA = "scendraft1"
        elif m == "H":
            GRIDDING_VERSION = f"{v}_h" # high scenario gcam test
            MODEL_SELECTION = "GCAM 7.1 scenarioMIP"
            SCENARIO_SELECTION = "SSP3 - High Emissions"
            if fixed_metadata:
                SCENARIO_SELECTION_GRIDDED_AFTER_METADATA = "scendraft2"
        if not fixed_metadata:
            SCENARIO_SELECTION_GRIDDED_AFTER_METADATA = None

    else:
        # from 0-4-0 onwards
        if m == "H":
            MODEL_SELECTION = "GCAM 8s"
            SCENARIO_SELECTION = "SSP3 - High Emissions"
        if m == "HL":
            MODEL_SELECTION = "WITCH 6.0"
            SCENARIO_SELECTION = "SSP5 - Medium-Low Emissions_a"
        if m == "M":
            MODEL_SELECTION = "IMAGE 3.4"
            SCENARIO_SELECTION = "SSP2 - Medium Emissions"
        if m == "ML":
            MODEL_SELECTION = "COFFEE 1.6"
            SCENARIO_SELECTION = "SSP2 - Medium-Low Emissions"
        if m == "L":
            MODEL_SELECTION = "MESSAGEix-GLOBIOM-GAINS 2.1-M-R12"
            SCENARIO_SELECTION = "SSP2 - Low Emissions_f"
        if m == "LN":
            MODEL_SELECTION = "AIM 3.0"
            SCENARIO_SELECTION = "SSP2 - Low Overshoot" # should be updated to "SSP2 - Low Overshoot_a" (email from Shinichiro, weekend of 25.10.2025)
        if m == "VL":
            MODEL_SELECTION = "REMIND-MAgPIE 3.5-4.11"
            SCENARIO_SELECTION = "SSP1 - Very Low Emissions"

    return GRIDDING_VERSION, MODEL_SELECTION, SCENARIO_SELECTION, SCENARIO_SELECTION_GRIDDED_AFTER_METADATA
