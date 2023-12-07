import pandas as pd
from paths import *

SIMULATIONS = {
    #### =========== SFHo ================
    "SFHo_135_135_res150" : {
        "idx":0,
        "name":"SFHo_135_135_150m_11",
        "datadir":DATADIR+"SFHo_135_135_150m_11"+"/",
        "tmerg":14.9,
        "mdot_extract":"Mdot_extraction_SFHo_135_135.txt",
        "rhomax":"rhomax_SFHo_135_135.txt",
        "EOS":"SFHo",
        "q":1.,
        "res":150,
        "given_time":"old",
        "given_date":"",
        "label":r"SFHo (1.35-1.35)$M_{\odot}$ (150m)"
    },
    # --------------------------
    "SFHo_13_14_res150" : {
        "idx":1,
        "name":"SFHoTim276_13_14_0025_150mstg_B0_HLLC",
        "datadir":DATADIR+"SFHoTim276_13_14_0025_150mstg_B0_HLLC"+"/",
        "tmerg":15.,#None,           # TODO FILL IT
        "mdot_extract":"Mdot_extraction_SFHo_13_14.txt",
        "rhomax":"rhomax_SFHo_13_14.txt",
        "EOS":"SFHo",
        "q":1.0769230769230769,
        "res":150,
        "given_time":"new",
        "given_date":"10/26/2023", # "9/7/2023"
        "label":"SFHo (1.3-1.4)$M_{\odot}$ (150m)"
    },
    # --------------------------
    "SFHo_125_145_res150" : {
        "idx":2,
        "name":"SFHo_125_145_150m_11",
        "datadir":DATADIR+"SFHo_125_145_150m_11"+"/",
        "EOS":"SFHo",
        "tmerg":None,           # TODO FILL IT
        "mdot_extract":"",    # TODO FILL IT
        "rhomax":"",          # TODO FILL IT
        "q":1.16,
        "res":150,
        "given_time":"old",
        "given_date":"",
        "label":"SFHo (1.25-1.45)$M_{\odot}$ (150m)"
    },
    "SFHo_125_145_res200" : {
        "idx":3,
        "name":"SFHoTim276_125_145_0025_200mstg_B0_HLLC",
        "datadir":DATADIR+"SFHoTim276_125_145_0025_200mstg_B0_HLLC"+"/",
        "EOS":"SFHo",
        "tmerg":None,           # TODO FILL IT
        "mdot_extract":"Mdot_SFHo_extraction_SFHo_125_145_200m.txt",
        "rhomax":"rhomax_SFHo_125_145_200m.txt",
        "q":1.16,
        "res":200,
        "given_time":"old",
        "given_date":"9/25/2023",
        "label":"SFHo (1.25-1.45)$M_{\odot}$ (200m)"
    },
    # --------------------------
    "SFHo_12_15_res150" : {
        "idx":4,
        "name":"SFHoTim276_12_15_0025_150mstg_B0_HLLC",
        "datadir":DATADIR+"SFHoTim276_12_15_0025_150mstg_B0_HLLC"+"/",
        "EOS":"SFHo",
        "tmerg":None,           # TODO FILL IT
        "mdot_extract":"Mdot_extraction_SFHo_12_15.txt",
        "rhomax":"rhomax_SFHo_12_15.txt",
        "q":1.25,
        "res":150,
        "given_time":"old",
        "given_date":"9/15/2023",
        "label":"SFHo (1.2-1.5)$M_{\odot}$ (150m)"
    },
    "SFHo_12_15_res200" : {
        "idx":5,
        "name":"SFHo_12_15_200m_11",
        "datadir":DATADIR+"SFHo_12_15_200m_11"+"/",
        "EOS":"SFHo",
        "tmerg":None,           # TODO FILL IT
        "mdot_extract":"",    # TODO FILL IT
        "rhomax":"",          # TODO FILL IT
        "q":1.25,
        "res":200,
        "given_time":"old",
        "given_date":"",
        "label":r"SFHo (1.2-1.5)$M_{\odot}$ (200m)"
    },
    "SFHo_12_15_res150_B15_HLLD":{
        "idx":6,
        "name":"SFHoTim276_12_15_0025_150mstg_B15_HLLD_CT_GS_onFugaku",
        "datadir":DATADIR+"SFHoTim276_12_15_0025_150mstg_B15_HLLD_CT_GS_onFugaku"+"/",
        "EOS":"SFHo",
        "tmerg":15.,#None,           # TODO FILL IT
        "mdot_extract":"Mdot_extraction_SFHo_12_15.txt",
        "rhomax":"max_SFHo_12_15.txt",
        "q":1.25,
        "res":150,
        "given_time":"new",
        "given_data":"11/01/2023",
        "label":r"SFHo$^*$ (1.2-1.5)$M_{\odot}$ (150m)"
    },
    #### =========== BHBlp ================
    "BHBLp_135_135_res150" : {
        "idx":7,
        "name":"BHBLpTim326_135_135_45km_150mstg_B0_HLLC",
        "datadir":DATADIR+"BHBLpTim326_135_135_45km_150mstg_B0_HLLC"+"/",
        "EOS":"BHBLp",
        "tmerg":15.5,         # ms
        # "mdot_extract":"Mdot_extraction_BHBLp_135_135.txt",
        "mdot_extract":"Mdot_extraction_BHBLp_135_135.txt",
        "rhomax":"rhomax_BHBLp_135_135.txt",
        "q":1.,
        "res":150,
        "given_time":"new",
        "given_date":"10/11/2023",
        "label":"BHBLp (1.35-1.35)$M_{\odot}$ (150m)"
    },
}

SIMS = pd.DataFrame.from_dict(SIMULATIONS).T
