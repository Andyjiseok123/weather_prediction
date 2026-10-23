import os
import pandas as pd

INPUT_PATH = './df_weather_featured'
OUTPUT_PATH = './df_weather_featured_drop'

DROP_RAW_COLUMNS = [
    "WR_DAY",
    "WD_MAX",
    "WS_MAX",
    "WS_MAX_TM",
    "WD_INS",
    "WS_INS",
    "WS_INS_TM",
    "TA_MAX_TM",
    "TA_MIN_TM",
    "TD_AVG",
    "TS_AVG",
    "TG_MIN",
    "HM_AVG",
    "HM_MIN",
    "HM_MIN_TM",
    "PV_AVG",
    "EV_S",
    "EV_L",
    "PA_AVG",
    "PS_AVG",
    "PS_MAX",
    "PSMAX_TM",
    "PS_MIN",
    "PS_MIN_TM",
    "SS_DAY",
    "SS_DUR",
    "SI_DAY",
    "SI_60M_MAX",
    "SI_60M_MAX_TM",
    "RN_D99",
    "RN_DUR",
    "RN_10M_MAX",
    "RN10M_MAX_TM",
    "RN_60M_MAX_TM",
    "RN_POW_MAX",
    "RN_POW_MAX_TM",
    "SD_NEW",
    "SD_NEW_TM",
    "SD_MAX",
    "SD_MAX_TM",
    "TE_05",
    "TE_10",
    "TE_15",
    "TE_30",
    "TE_50",
]

for f in os.listdir(INPUT_PATH):
    df = pd.read_csv(INPUT_PATH+'/'+f)
    # print(df.shape[1]) #130
    df.drop(columns=DROP_RAW_COLUMNS,inplace=True)
    df.to_csv(OUTPUT_PATH+'/'+f, index=False, encoding="utf-8-sig")