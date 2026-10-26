# 2단계 데이터 전처리 ex) 불필요한 컬럼(발생 시각 관련) 제외
import os
import pandas as pd

INPUT_PATH = './2_df_weather_preprocessed'
OUTPUT_PATH = './3_df_weather_drop'

DROP_RAW_COLUMNS = [
    "WS_MAX_TM",
    "WS_INS_TM",
    "TA_MAX_TM",
    "TA_MIN_TM",
    "HM_MIN_TM",
    "PSMAX_TM",
    "PS_MIN_TM",
    "SI_60M_MAX_TM",
    "RN10M_MAX_TM",
    "RN_60M_MAX_TM",
    "RN_POW_MAX_TM",
    "SD_NEW_TM",
    "SD_MAX_TM",
    "WD_INS",
]

for f in os.listdir(INPUT_PATH):
    df = pd.read_csv(INPUT_PATH+'/'+f)
    # print(df.shape[1]) #130
    df.drop(columns=DROP_RAW_COLUMNS,inplace=True)
    print(df.shape) # 90
    df.to_csv(OUTPUT_PATH+'/'+f, index=False, encoding="utf-8-sig")