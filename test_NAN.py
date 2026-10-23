import pandas as pd
import os

# 1. CSV 파일을 데이터프레임으로 변환
path = './df_weather_preprocessed'
for f in os.listdir(path):
    df = pd.read_csv(path + '/'+ f)
    # 각 컬럼별 NaN 개수 확인
    print(df.isnull().sum())