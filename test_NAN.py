import pandas as pd
import os
"""
# 1. CSV 파일을 데이터프레임으로 변환
path = './df_weather_preprocessed'
for f in os.listdir(path):
    df = pd.read_csv(path + '/'+ f)
    # 각 컬럼별 NaN 개수 확인
    print(df.isnull().sum())
"""
df = pd.read_csv('./df_weather_featured_drop/df_weather_강릉_강원특별자치도_preprocessed_featured.csv')

result = df['RN_DAY'].value_counts()
print(result)