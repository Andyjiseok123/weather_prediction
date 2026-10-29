from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "1_df_weather_bf_process"
COLUMNS_TO_COUNT = ("CA_TOT", "FG_DUR")


def main() -> None:
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {DATA_DIR}")

    csv_path = csv_files[0]
    weather_data = pd.read_csv(csv_path, encoding="utf-8-sig")

    missing_columns = [
        column for column in COLUMNS_TO_COUNT if column not in weather_data.columns
    ]
    if missing_columns:
        raise ValueError(f"{csv_path.name}에 필요한 열이 없습니다: {missing_columns}")

    print(f"파일: {csv_path.name}")
    for column in COLUMNS_TO_COUNT:
        print(f"\n[{column}] 값별 개수")
        print(weather_data[column].value_counts(dropna=False).sort_index().to_string())


if __name__ == "__main__":
    main()