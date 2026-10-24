# STN코드와 지점명을 매핑하여 STN별 CSV로 분리 저장하는 전처리 스크립트
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent
WEATHER_DATA_FILE = "weather_data.csv"
STATION_METADATA_FILE = "META_관측지점정보.csv"


def normalize_station_codes(station_codes: pd.Series) -> pd.Series:
    return station_codes.astype("string").str.strip().str.strip("'\"").str.strip()


def load_station_metadata(metadata_path: Path) -> dict[str, tuple[str, str]]:
    station_info = pd.read_csv(
        metadata_path,
        encoding="cp949",
        dtype={"지점": "string"},
    )
    station_info["지점"] = normalize_station_codes(station_info["지점"])
    station_info["지점명"] = station_info["지점명"].astype("string").str.strip()
    station_info["지점주소"] = station_info["지점주소"].astype("string").str.strip()

    has_station_details = (
        station_info["지점"].notna()
        & station_info["지점명"].notna()
        & station_info["지점명"].ne("")
        & station_info["지점주소"].notna()
        & station_info["지점주소"].ne("")
    )
    station_info = station_info.loc[
        has_station_details, ["지점", "지점명", "지점주소"]
    ].drop_duplicates("지점", keep="first")

    return {
        station_code: (station_name, station_address)
        for station_code, station_name, station_address in station_info.itertuples(
            index=False, name=None
        )
    }


def preprocess_weather_data(
    data_dir: Path = DATA_DIR,
    output_dir: Path | None = None,
) -> int:
    if output_dir is None:
        output_dir = data_dir

    weather_path = data_dir / WEATHER_DATA_FILE
    metadata_path = data_dir / STATION_METADATA_FILE
    df_weather = pd.read_csv(
        weather_path,
        encoding="cp949",
        skiprows=[1, 2],
        dtype={"STN": "string"},
        low_memory=False,
    )
    df_weather["STN"] = normalize_station_codes(df_weather["STN"])

    missing_station_rows = df_weather["STN"].isna()
    if missing_station_rows.any():
        raise ValueError(
            f"{weather_path}에 STN 코드가 없는 행이 "
            f"{int(missing_station_rows.sum())}개 있습니다."
        )

    station_metadata = load_station_metadata(metadata_path)
    rows_by_station = df_weather.groupby("STN", sort=False)
    unmapped_station_codes = [
        station_code
        for station_code in rows_by_station.groups
        if station_code not in station_metadata
    ]
    if unmapped_station_codes:
        print(
            "메타데이터에 없는 STN은 df_weather_{STN}_미등록.csv로 저장합니다: "
            + ", ".join(sorted(unmapped_station_codes))
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for station_code, station_weather in rows_by_station:
        if station_code in station_metadata:
            station_name, station_address = station_metadata[station_code]
            region = station_address.split()[0]
            filename = f"df_weather_{station_name}_{region}.csv"
        else:
            filename = f"df_weather_{station_code}_미등록.csv"

        station_weather.to_csv(
            output_dir / filename,
            index=False,
            encoding="utf-8-sig",
        )

    return len(rows_by_station)


if __name__ == "__main__":
    station_count = preprocess_weather_data()
    print(f"총 {station_count}개 STN의 기상 CSV 저장을 완료했습니다.")
