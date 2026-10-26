from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "3_df_weather_drop"
OUTPUT_DIR = BASE_DIR / "3.5_df_weather_wind_feature"

WIND_DIRECTION_COLUMN = "WD_MAX"
WIND_DIRECTION_FEATURES = ["WD_MAX_SIN", "WD_MAX_COS"]


def add_wind_direction_features(csv_path: Path) -> Path:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    if WIND_DIRECTION_COLUMN not in df.columns:
        raise KeyError(
            f"{csv_path.name}에 필수 컬럼이 없습니다: "
            f"{WIND_DIRECTION_COLUMN}"
        )

    existing_features = [
        column for column in WIND_DIRECTION_FEATURES if column in df.columns
    ]
    if existing_features:
        raise ValueError(
            f"{csv_path.name}에 생성할 풍향 파생변수 컬럼이 "
            f"이미 있습니다: {existing_features}"
        )

    degrees = pd.to_numeric(df[WIND_DIRECTION_COLUMN], errors="raise")
    missing_mask = ~degrees.between(0, 360)
    valid_degrees = degrees.mask(missing_mask)
    interpolated_degrees = (
        valid_degrees.interpolate(method="linear", limit_area="inside")
        .bfill()
        .ffill()
    )
    filled_count = int(
        (valid_degrees.isna() & interpolated_degrees.notna()).sum()
    )
    remaining_count = int(interpolated_degrees.isna().sum())
    print(
        f"{csv_path.name}: 결측/범위 외 풍향 {int(missing_mask.sum())}개, "
        f"앞뒤 값으로 보간 {filled_count}개, "
        f"보간 불가 {remaining_count}개"
    )

    radians = np.deg2rad(interpolated_degrees)

    df["WD_MAX_SIN"] = np.sin(radians)
    df["WD_MAX_COS"] = np.cos(radians)
    df = df.drop(columns=[WIND_DIRECTION_COLUMN])

    output_path = OUTPUT_DIR / csv_path.name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def main() -> None:
    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(f"입력 폴더를 찾을 수 없습니다: {INPUT_DIR}")

    csv_files = sorted(INPUT_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"입력 폴더에 CSV 파일이 없습니다: {INPUT_DIR}")

    for csv_path in csv_files:
        output_path = add_wind_direction_features(csv_path)
        print(f"저장 완료: {output_path}")

    print(
        f"\n총 {len(csv_files)}개 CSV의 풍향 Feature 생성을 완료했습니다. "
        f"저장 폴더: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
