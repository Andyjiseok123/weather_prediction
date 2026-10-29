import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

import os
import hashlib
import tempfile

import psycopg2
from dotenv import load_dotenv

from weather_model import (
    load_dataset,
    recursive_forecast_x,
    create_lightgbm_features,
)

def restore_model_from_db(model_id, model_path):
    """
    PostgreSQL weather_models 테이블에서 모델 아티팩트를 읽어
    model_path에 복원하고 SHA-256 체크섬을 검증한다.
    """
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(env_path)

    # 아래 환경변수 키는 save_model_to_db.py에서
    # 실제 사용하는 키와 동일하게 맞춰야 한다.
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME")
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")

    missing = [
        name
        for name, value in {
            "DB_HOST": db_host,
            "DB_NAME": db_name,
            "DB_USER": db_user,
            "DB_PASSWORD": db_password,
        }.items()
        if not value
    ]
    if missing:
        raise EnvironmentError(
            "DB 환경변수가 설정되지 않았습니다: "
            + ", ".join(missing)
            + ". save_model_to_db.py와 동일한 환경변수 키인지 확인하세요."
        )

    artifact_columns = {
        "x_model.keras": "x_model",
        "y_model.keras": "y_model",
        "dense_ensemble_model.keras": "dense_ensemble_model",
        "lightgbm_model.pkl": "lightgbm_model",
        "ensemble_input_scaler.pkl": "ensemble_input_scaler",
        "metadata.pkl": "metadata_pickle",
    }

    conn = None

    try:
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            dbname=db_name,
            user=db_user,
            password=db_password,
            connect_timeout=10,
        )

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    x_model,
                    y_model,
                    dense_ensemble_model,
                    lightgbm_model,
                    ensemble_input_scaler,
                    metadata_pickle,
                    file_checksums
                FROM weather_models
                WHERE model_id = %s
                """,
                (model_id,),
            )
            row = cur.fetchone()

        if row is None:
            raise LookupError(
                f"DB에서 모델을 찾을 수 없습니다: {model_id}"
            )

        blobs = row[:6]
        stored_checksums = row[6]

        if isinstance(stored_checksums, str):
            import json
            stored_checksums = json.loads(stored_checksums)

        if not isinstance(stored_checksums, dict):
            raise ValueError(
                "DB의 file_checksums가 올바른 JSON 객체가 아닙니다."
            )

        model_path.mkdir(parents=True, exist_ok=True)

        
        for (filename, column_name), blob in zip(
            artifact_columns.items(), blobs
        ):
            if blob is None:
                raise ValueError(
                    f"DB의 {column_name} 아티팩트가 비어 있습니다."
                )

            blob = bytes(blob)
            actual_hash = hashlib.sha256(blob).hexdigest()

            # DB의 file_checksums는 파일명이 아닌 컬럼 키를 사용한다.
            expected_hash = stored_checksums.get(column_name)

            if expected_hash is None:
                raise ValueError(
                    f"DB 체크섬을 찾을 수 없습니다: {column_name}"
                )

            if actual_hash != expected_hash:
                raise ValueError(
                    f"SHA-256 검증 실패: {filename}"
                )

            (model_path / filename).write_bytes(blob)


        print(f"DB 모델 로드 및 체크섬 검증 성공: {model_id}")

    finally:
        if conn is not None:
            conn.close()


def predict_one_dataset(
    model_id,
    target_date,
    data_dir,
    model_dir,
    model_source="local"
):
    model_path = model_dir / model_id

    if model_source == "db":
        restore_model_from_db(
            model_id=model_id,
            model_path=model_path,
        )

    metadata_path = model_path / "metadata.pkl"
    x_model_path = model_path / "x_model.keras"
    y_model_path = model_path / "y_model.keras"
    dense_model_path = model_path / "dense_ensemble_model.keras"
    lgbm_model_path = model_path / "lightgbm_model.pkl"
    ensemble_scaler_path = model_path / "ensemble_input_scaler.pkl"

    # -----------------------------------------------------
    # 1. 저장된 메타데이터 로드
    # -----------------------------------------------------

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"메타데이터가 없습니다: {metadata_path}"
        )

    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    lookback = metadata["lookback"]
    x_columns = metadata["x_columns"]
    target_column = metadata["target_column"]

    x_scaler = metadata["x_scaler"]
    y_scaler = metadata["y_scaler"]

    # -----------------------------------------------------
    # 2. 원본 CSV 로드
    # -----------------------------------------------------

    csv_path = data_dir / metadata["relative_csv_path"]

    if not csv_path.exists():
        raise FileNotFoundError(
            f"원본 CSV를 찾을 수 없습니다: {csv_path}"
        )

    data = load_dataset(
        csv_path,
        start_date=metadata["start_date"],
        date_column=metadata["date_column"],
    )

    X_raw = data["X"]
    dates = data["dates"]

    # 학습 당시와 특성 열의 순서가 같은지 확인
    if data["x_columns"] != x_columns:
        raise ValueError(
            "학습 당시와 현재 CSV의 X 열 또는 열 순서가 다릅니다.\n"
            "원본 CSV가 변경되었는지 확인하세요."
        )

    if data["target_column"] != target_column:
        raise ValueError(
            "학습 당시와 현재 CSV의 Y 열이 다릅니다."
        )

    last_observed_date = pd.Timestamp(
        metadata["last_observed_date"]
    ).normalize()

    if dates[-1] != last_observed_date:
        raise ValueError(
            "현재 CSV의 마지막 관측일이 학습 당시와 다릅니다.\n"
            f"학습 당시: {last_observed_date.date()}\n"
            f"현재 CSV: {dates[-1].date()}\n"
            "데이터가 갱신되었다면 모델을 다시 학습하세요."
        )

    # -----------------------------------------------------
    # 3. 목표 날짜 확인
    # -----------------------------------------------------

    target_date = pd.Timestamp(target_date).normalize()

    if target_date <= last_observed_date:
        raise ValueError(
            "target_date는 마지막 관측일보다 미래여야 합니다.\n"
            f"마지막 관측일: {last_observed_date.date()}\n"
            f"요청 날짜: {target_date.date()}"
        )

    future_dates = pd.date_range(
        start=last_observed_date + pd.Timedelta(days=1),
        end=target_date,
        freq="D"
    )

    # -----------------------------------------------------
    # 4. 모델 로드
    # -----------------------------------------------------

    if not x_model_path.exists() or not y_model_path.exists():
        raise FileNotFoundError(
            f"저장된 모델 파일이 없습니다: {model_path}"
        )

    model_x = tf.keras.models.load_model(x_model_path)
    model_y = tf.keras.models.load_model(y_model_path)

    # -----------------------------------------------------
    # 5. 목표 날짜까지 X 재귀 예측
    # -----------------------------------------------------

    future_X_raw = recursive_forecast_x(
        model_x=model_x,
        x_scaler=x_scaler,
        x_columns=x_columns,
        history_X_raw=X_raw,
        future_dates=future_dates,
        lookback=lookback,
    )

    # 실제 과거 X + 예측 미래 X
    combined_X_raw = np.vstack([
        X_raw,
        future_X_raw
    ])

    # -----------------------------------------------------
    # 6. 목표 날짜의 X 시퀀스로 Y 예측
    # -----------------------------------------------------

    combined_X_scaled = x_scaler.transform(combined_X_raw)

    X_target_sequence = combined_X_scaled[-lookback:]
    X_target_sequence = X_target_sequence.reshape(
        1,
        lookback,
        len(x_columns)
    )

    y_pred_scaled = model_y.predict(
        X_target_sequence,
        verbose=0
    )

    y_pred_bilstm = float(y_scaler.inverse_transform(y_pred_scaled)[0, 0])

    # 미래 X 중 목표 날짜에 해당하는 마지막 행
    target_X = future_X_raw[-1]

    # 새 모델은 Bi-LSTM + LightGBM 예측값을 Dense stacking 모델로 결합한다.
    # 이전에 저장한 모델 디렉터리도 사용할 수 있도록 파일이 없으면 Bi-LSTM 예측으로 fallback한다.
    y_pred = y_pred_bilstm
    y_pred_lgbm = None
    ensemble_type = "bilstm_only"
    if all(path.exists() for path in [dense_model_path, lgbm_model_path, ensemble_scaler_path]):
        with open(lgbm_model_path, "rb") as f:
            model_lgbm = pickle.load(f)
        with open(ensemble_scaler_path, "rb") as f:
            ensemble_input_scaler = pickle.load(f)

        all_features = create_lightgbm_features(combined_X_raw, x_columns)
        lgbm_target_features = all_features.iloc[[-1]].copy()
        lgbm_target_features = lgbm_target_features.replace([np.inf, -np.inf], np.nan)
        if lgbm_target_features.isna().any().any():
            raise ValueError(
                "목표 날짜의 LightGBM Feature에 결측치가 있습니다. "
                "관측 이력과 LAG/rolling feature를 확인하세요."
            )
        y_pred_lgbm = float(model_lgbm.predict(lgbm_target_features)[0])
        meta_features = np.asarray([[y_pred_bilstm, y_pred_lgbm]], dtype=np.float32)
        meta_features = ensemble_input_scaler.transform(meta_features).astype(np.float32)
        model_dense = tf.keras.models.load_model(dense_model_path)
        y_pred_ensemble_scaled = model_dense.predict(meta_features, verbose=0)
        y_pred = float(y_scaler.inverse_transform(y_pred_ensemble_scaled)[0, 0])
        ensemble_type = "dense_stacking"

    result = {
        "model_id": model_id,
        "target_date": str(target_date.date()),
        "last_observed_date": str(last_observed_date.date()),
        "target_column": target_column,
        "predicted_y": float(y_pred),
        "predicted_y_bilstm": y_pred_bilstm,
        "predicted_y_lightgbm": y_pred_lgbm,
        "ensemble_type": ensemble_type,
        "predicted_x": {
            col: float(value)
            for col, value in zip(x_columns, target_X)
        },
        "forecast_days": len(future_dates),
    }

    return result


def predict_many_dates(
    model_id,
    target_dates,
    data_dir,
    model_dir,
):
    """모델을 한 번 로드하여 여러 미래 날짜의 예측 결과를 반환한다."""

    model_path = model_dir / model_id

    with open(model_path / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    x_columns = metadata["x_columns"]
    target_column = metadata["target_column"]
    x_scaler = metadata["x_scaler"]
    y_scaler = metadata["y_scaler"]
    lookback = metadata["lookback"]

    csv_path = data_dir / metadata["relative_csv_path"]
    if not csv_path.exists():
        raise FileNotFoundError(f"원본 CSV가 없습니다: {csv_path}")

    data = load_dataset(
        csv_path,
        start_date=metadata["start_date"],
        date_column=metadata["date_column"],
    )

    X_raw = data["X"]
    dates = data["dates"]

    if data["x_columns"] != x_columns:
        raise ValueError("학습 당시와 CSV의 입력 컬럼이 다릅니다.")

    if data["target_column"] != target_column:
        raise ValueError("학습 당시와 CSV의 타깃 컬럼이 다릅니다.")

    last_observed_date = pd.Timestamp(
        metadata["last_observed_date"]
    ).normalize()

    if dates[-1] != last_observed_date:
        raise ValueError(
            "CSV의 마지막 관측일이 모델 메타데이터와 다릅니다."
        )

    requested_dates = [
        pd.Timestamp(d).normalize()
        for d in target_dates
    ]

    if not requested_dates:
        raise ValueError("예측 날짜가 없습니다.")

    if len(requested_dates) != len(set(requested_dates)):
        raise ValueError("중복된 예측 날짜가 있습니다.")

    for target_date in requested_dates:
        if target_date <= last_observed_date:
            raise ValueError(
                f"{target_date.date()}는 마지막 관측일 "
                f"{last_observed_date.date()} 이후여야 합니다."
            )

    # 가장 먼 요청 날짜까지 X를 재귀 예측한다.
    future_dates = pd.date_range(
        start=last_observed_date + pd.Timedelta(days=1),
        end=max(requested_dates),
        freq="D",
    )

    model_x = tf.keras.models.load_model(
        model_path / "x_model.keras"
    )
    model_y = tf.keras.models.load_model(
        model_path / "y_model.keras"
    )

    future_X_raw = recursive_forecast_x(
        model_x=model_x,
        x_scaler=x_scaler,
        x_columns=x_columns,
        history_X_raw=X_raw,
        future_dates=future_dates,
        lookback=lookback,
    )

    combined_X_raw = np.vstack([X_raw, future_X_raw])
    combined_X_scaled = x_scaler.transform(combined_X_raw)

    # 앙상블 모델이 존재하면 한 번만 로드한다.
    dense_path = model_path / "dense_ensemble_model.keras"
    lgbm_path = model_path / "lightgbm_model.pkl"
    scaler_path = model_path / "ensemble_input_scaler.pkl"

    use_ensemble = all(
        path.exists()
        for path in [dense_path, lgbm_path, scaler_path]
    )

    if use_ensemble:
        with open(lgbm_path, "rb") as f:
            model_lgbm = pickle.load(f)

        with open(scaler_path, "rb") as f:
            ensemble_scaler = pickle.load(f)

        model_dense = tf.keras.models.load_model(dense_path)

        all_lgbm_features = create_lightgbm_features(
            combined_X_raw,
            x_columns,
        )

    results = []

    for target_date in requested_dates:
        # combined_X_raw에서 해당 날짜의 행 인덱스
        day_offset = (target_date - last_observed_date).days
        target_idx = len(X_raw) + day_offset - 1

        sequence = combined_X_scaled[
            target_idx - lookback + 1 : target_idx + 1
        ]

        if len(sequence) != lookback:
            raise ValueError("예측 입력 시퀀스 길이가 올바르지 않습니다.")

        sequence = sequence.reshape(
            1, lookback, len(x_columns)
        )

        y_scaled = model_y.predict(sequence, verbose=0)
        y_bilstm = float(
            y_scaler.inverse_transform(y_scaled)[0, 0]
        )

        y_lgbm = None
        y_pred = y_bilstm
        ensemble_type = "bilstm_only"

        if use_ensemble:
            lgbm_features = all_lgbm_features.iloc[
                [target_idx]
            ].copy()

            if lgbm_features.isna().any().any():
                raise ValueError(
                    f"{target_date.date()}의 LightGBM 입력에 "
                    "결측치가 있습니다."
                )

            y_lgbm = float(
                model_lgbm.predict(lgbm_features)[0]
            )

            meta_features = np.asarray(
                [[y_bilstm, y_lgbm]],
                dtype=np.float32,
            )
            meta_features = ensemble_scaler.transform(
                meta_features
            ).astype(np.float32)

            y_ensemble_scaled = model_dense.predict(
                meta_features,
                verbose=0,
            )
            y_pred = float(
                y_scaler.inverse_transform(y_ensemble_scaled)[0, 0]
            )

            # 강수량은 음수가 될 수 없으므로 최종 예측값 보정
            if target_column == "RN_DAY":
                y_pred = max(0.0, y_pred)

            ensemble_type = "dense_stacking"

        target_x = future_X_raw[day_offset - 1]

        results.append({
            "model_id": model_id,
            "target_date": str(target_date.date()),
            "last_observed_date": str(last_observed_date.date()),
            "target_column": target_column,
            "predicted_y": float(y_pred),
            "predicted_y_bilstm": y_bilstm,
            "predicted_y_lightgbm": y_lgbm,
            "ensemble_type": ensemble_type,
            "predicted_x": {
                col: float(value)
                for col, value in zip(x_columns, target_x)
            },
            "forecast_days": day_offset,
        })

    return results

def main():
    parser = argparse.ArgumentParser(
        description="저장된 Bi-LSTM/LightGBM/Dense 앙상블 모델로 목표 날짜의 Y 예측"
    )

    parser.add_argument(
        "--model-id",
        required=True,
        help="학습 시 출력된 모델 ID"
    )

    parser.add_argument(
        "--target-date",
        required=True,
        help="예측할 날짜. 예: 2026-11-05"
    )

    parser.add_argument(
        "--data-dir",
        default="6_dataset",
        help="원본 데이터 폴더"
    )

    parser.add_argument(
        "--model-dir",
        default="saved_models",
        help="저장된 모델 폴더"
    )

    parser.add_argument(
        "--model-source",
        choices=["local", "db"],
        default="local",
        help="모델 로딩 위치: local 또는 db (기본값: local)",
    )


    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    model_dir = Path(args.model_dir).resolve()

    
    if args.model_source == "db":
        # DB에서 내려받은 모델 파일은 임시 폴더에 보관한다.
        with tempfile.TemporaryDirectory(
            prefix="weather_db_models_"
        ) as temp_dir:
            result = predict_one_dataset(
                model_id=args.model_id,
                target_date=args.target_date,
                data_dir=data_dir,
                model_dir=Path(temp_dir),
                model_source="db",
            )
    else:
        result = predict_one_dataset(
            model_id=args.model_id,
            target_date=args.target_date,
            data_dir=data_dir,
            model_dir=model_dir,
            model_source="local",
        )


    print("\n" + "=" * 60)
    print("기상 예측 결과")
    print("=" * 60)
    print(f"모델 ID       : {result['model_id']}")
    print(f"목표 날짜     : {result['target_date']}")
    print(f"마지막 관측일 : {result['last_observed_date']}")
    print(f"예측 대상 Y   : {result['target_column']}")
    print(f"예측 Y 값     : {result['predicted_y']:.6f}")
    print(f"예측 방식     : {result['ensemble_type']}")
    print(f"Bi-LSTM 예측  : {result['predicted_y_bilstm']:.6f}")
    if result["predicted_y_lightgbm"] is not None:
        print(f"LightGBM 예측 : {result['predicted_y_lightgbm']:.6f}")
    print(f"재귀 예측 일수: {result['forecast_days']}일")

    print("\n[목표 날짜의 예측 X]")
    for column, value in result["predicted_x"].items():
        print(f"{column}: {value:.6f}")


if __name__ == "__main__":
    main()