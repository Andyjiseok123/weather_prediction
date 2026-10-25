import argparse
import pickle
import traceback
from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from weather_model import (
    LOOKBACK,
    load_dataset,
    create_x_sequences,
    create_y_sequences,
    build_x_model,
    build_y_model,
    fit_model,
    recursive_forecast_x,
)


def make_model_id(csv_path, data_root):
    """
    상대 경로를 이용해 데이터셋별 고유 ID를 생성한다.

    예:
    6_dataset/지역A/LSTM_WS_AVG.csv
    -> 지역A__LSTM_WS_AVG
    """

    relative_path = csv_path.relative_to(data_root)
    parts = relative_path.with_suffix("").parts

    return "__".join(parts)


def validate_pipeline(
    X_raw,
    y_raw,
    dates,
    split_index,
    x_columns,
    lookback,
    epochs,
    batch_size,
):
    """
    시간 순서 검증:
    1. 앞쪽 데이터로 스케일러 학습
    2. 앞쪽 데이터로 X/Y 모델 학습
    3. 검증 구간의 X를 재귀 예측
    4. 예측된 X로 검증 구간의 Y를 예측
    """

    X_train_raw = X_raw[:split_index]
    y_train_raw = y_raw[:split_index]

    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_scaler.fit(X_train_raw)
    y_scaler.fit(y_train_raw)

    X_train_scaled = X_scaler.transform(X_train_raw)
    y_train_scaled = y_scaler.transform(y_train_raw)

    # 검증 구간 이전 데이터만으로 학습 샘플 구성
    X_x_train, y_x_train = create_x_sequences(
        X_train_scaled,
        lookback
    )

    X_y_train, y_y_train = create_y_sequences(
        X_train_scaled,
        y_train_scaled,
        lookback
    )

    n_features = X_raw.shape[1]

    model_x = build_x_model(lookback, n_features)
    model_y = build_y_model(lookback, n_features)

    fit_model(
        model_x,
        X_x_train,
        y_x_train,
        epochs=epochs,
        batch_size=batch_size
    )

    fit_model(
        model_y,
        X_y_train,
        y_y_train,
        epochs=epochs,
        batch_size=batch_size
    )

    # 검증 구간 전체의 X를 하루씩 재귀 예측
    validation_dates = dates[split_index:]

    predicted_X_raw = recursive_forecast_x(
        model_x=model_x,
        x_scaler=X_scaler,
        x_columns=x_columns,
        history_X_raw=X_train_raw,
        future_dates=validation_dates,
        lookback=lookback
    )

    # X 예측 성능
    true_X_validation = X_raw[split_index:]

    x_mae = mean_absolute_error(
        true_X_validation,
        predicted_X_raw
    )

    x_rmse = np.sqrt(
        mean_squared_error(
            true_X_validation,
            predicted_X_raw
        )
    )

    # 실제 검증 X 대신 Model A가 예측한 X를 사용
    combined_X_raw = np.vstack([
        X_train_raw,
        predicted_X_raw
    ])

    combined_X_scaled = X_scaler.transform(combined_X_raw)

    X_y_validation = []

    for i in range(split_index, len(X_raw)):
        start = i - lookback + 1
        end = i + 1

        X_y_validation.append(
            combined_X_scaled[start:end]
        )

    X_y_validation = np.asarray(
        X_y_validation,
        dtype=np.float32
    )

    y_pred_scaled = model_y.predict(
        X_y_validation,
        verbose=0
    )

    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_true = y_raw[split_index:]

    y_mae = mean_absolute_error(y_true, y_pred)
    y_rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return {
        "x_mae": float(x_mae),
        "x_rmse": float(x_rmse),
        "y_mae": float(y_mae),
        "y_rmse": float(y_rmse),
    }


def train_one_csv(
    csv_path,
    data_root,
    output_root,
    start_date,
    date_column,
    lookback,
    epochs,
    batch_size,
    validation_fraction,
):
    print("\n" + "=" * 70)
    print(f"학습 대상: {csv_path}")

    data = load_dataset(
        csv_path,
        start_date=start_date,
        date_column=date_column
    )

    X_raw = data["X"]
    y_raw = data["y"]
    dates = data["dates"]
    x_columns = data["x_columns"]
    target_column = data["target_column"]

    n_rows, n_features = X_raw.shape

    if n_rows <= lookback + 10:
        raise ValueError(
            f"데이터가 부족합니다: {n_rows}행, lookback={lookback}"
        )

    split_index = int(n_rows * (1.0 - validation_fraction))

    if split_index <= lookback + 1 or split_index >= n_rows:
        raise ValueError(
            "학습/검증 분할 비율이 적절하지 않습니다."
        )

    print(f"데이터 행 수: {n_rows}")
    print(f"X 특성 개수: {n_features}")
    print(f"Y 대상 열: {target_column}")
    print(f"관측 기간: {dates[0].date()} ~ {dates[-1].date()}")
    print(f"학습 구간: {split_index}행")
    print(f"검증 구간: {n_rows - split_index}행")

    # -----------------------------------------------------
    # 1. 전체 파이프라인의 시간 순서 검증
    # -----------------------------------------------------

    metrics = validate_pipeline(
        X_raw=X_raw,
        y_raw=y_raw,
        dates=dates,
        split_index=split_index,
        x_columns=x_columns,
        lookback=lookback,
        epochs=epochs,
        batch_size=batch_size,
    )

    print("\n[재귀 예측 기반 검증 결과]")
    print(f"X MAE  : {metrics['x_mae']:.6f}")
    print(f"X RMSE : {metrics['x_rmse']:.6f}")
    print(f"Y MAE  : {metrics['y_mae']:.6f}")
    print(f"Y RMSE : {metrics['y_rmse']:.6f}")

    # -----------------------------------------------------
    # 2. 최종 모델 재학습
    # 검증이 끝나면 전체 관측 데이터를 사용한다.
    # -----------------------------------------------------

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_scaled = x_scaler.fit_transform(X_raw)
    y_scaled = y_scaler.fit_transform(y_raw)

    X_x, y_x = create_x_sequences(
        X_scaled,
        lookback
    )

    X_y, y_y = create_y_sequences(
        X_scaled,
        y_scaled,
        lookback
    )

    final_x_model = build_x_model(lookback, n_features)
    final_y_model = build_y_model(lookback, n_features)

    print("\n최종 X 모델 학습 중...")

    fit_model(
        final_x_model,
        X_x,
        y_x,
        epochs=epochs,
        batch_size=batch_size
    )

    print("최종 Y 모델 학습 중...")

    fit_model(
        final_y_model,
        X_y,
        y_y,
        epochs=epochs,
        batch_size=batch_size
    )

    # -----------------------------------------------------
    # 3. 모델 및 메타데이터 저장
    # -----------------------------------------------------

    model_id = make_model_id(csv_path, data_root)
    model_dir = output_root / model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    final_x_model.save(model_dir / "x_model.keras")
    final_y_model.save(model_dir / "y_model.keras")

    metadata = {
        "model_id": model_id,
        "relative_csv_path": str(
            csv_path.relative_to(data_root)
        ),
        "x_columns": x_columns,
        "target_column": target_column,
        "date_column": data["date_column"],
        "start_date": start_date,
        "last_observed_date": str(dates[-1].date()),
        "lookback": lookback,
        "n_features": n_features,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "validation_metrics": metrics,
    }

    with open(model_dir / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"\n저장 완료: {model_dir}")
    print(f"모델 ID: {model_id}")

    return model_id, metrics


def main():
    parser = argparse.ArgumentParser(
        description="6_dataset의 모든 LSTM_*.csv 학습"
    )

    parser.add_argument(
        "--data-dir",
        default="6_dataset",
        help="데이터 폴더"
    )

    parser.add_argument(
        "--output-dir",
        default="saved_models",
        help="학습 모델 저장 폴더"
    )

    parser.add_argument(
        "--start-date",
        default=None,
        help=(
            "날짜 열이 없는 CSV의 첫 행 날짜. "
            "예: 2016-01-01"
        )
    )

    parser.add_argument(
        "--date-column",
        default=None,
        help="날짜 열 이름을 명시할 때 사용"
    )

    parser.add_argument(
        "--lookback",
        type=int,
        default=LOOKBACK
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32
    )

    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2
    )

    args = parser.parse_args()

    if args.lookback < 1:
        raise ValueError("--lookback은 1 이상이어야 합니다.")

    if not 0.05 <= args.validation_fraction < 0.5:
        raise ValueError(
            "--validation-fraction은 0.05 이상 0.5 미만이어야 합니다."
        )

    data_root = Path(args.data_dir).resolve()
    output_root = Path(args.output_dir).resolve()

    if not data_root.exists():
        raise FileNotFoundError(
            f"데이터 폴더를 찾을 수 없습니다: {data_root}"
        )

    # 모든 하위 폴더에서 LSTM_으로 시작하는 CSV 검색
    csv_files = sorted(
        path for path in data_root.rglob("LSTM_*.csv")
        if path.is_file()
    )

    if not csv_files:
        print(f"학습 대상 CSV가 없습니다: {data_root}")
        return

    print(f"검색된 CSV 개수: {len(csv_files)}")
    print(f"모델 저장 위치: {output_root}")

    success_count = 0
    failed_files = []

    for csv_path in csv_files:
        try:
            train_one_csv(
                csv_path=csv_path,
                data_root=data_root,
                output_root=output_root,
                start_date=args.start_date,
                date_column=args.date_column,
                lookback=args.lookback,
                epochs=args.epochs,
                batch_size=args.batch_size,
                validation_fraction=args.validation_fraction,
            )

            success_count += 1

        except Exception as exc:
            print(f"\n[학습 실패] {csv_path}")
            print(f"원인: {exc}")
            traceback.print_exc()
            failed_files.append((csv_path, str(exc)))

    print("\n" + "=" * 70)
    print("전체 학습 작업 종료")
    print(f"성공: {success_count}")
    print(f"실패: {len(failed_files)}")

    if failed_files:
        print("\n실패한 파일 목록:")

        for path, reason in failed_files:
            print(f"- {path}")
            print(f"  원인: {reason}")


if __name__ == "__main__":
    main()