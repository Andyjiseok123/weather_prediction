
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lightgbm import LGBMRegressor, LGBMClassifier, early_stopping, log_evaluation
from catboost import CatBoostRegressor, CatBoostClassifier

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    f1_score,
    log_loss,
    average_precision_score,
    roc_auc_score,
)

# ============================================================
# 1. 사용자 설정 변수: 필요할 때 이 부분만 수정
# ============================================================

# 검증 데이터와 테스트 데이터의 기간(일)
VALIDATION_DAYS = 30
TEST_DAYS = 30

# 조기 종료 설정
EARLY_STOPPING_ROUNDS = 200
MAX_ITERATIONS = 2000

# 재현성을 위한 난수 시드
RANDOM_STATE = 42

# 입력 데이터 파일명
DATASET_FILENAME = "df_weather_강릉_30day_dataset.csv"

# 결과 저장 폴더명
OUTPUT_DIRNAME = "weather_training_results2"

# 모델 학습에 사용할 Target
REGRESSION_TARGETS = [
    "Y_WS_AVG",
    "Y_TA_AVG",
    "Y_TA_MAX",
    "Y_TA_MIN",
    "Y_RN_DAY",
]

CLASSIFICATION_TARGETS = [
    "Y_RAIN",
]

# ============================================================
# 2. 경로 및 기본 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / DATASET_FILENAME
OUTPUT_DIR = BASE_DIR / OUTPUT_DIRNAME
MODEL_DIR = OUTPUT_DIR / "models"
LOSS_DIR = OUTPUT_DIR / "loss_curves"

for directory in [OUTPUT_DIR, MODEL_DIR, LOSS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

META_COLUMNS = [
    "ORIGIN_DATE",
    "TARGET_DATE",
    "STN",
    "HORIZON_DAYS",
]

ALL_TARGETS = REGRESSION_TARGETS + CLASSIFICATION_TARGETS

# ============================================================
# 3. 데이터 불러오기 및 검사
# ============================================================

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"학습 데이터 파일을 찾을 수 없습니다: {DATASET_PATH}"
    )

df = pd.read_csv(DATASET_PATH)

required_columns = META_COLUMNS + ALL_TARGETS

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"필수 컬럼이 없습니다: {missing_columns}"
    )

df["ORIGIN_DATE"] = pd.to_datetime(
    df["ORIGIN_DATE"], errors="coerce"
)
df["TARGET_DATE"] = pd.to_datetime(
    df["TARGET_DATE"], errors="coerce"
)

df = df.dropna(
    subset=["ORIGIN_DATE", "TARGET_DATE"]
).copy()

# 날짜 순서대로 정렬
df = df.sort_values(
    ["ORIGIN_DATE", "STN"]
).reset_index(drop=True)

# Feature는 메타데이터와 정답 컬럼을 제외한 나머지
FEATURE_COLUMNS = [
    col for col in df.columns
    if col not in META_COLUMNS + ALL_TARGETS
]

if not FEATURE_COLUMNS:
    raise ValueError("입력 Feature가 없습니다.")

# 입력 Feature를 수치형으로 변환
# LightGBM과 CatBoost는 수치형 NaN을 자체 처리할 수 있음
for col in FEATURE_COLUMNS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Label은 숫자형이어야 함
for col in ALL_TARGETS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 정답이 없는 행 제거
df = df.dropna(subset=ALL_TARGETS).copy()

if df.empty:
    raise ValueError("Label 결측치 제거 후 학습 데이터가 없습니다.")

print("=" * 65)
print("데이터 로딩 완료")
print("=" * 65)
print(f"데이터 크기: {df.shape}")
print(f"입력 Feature 수: {len(FEATURE_COLUMNS)}")
print(f"회귀 Target 수: {len(REGRESSION_TARGETS)}")
print(f"분류 Target 수: {len(CLASSIFICATION_TARGETS)}")
print(f"전체 Target 수: {len(ALL_TARGETS)}")
print(f"Origin 날짜 범위: {df['ORIGIN_DATE'].min().date()} ~ "
      f"{df['ORIGIN_DATE'].max().date()}")
print(f"결측치가 있는 Feature 수: "
      f"{int(df[FEATURE_COLUMNS].isna().any().sum())}")

# ============================================================
# 4. 시계열 분할
# ============================================================

if VALIDATION_DAYS < 1 or TEST_DAYS < 1:
    raise ValueError(
        "VALIDATION_DAYS와 TEST_DAYS는 1 이상이어야 합니다."
    )

# 같은 날짜의 여러 관측소가 서로 다른 데이터셋으로
# 분리되지 않도록 고유 날짜를 기준으로 분할
unique_dates = np.array(
    sorted(df["ORIGIN_DATE"].dropna().unique())
)

required_days = VALIDATION_DAYS + TEST_DAYS

if len(unique_dates) <= required_days:
    raise ValueError(
        "학습 데이터가 너무 적습니다. "
        "검증·테스트 기간을 줄이거나 데이터를 확인하세요."
    )

test_start_idx = len(unique_dates) - TEST_DAYS
validation_start_idx = test_start_idx - VALIDATION_DAYS

validation_start_date = pd.Timestamp(
    unique_dates[validation_start_idx]
)
test_start_date = pd.Timestamp(
    unique_dates[test_start_idx]
)

train_mask = df["ORIGIN_DATE"] < validation_start_date

validation_mask = (
    (df["ORIGIN_DATE"] >= validation_start_date)
    & (df["ORIGIN_DATE"] < test_start_date)
)

test_mask = df["ORIGIN_DATE"] >= test_start_date

train_df = df.loc[train_mask].copy()
validation_df = df.loc[validation_mask].copy()
test_df = df.loc[test_mask].copy()

if train_df.empty or validation_df.empty or test_df.empty:
    raise ValueError("학습·검증·테스트 중 비어 있는 데이터셋이 있습니다.")

# 시계열 순서가 올바른지 확인
assert (
    train_df["ORIGIN_DATE"].max()
    < validation_df["ORIGIN_DATE"].min()
)
assert (
    validation_df["ORIGIN_DATE"].max()
    < test_df["ORIGIN_DATE"].min()
)

print("\n" + "=" * 65)
print("시계열 분할 결과")
print("=" * 65)

print(
    f"학습: {len(train_df):,}행 | "
    f"{train_df['ORIGIN_DATE'].min().date()} ~ "
    f"{train_df['ORIGIN_DATE'].max().date()}"
)

print(
    f"검증: {len(validation_df):,}행 | "
    f"{validation_df['ORIGIN_DATE'].min().date()} ~ "
    f"{validation_df['ORIGIN_DATE'].max().date()}"
)

print(
    f"테스트: {len(test_df):,}행 | "
    f"{test_df['ORIGIN_DATE'].min().date()} ~ "
    f"{test_df['ORIGIN_DATE'].max().date()}"
)

print(
    f"설정 기간: 검증 {VALIDATION_DAYS}일 / "
    f"테스트 {TEST_DAYS}일"
)

# ============================================================
# 5. 공통 학습 데이터 준비
# ============================================================

X_train = train_df[FEATURE_COLUMNS]
X_validation = validation_df[FEATURE_COLUMNS]
X_test = test_df[FEATURE_COLUMNS]

results = []
loss_records = []

# ============================================================
# 6. Loss 그래프 저장 함수
# ============================================================

def save_loss_curve(
    model_name,
    target_name,
    history,
    train_key,
    validation_key,
    metric_name,
):
    """
    학습 및 검증 loss를 그래프로 저장한다.
    history는 반복 횟수별 metric 값의 딕셔너리다.
    """
    if train_key not in history or validation_key not in history:
        print(
            f"[경고] Loss 그래프 생성 생략: "
            f"{model_name} / {target_name}"
        )
        return

    train_values = history[train_key].get(metric_name, [])
    validation_values = history[validation_key].get(
        metric_name, []
    )

    if not train_values or not validation_values:
        return

    plt.figure(figsize=(10, 5))
    plt.plot(
        range(1, len(train_values) + 1),
        train_values,
        label="Train",
    )
    plt.plot(
        range(1, len(validation_values) + 1),
        validation_values,
        label="Validation",
    )
    plt.xlabel("Iteration")
    plt.ylabel(metric_name)
    plt.title(f"{model_name} - {target_name} - {metric_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filename = f"{model_name}_{target_name}_loss.png"
    plt.savefig(LOSS_DIR / filename, dpi=150)
    plt.close()

# ============================================================
# 7. 회귀 모델 학습 및 평가
# ============================================================

def train_regression_model(
    model_name,
    target_name,
    y_train,
    y_validation,
    y_test,
):
    print(f"\n[{model_name}] {target_name} 학습 시작")

    if model_name == "LightGBM":
        model = LGBMRegressor(
            objective="regression",
            n_estimators=MAX_ITERATIONS,
            learning_rate=0.03,
            num_leaves=31,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[
                (X_train, y_train),
                (X_validation, y_validation),
            ],
            eval_names=["training", "validation"],
            eval_metric="rmse",
            callbacks=[
                early_stopping(
                    EARLY_STOPPING_ROUNDS,
                    verbose=False,
                ),
                log_evaluation(0),
            ],
        )

        history = model.evals_result_
        train_key = "training"
        validation_key = "validation"
        metric_name = "rmse"

        save_loss_curve(
            model_name,
            target_name,
            history,
            train_key,
            validation_key,
            metric_name,
        )
        best_iteration = model.best_iteration_ or len(
            history[validation_key][metric_name]
        )
        best_idx = max(0, best_iteration - 1)
        train_loss = history[train_key][metric_name][best_idx]
        validation_loss = history[validation_key][metric_name][best_idx]

    else:
        model = CatBoostRegressor(
            iterations=MAX_ITERATIONS,
            learning_rate=0.03,
            depth=6,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=(X_validation, y_validation),
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose=False,
        )

        history = model.get_evals_result()
        train_key = "learn"
        validation_key = "validation"
        metric_name = "RMSE"

        save_loss_curve(
            model_name,
            target_name,
            history,
            train_key,
            validation_key,
            metric_name,
        )

        train_loss = history[train_key][metric_name][-1]
        validation_loss = history[validation_key][metric_name][-1]
        best_iteration = model.get_best_iteration()

    # 테스트 데이터 예측
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    model_path = MODEL_DIR / f"{model_name}_{target_name}.cbm"

    if model_name == "CatBoost":
        model.save_model(str(model_path))
    else:
        # LightGBM 모델은 pickle로 저장
        import joblib
        model_path = MODEL_DIR / f"{model_name}_{target_name}.pkl"
        joblib.dump(model, model_path)

    result = {
        "Model": model_name,
        "Target": target_name,
        "Train_Loss": float(train_loss),
        "Validation_Loss": float(validation_loss),
        "Test_MAE": float(mae),
        "Test_RMSE": float(rmse),
        "Best_Iteration": int(best_iteration),
        "Model_Path": str(model_path),
    }

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Validation Loss: {validation_loss:.4f}"
    )
    print(f"Test MAE: {mae:.4f} | Test RMSE: {rmse:.4f}")
    print(f"Best Iteration: {best_iteration}")

    return model, result

# ============================================================
# 8. 분류 모델 학습 및 평가
# ============================================================

def train_classification_model(
    model_name,
    target_name,
    y_train,
    y_validation,
    y_test,
):
    print(f"\n[{model_name}] {target_name} 학습 시작")

    # 강수 여부가 한 종류만 있으면 분류 모델을 학습할 수 없음
    if y_train.nunique() < 2:
        print(
            f"[경고] 학습 데이터의 {target_name} 값이 "
            "한 종류뿐이므로 학습을 건너뜁니다."
        )
        return None, {
            "Model": model_name,
            "Target": target_name,
            "Train_Loss": np.nan,
            "Validation_Loss": np.nan,
            "Test_MAE": np.nan,
            "Test_RMSE": np.nan,
            "Test_F1": np.nan,
            "Test_LogLoss": np.nan,
            "Test_PR_AUC": np.nan,
            "Test_ROC_AUC": np.nan,
            "Best_Iteration": np.nan,
            "Model_Path": "",
        }

    if model_name == "LightGBM":
        model = LGBMClassifier(
            objective="binary",
            n_estimators=MAX_ITERATIONS,
            learning_rate=0.03,
            num_leaves=31,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        )

        model.fit(
            X_train,
            y_train.astype(int),
            eval_set=[
                (X_train, y_train.astype(int)),
                (X_validation, y_validation.astype(int)),
            ],
            eval_names=["training", "validation"],
            eval_metric="binary_logloss",
            callbacks=[
                early_stopping(
                    EARLY_STOPPING_ROUNDS,
                    verbose=False,
                ),
                log_evaluation(0),
            ],
        )

        history = model.evals_result_
        train_key = "training"
        validation_key = "validation"
        metric_name = "binary_logloss"

        save_loss_curve(
            model_name,
            target_name,
            history,
            train_key,
            validation_key,
            metric_name,
        )
        best_iteration = model.best_iteration_ or len(
            history[validation_key][metric_name]
        )
        best_idx = max(0, best_iteration - 1)
        train_loss = history[train_key][metric_name][best_idx]
        validation_loss = history[validation_key][metric_name][best_idx]

        model_path = MODEL_DIR / f"{model_name}_{target_name}.pkl"
        import joblib
        joblib.dump(model, model_path)

    else:
        model = CatBoostClassifier(
            iterations=MAX_ITERATIONS,
            learning_rate=0.03,
            depth=6,
            loss_function="Logloss",
            eval_metric="Logloss",
            random_seed=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
        )

        model.fit(
            X_train,
            y_train.astype(int),
            eval_set=(
                X_validation,
                y_validation.astype(int),
            ),
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose=False,
        )

        history = model.get_evals_result()
        train_key = "learn"
        validation_key = "validation"
        metric_name = "Logloss"

        save_loss_curve(
            model_name,
            target_name,
            history,
            train_key,
            validation_key,
            metric_name,
        )

        train_loss = history[train_key][metric_name][-1]
        validation_loss = history[validation_key][metric_name][-1]
        best_iteration = model.get_best_iteration()

        model_path = MODEL_DIR / f"{model_name}_{target_name}.cbm"
        model.save_model(str(model_path))

    # 테스트 예측
    y_pred = model.predict(X_test).reshape(-1).astype(int)
    y_probability = model.predict_proba(X_test)[:, 1]

    y_test_int = y_test.astype(int)

    test_logloss = log_loss(
        y_test_int,
        y_probability,
        labels=[0, 1],
    )

    test_f1 = f1_score(
        y_test_int,
        y_pred,
        zero_division=0,
    )

    # PR-AUC 및 ROC-AUC는 테스트 데이터에 양성·음성이 모두
    # 존재할 때만 계산 가능
    if y_test_int.nunique() == 2:
        test_pr_auc = average_precision_score(
            y_test_int,
            y_probability,
        )
        test_roc_auc = roc_auc_score(
            y_test_int,
            y_probability,
        )
    else:
        test_pr_auc = np.nan
        test_roc_auc = np.nan

    result = {
        "Model": model_name,
        "Target": target_name,
        "Train_Loss": float(train_loss),
        "Validation_Loss": float(validation_loss),
        "Test_F1": float(test_f1),
        "Test_LogLoss": float(test_logloss),
        "Test_PR_AUC": float(test_pr_auc)
        if not np.isnan(test_pr_auc) else np.nan,
        "Test_ROC_AUC": float(test_roc_auc)
        if not np.isnan(test_roc_auc) else np.nan,
        "Best_Iteration": int(best_iteration),
        "Model_Path": str(model_path),
    }

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Validation Loss: {validation_loss:.4f}"
    )
    print(
        f"Test LogLoss: {test_logloss:.4f} | "
        f"Test F1: {test_f1:.4f}"
    )
    print(
        f"Test PR-AUC: {test_pr_auc:.4f}"
        if not np.isnan(test_pr_auc)
        else "Test PR-AUC: 계산 불가 (테스트에 한 클래스만 존재)"
    )
    print(f"Best Iteration: {best_iteration}")

    return model, result

# ============================================================
# 9. 모든 Target에 대해 LightGBM / CatBoost 학습
# ============================================================

for target_name in REGRESSION_TARGETS:
    y_train = train_df[target_name]
    y_validation = validation_df[target_name]
    y_test = test_df[target_name]

    for model_name in ["LightGBM", "CatBoost"]:
        _, result = train_regression_model(
            model_name,
            target_name,
            y_train,
            y_validation,
            y_test,
        )
        results.append(result)

for target_name in CLASSIFICATION_TARGETS:
    y_train = train_df[target_name]
    y_validation = validation_df[target_name]
    y_test = test_df[target_name]

    for model_name in ["LightGBM", "CatBoost"]:
        _, result = train_classification_model(
            model_name,
            target_name,
            y_train,
            y_validation,
            y_test,
        )
        results.append(result)

# ============================================================
# 10. 결과 저장
# ============================================================

results_df = pd.DataFrame(results)

results_path = OUTPUT_DIR / "model_evaluation_results.csv"
results_df.to_csv(
    results_path,
    index=False,
    encoding="utf-8-sig",
)

print("\n" + "=" * 65)
print("전체 모델 학습 완료")
print("=" * 65)

display_columns = [
    col for col in [
        "Model",
        "Target",
        "Train_Loss",
        "Validation_Loss",
        "Test_MAE",
        "Test_RMSE",
        "Test_F1",
        "Test_LogLoss",
        "Test_PR_AUC",
        "Best_Iteration",
    ]
    if col in results_df.columns
]

print(results_df[display_columns].to_string(index=False))

print(f"\n평가 결과 저장: {results_path}")
print(f"모델 저장 폴더: {MODEL_DIR}")
print(f"Loss 그래프 폴더: {LOSS_DIR}")
