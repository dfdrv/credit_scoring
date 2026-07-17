from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MODEL_FEATURE_COLUMNS = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "DAYS_LAST_PHONE_CHANGE",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "OBS_30_CNT_SOCIAL_CIRCLE",
    "DEF_30_CNT_SOCIAL_CIRCLE",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "AMT_REQ_CREDIT_BUREAU_MON",
    "AMT_REQ_CREDIT_BUREAU_QRT",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
    "CODE_GENDER",
    "NAME_CONTRACT_TYPE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
    "credit_to_income_ratio",
    "annuity_to_income_ratio",
    "active_credit_count",
    "credit_history_days",
    "prev_application_count",
]

META_COLUMNS = ["SK_ID_CURR", "TARGET", "DATASET_SOURCE"]


def load_raw_tables(
    data_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Читает сырые таблицы, из которых собираем признаки"""

    data_dir = Path(data_dir)

    application_df = pd.read_parquet(data_dir / "application.parquet")
    bureau_df = pd.read_parquet(data_dir / "bureau.parquet")
    previous_application_df = pd.read_parquet(data_dir / "previous_application.parquet")

    return application_df, bureau_df, previous_application_df


def build_hw22_features(
    application_df: pd.DataFrame,
    bureau_df: pd.DataFrame,
    previous_application_df: pd.DataFrame,
) -> pd.DataFrame:
    """Собирает ручные признаки, которые использовались в HW2/HW6"""

    features_app = application_df[
        ["SK_ID_CURR", "AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY"]
    ].copy()

    income_nonzero = features_app["AMT_INCOME_TOTAL"].replace(0, np.nan)

    features_app["credit_to_income_ratio"] = (
        features_app["AMT_CREDIT"] / income_nonzero
    ).round(4)

    features_app["annuity_to_income_ratio"] = (
        features_app["AMT_ANNUITY"] / income_nonzero
    ).round(4)

    features_app = features_app[
        ["SK_ID_CURR", "credit_to_income_ratio", "annuity_to_income_ratio"]
    ]

    active_credit_count = (
        bureau_df.loc[bureau_df["CREDIT_ACTIVE"] == "Active", ["SK_ID_CURR"]]
        .groupby("SK_ID_CURR", as_index=False)
        .size()
        .rename(columns={"size": "active_credit_count"})
    )

    credit_history_days = (
        bureau_df.groupby("SK_ID_CURR", as_index=False)["DAYS_CREDIT"]
        .min()
        .rename(columns={"DAYS_CREDIT": "credit_history_days"})
    )
    credit_history_days["credit_history_days"] = (
        credit_history_days["credit_history_days"].abs()
    )

    prev_application_count = (
        previous_application_df.groupby("SK_ID_CURR", as_index=False)
        .size()
        .rename(columns={"size": "prev_application_count"})
    )

    features = features_app.merge(active_credit_count, on="SK_ID_CURR", how="left")
    features = features.merge(credit_history_days, on="SK_ID_CURR", how="left")
    features = features.merge(prev_application_count, on="SK_ID_CURR", how="left")

    fill_zero_cols = [
        "active_credit_count",
        "credit_history_days",
        "prev_application_count",
    ]
    features[fill_zero_cols] = features[fill_zero_cols].fillna(0)

    return features


def build_feature_frame(
    application_df: pd.DataFrame,
    bureau_df: pd.DataFrame,
    previous_application_df: pd.DataFrame,
) -> pd.DataFrame:
    """Собирает итоговый DataFrame признаков для модели"""

    hw22_features = build_hw22_features(
        application_df=application_df,
        bureau_df=bureau_df,
        previous_application_df=previous_application_df,
    )

    full_df = application_df.merge(hw22_features, on="SK_ID_CURR", how="left")

    missing_columns = sorted(set(MODEL_FEATURE_COLUMNS) - set(full_df.columns))
    if missing_columns:
        raise ValueError(
            "В итоговом DataFrame не хватает колонок для модели: "
            f"{missing_columns}"
        )

    return full_df[META_COLUMNS + MODEL_FEATURE_COLUMNS].copy()


def make_train_validation_split(
    feature_frame: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Делит train-часть на обучающую и валидационную выборки."""

    from sklearn.model_selection import train_test_split

    train_frame = feature_frame[feature_frame["DATASET_SOURCE"].eq("train")].copy()

    # Stratify нужен, чтобы доля дефолтов в обеих выборках осталась сопоставимой.
    train_part, valid_part = train_test_split(
        train_frame,
        test_size=test_size,
        random_state=random_state,
        stratify=train_frame["TARGET"].astype(int),
    )

    return train_part.reset_index(drop=True), valid_part.reset_index(drop=True)
