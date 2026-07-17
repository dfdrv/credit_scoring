import sys
from pathlib import Path


src_path = Path(__file__).resolve().parents[3]

if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

import numpy as np
import pandas as pd

from config.db_config import DB_ARGS
from app.utils.db_manager import PostgresDB


FEATURES_PATH = Path(__file__).resolve().parent
FEATURES_PATH.mkdir(parents=True, exist_ok=True)


def load_application(db: PostgresDB) -> pd.DataFrame:
    """
    Загружает объединенную таблицу application.

    """
    query = "SELECT * FROM application"
    df = db.get_df(query)
    df.columns = df.columns.str.upper()
    return df


def load_previous_application_rate(db: PostgresDB) -> pd.DataFrame:
    """
    Загружает данные previous_application для оценки процентной ставки.

    """
    query = """
    SELECT
        SK_ID_CURR,
        AMT_CREDIT,
        AMT_ANNUITY,
        CNT_PAYMENT
    FROM previous_application
    """
    df = db.get_df(query)
    df.columns = df.columns.str.upper()
    return df


def generate_application_features(
    app_df: pd.DataFrame,
    prev_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Генерирует признаки по домену application на уровне клиента (SK_ID_CURR).
    """
    # Переводим application в формат с ключом клиента в индексе,
    # чтобы все рассчитанные series корректно выравнивались по SK_ID_CURR
    app_df = app_df.copy().set_index("SK_ID_CURR")

    features = pd.DataFrame(index=app_df.index)

    # =========================================================
    # Количество документов
    # =========================================================
    doc_cols = [col for col in app_df.columns if col.startswith("FLAG_DOCUMENT_")]
    features["app_document_count"] = app_df[doc_cols].sum(axis=1)

    # =========================================================
    # Полная информация о доме
    # =========================================================
    house_cols = [
        col for col in app_df.columns
        if col.endswith(("_AVG", "_MEDI", "_MODE"))
    ]

    house_non_null_cnt = app_df[house_cols].notna().sum(axis=1)
    features["app_house_info_count"] = house_non_null_cnt
    features["app_house_info_full"] = (house_non_null_cnt >= 30).astype(int)

    # =========================================================
    # Количество полных лет
    # =========================================================
    age_years = (-app_df["DAYS_BIRTH"] / 365).astype(int)
    features["app_full_years"] = age_years

    # =========================================================
    # Сколько лет назад менялся документ
    # =========================================================
    doc_change_years_ago = -app_df["DAYS_ID_PUBLISH"] / 365
    features["app_doc_changed_years_ago_int"] = np.floor(doc_change_years_ago).astype(int)

    # =========================================================
    # Возраст клиента на момент смены документа
    # =========================================================
    age_when_doc_changed = age_years - doc_change_years_ago
    features["app_age_when_doc_changed_int"] = np.floor(age_when_doc_changed).astype(int)

    # =========================================================
    # Задержка смены документа
    # =========================================================
    features["app_document_change_delay_flag"] = (
        ~features["app_age_when_doc_changed"].astype(int).isin([14, 20, 45])
    ).astype(int)

    # =========================================================
    # Доля дохода, которую клиент отдает на займ
    # =========================================================
    features["app_credit_payment_income_share"] = (
        app_df["AMT_ANNUITY"] / app_df["AMT_INCOME_TOTAL"]
    )

    # =========================================================
    # Среднее количество детей на одного взрослого
    # =========================================================
    adults = app_df["CNT_FAM_MEMBERS"] - app_df["CNT_CHILDREN"]
    adults_safe = adults.replace(0, np.nan)

    features["app_children_per_adult"] = (
        app_df["CNT_CHILDREN"] / adults_safe
    )

    # =========================================================
    # Средний доход на ребенка
    # =========================================================
    features["app_income_per_child"] = (
        app_df["AMT_INCOME_TOTAL"] /
        app_df["CNT_CHILDREN"].replace(0, np.nan)
    )

    # =========================================================
    # Средний доход на взрослого
    # =========================================================
    features["app_income_per_adult"] = (
        app_df["AMT_INCOME_TOTAL"] / adults_safe
    )

    # =========================================================
    # Взвешенный скор внешних источников
    # Весовая схема:
    # EXT_SOURCE_2 -> 0.5
    # EXT_SOURCE_3 -> 0.3
    # EXT_SOURCE_1 -> 0.2
    # =========================================================
    features["app_weighted_ext_score"] = (
        0.5 * app_df["EXT_SOURCE_2"].fillna(0) +
        0.3 * app_df["EXT_SOURCE_3"].fillna(0) +
        0.2 * app_df["EXT_SOURCE_1"].fillna(0)
    )

    # =========================================================
    # Отклонение дохода от среднего по группе
    # считаем средний доход только на train,
    # чтобы не было утечки из test
    # =========================================================
    app_reset = app_df.reset_index()

    train_df = app_reset[app_reset["DATASET_SOURCE"] == "train"].copy()

    train_global_mean_income = train_df["AMT_INCOME_TOTAL"].mean()

    group_mean_income_df = (
        train_df.groupby(
            ["CODE_GENDER", "NAME_EDUCATION_TYPE"],
            as_index=False
        )["AMT_INCOME_TOTAL"]
        .mean()
        .rename(columns={"AMT_INCOME_TOTAL": "group_mean_income"})
    )

    app_with_group_mean = (
        app_reset.merge(
            group_mean_income_df,
            on=["CODE_GENDER", "NAME_EDUCATION_TYPE"],
            how="left"
        )
        .set_index("SK_ID_CURR")
    )

    # Если в test встретилась новая группа, которой не было в train,
    # используем глобальный средний доход по train
    app_with_group_mean["group_mean_income"] = (
        app_with_group_mean["group_mean_income"]
        .fillna(train_global_mean_income)
    )

    features["app_income_vs_group_mean_diff"] = (
        app_with_group_mean["AMT_INCOME_TOTAL"] -
        app_with_group_mean["group_mean_income"]
    )

    # =========================================================
    # Приближенная процентная ставка
    # =========================================================
    prev_df = prev_df.copy()

    total_payment = prev_df["AMT_ANNUITY"] * prev_df["CNT_PAYMENT"]

    prev_df["approx_interest_rate"] = (
        (total_payment - prev_df["AMT_CREDIT"]) /
        prev_df["AMT_CREDIT"]
    )

    rate_features = (
        prev_df.groupby("SK_ID_CURR")["approx_interest_rate"]
        .agg(
            app_rate_mean="mean",
            app_rate_max="max",
            app_rate_min="min"
        )
    )

    features = features.join(rate_features)

    return features



def save_features(df: pd.DataFrame):
    """
    Сохраняет признаки.

    :param df: feature dataframe
    """
    output_path = FEATURES_PATH / "application_features.parquet"

    df.to_parquet(output_path)

    print(f"Файл сохранен: {output_path}")
    print(f"Размер: {df.shape}")


def main():
    
    db = PostgresDB(DB_ARGS)
    db.connect()

    try:
        app_df = load_application(db)
        prev_df = load_previous_application_rate(db)

        features = generate_application_features(app_df, prev_df)
        save_features(features)

    finally:
        db.close()

if __name__ == "__main__":
    
    main()
    