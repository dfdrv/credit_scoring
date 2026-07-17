import sys
from pathlib import Path

# Добавляем корень проекта в sys.path для корректных импортов
src_path = Path(__file__).resolve().parents[3]

if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

import pandas as pd

from config.db_config import DB_ARGS
from app.utils.db_manager import PostgresDB


# Папка для сохранения итоговых признаков
FEATURES_PATH = Path(__file__).resolve().parent
FEATURES_PATH.mkdir(parents=True, exist_ok=True)


def load_bureau_balance(db: PostgresDB) -> pd.DataFrame:
    """
    Загружает данные из bureau_balance и подтягивает SK_ID_CURR из bureau.

    """
    query = """
    SELECT
        bb.SK_ID_BUREAU,
        b.SK_ID_CURR,
        bb.MONTHS_BALANCE,
        bb.STATUS
    FROM bureau_balance bb
    LEFT JOIN bureau b
        ON bb.SK_ID_BUREAU = b.SK_ID_BUREAU
    """
    df = db.get_df(query)
    df.columns = df.columns.str.upper()
    return df


def generate_bureau_balance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Генерирует признаки по домену bureau_balance на уровне клиента (SK_ID_CURR).

    """
    # Убираем записи без SK_ID_CURR, если такие появились 
    df = df[df["SK_ID_CURR"].notna()].copy()

    # =========================================================
    # Последний актуальный статус по каждому кредиту
    # Чем больше MONTHS_BALANCE, тем ближе запись к текущему моменту
    # =========================================================
    last_status_df = (
        df.sort_values(["SK_ID_BUREAU", "MONTHS_BALANCE"])
        .groupby("SK_ID_BUREAU", as_index=False)
        .tail(1)
        .copy()
    )

    # =========================================================
    # Количество кредитов в разных статусах
    # =========================================================
    status_cnt = (
        last_status_df.groupby("SK_ID_CURR")["STATUS"]
        .value_counts()
        .unstack(fill_value=0)
    )

    # Если какого-то статуса нет в данных клиента, добавляем нулевой столбец
    expected_statuses = ["C", "0", "1", "2", "3", "4", "5", "X"]
    status_cnt = status_cnt.reindex(columns=expected_statuses, fill_value=0)


    status_cnt = status_cnt.rename(columns={
        "C": "bb_closed_credit_cnt",
        "0": "bb_open_credit_cnt",
        "1": "bb_dpd_1_credit_cnt",
        "2": "bb_dpd_2_credit_cnt",
        "3": "bb_dpd_3_credit_cnt",
        "4": "bb_dpd_4_credit_cnt",
        "5": "bb_dpd_5_credit_cnt",
        "X": "bb_unknown_status_credit_cnt",
    })

    # =========================================================
    # Общее количество кредитов
    # =========================================================
    total_credit_cnt = (
        last_status_df.groupby("SK_ID_CURR")["SK_ID_BUREAU"]
        .nunique()
        .rename("bb_total_credit_cnt")
    )

    features = status_cnt.join(total_credit_cnt)

    # =========================================================
    # Доли кредитов по статусам
    # =========================================================
    status_share = status_cnt.divide(total_credit_cnt, axis=0)
    status_share = status_share.rename(
        columns=lambda col: col.replace("_cnt", "_share")
    )
    features = features.join(status_share)

    # =========================================================
    # Интервал до последнего закрытого кредита
    # Берем максимум MONTHS_BALANCE среди кредитов с текущим статусом C
    # =========================================================
    last_closed_interval = (
        df[df["STATUS"] == "C"]
        .groupby(["SK_ID_CURR", "SK_ID_BUREAU"], as_index=False)[["MONTHS_BALANCE"]]
        .min()
        .groupby("SK_ID_CURR")["MONTHS_BALANCE"]
        .max()
        .rename("bb_last_closed_interval")
    )

    features = features.join(last_closed_interval)

    # =========================================================
    # Интервал до последнего активного кредита
    # Активными считаем кредиты со статусом 0
    # =========================================================
    active_loans_id = last_status_df.loc[
        last_status_df["STATUS"] <= "5",
        "SK_ID_BUREAU"
    ]

    last_active_interval = (
        df[df["SK_ID_BUREAU"].isin(active_loans_id)]
        .groupby(["SK_ID_CURR", "SK_ID_BUREAU"], as_index=False)[["MONTHS_BALANCE"]]
        .min()
        .groupby("SK_ID_CURR")["MONTHS_BALANCE"]
        .max()
        .rename("bb_last_active_interval")
    )

    features = features.join(last_active_interval)

    return features.fillna(0)


def save_features(df: pd.DataFrame) -> None:
    """
    Сохраняет рассчитанные признаки в parquet файл.

    """
    output_path = FEATURES_PATH / "bureau_balance_features.parquet"
    df.to_parquet(output_path)

    print(f"Файл с признаками сохранен: {output_path}")
    print(f"Размер таблицы признаков: {df.shape}")


def main():
    """
    Основная функция:
    - подключение к БД
    - загрузка данных bureau_balance
    - генерация признаков
    - сохранение результата
    """
    db = PostgresDB(DB_ARGS)
    db.connect()

    try:
        bureau_balance_df = load_bureau_balance(db)

        print("Колонки bureau_balance:")
        print(bureau_balance_df.columns.tolist())

        features = generate_bureau_balance_features(bureau_balance_df)

        print("\nПример рассчитанных признаков:")
        print(features.sample(min(5, len(features))))

        save_features(features)

    finally:
        db.close()


if __name__ == "__main__":
    main()