import sys
from pathlib import Path

src_path = Path(__file__).resolve().parents[3]

if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

import pandas as pd

from config.db_config import DB_ARGS
from app.utils.db_manager import PostgresDB


# Папка для сохранения итоговых признаков
FEATURES_PATH = Path(__file__).resolve().parent
FEATURES_PATH.mkdir(parents=True, exist_ok=True)


def load_bureau(db: PostgresDB) -> pd.DataFrame:
    """
    Загружает необходимые поля из таблицы bureau.

    """
    query = """
    SELECT
        SK_ID_CURR,
        CREDIT_TYPE,
        CREDIT_ACTIVE,
        AMT_CREDIT_SUM,
        AMT_CREDIT_SUM_OVERDUE
    FROM bureau
    """
    df = db.get_df(query)

    # Приводим названия столбцов к единому формату
    df.columns = df.columns.str.upper()

    return df


def generate_bureau_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Генерирует признаки по домену bureau на уровне клиента (SK_ID_CURR).

    """
    # Создаем базовый датафрейм признаков с индексом клиента
    features = pd.DataFrame(index=df["SK_ID_CURR"].unique())
    features.index.name = "SK_ID_CURR"
    # подсказка преподавателя
    # надо посчитать на данный момент, а не в момент подачи заявки
    # =========================================================
    # Максимальная и минимальная текущая просрочка
    # =========================================================
    overdue_stats = (
        df.groupby("SK_ID_CURR")["AMT_CREDIT_SUM_OVERDUE"]
        .agg(
            bureau_max_overdue="max",
            bureau_min_overdue="min"
        )
    )

    features = features.join(overdue_stats)
    # подсказка преподавателя
    # Для каждого открытого sk_id_curr необходимо посчитать сумму, которая должна быть внесена по кредиту, а также сумму просрочки.
    # =========================================================
    # Доля просрочки по активным кредитам
    # =========================================================
    active_df = df[df["CREDIT_ACTIVE"] == "Active"].copy()

    overdue_share = (
        active_df.groupby("SK_ID_CURR")
        .agg(
            active_overdue_sum=("AMT_CREDIT_SUM_OVERDUE", "sum"),
            active_credit_sum=("AMT_CREDIT_SUM", "sum")
        )
    )

    overdue_share["bureau_active_overdue_share"] = (
        overdue_share["active_overdue_sum"]
        / overdue_share["active_credit_sum"]
    )

    features = features.join(
        overdue_share[["bureau_active_overdue_share"]]
    )

    # =========================================================
    # Количество кредитов каждого типа
    # =========================================================
    credit_type_cnt = (
        df.groupby(["SK_ID_CURR", "CREDIT_TYPE"])
        .size()
        .unstack(fill_value=0)
        .add_prefix("bureau_credit_cnt_")
    )

    features = features.join(credit_type_cnt)

    # =========================================================
    # Количество просроченных кредитов каждого типа
    # =========================================================
    overdue_df = df[df["AMT_CREDIT_SUM_OVERDUE"] > 0]

    overdue_type_cnt = (
        overdue_df.groupby(["SK_ID_CURR", "CREDIT_TYPE"])
        .size()
        .unstack(fill_value=0)
        .add_prefix("bureau_overdue_credit_cnt_")
    )

    features = features.join(overdue_type_cnt)

    # =========================================================
    # Количество закрытых кредитов каждого типа
    # =========================================================
    closed_df = df[df["CREDIT_ACTIVE"] == "Closed"]

    closed_type_cnt = (
        closed_df.groupby(["SK_ID_CURR", "CREDIT_TYPE"])
        .size()
        .unstack(fill_value=0)
        .add_prefix("bureau_closed_credit_cnt_")
    )

    features = features.join(closed_type_cnt)

    return features.fillna(0)


def save_features(df: pd.DataFrame):
    """
    Сохраняет рассчитанные признаки в parquet файл.

    """
    output_path = FEATURES_PATH / "bureau_features.parquet"

    df.to_parquet(output_path)

    print(f"Файл с признаками сохранен: {output_path}")
    print(f"Размер таблицы признаков: {df.shape}")


def main():
    """
    Основная функция:
    - подключение к БД
    - загрузка данных
    - генерация признаков
    - сохранение результата
    """
    db = PostgresDB(DB_ARGS)
    db.connect()

    try:
        bureau_df = load_bureau(db)

        print("Колонки bureau:")
        print(bureau_df.columns.tolist())

        features = generate_bureau_features(bureau_df)

        save_features(features)


    finally:
        db.close()


if __name__ == "__main__":
    main()