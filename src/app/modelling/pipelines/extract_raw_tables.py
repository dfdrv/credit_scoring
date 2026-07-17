import sys
from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from config.db_config import DB_ARGS
from app.utils.db_manager import PostgresDB


TABLE_QUERIES = {
    "application": "SELECT * FROM application",
    "previous_application": "SELECT * FROM previous_application",
    "bureau": "SELECT * FROM bureau",
    "bureau_balance": "SELECT * FROM bureau_balance",
}


def dump_table(db: PostgresDB, table_name: str, output_dir: Path) -> None:
    """
    Выгружает одну таблицу из PostgreSQL и сохраняет ее в parquet.
    """
    query = TABLE_QUERIES[table_name]
    df = db.get_df(query)
    df.columns = df.columns.str.upper()

    output_path = output_dir / f"{table_name}.parquet"
    df.to_parquet(output_path, index=False)

    print(f"Таблица {table_name} сохранена: {output_path}")
    print(f"Размер таблицы: {df.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Выгрузка базовых таблиц из PostgreSQL в data/raw/"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Папка для сохранения parquet-файлов с сырыми данными",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    db = PostgresDB(DB_ARGS)
    db.connect()

    try:
        for table_name in TABLE_QUERIES:
            dump_table(db, table_name, args.output_dir)
    finally:
        db.close()


if __name__ == "__main__":
    main()
