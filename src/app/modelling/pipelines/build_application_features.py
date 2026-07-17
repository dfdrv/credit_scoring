import sys
from pathlib import Path
import argparse

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from app.modelling.features.application_features import generate_application_features


def read_raw_parquet(path: Path) -> pd.DataFrame:
    """
    Читает parquet-файл и приводит названия колонок к верхнему регистру.
    """
    df = pd.read_parquet(path)
    df.columns = df.columns.str.upper()
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Расчет признаков по домену application из raw parquet-файлов"
    )
    parser.add_argument(
        "--application-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "application.parquet",
    )
    parser.add_argument(
        "--previous-application-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "previous_application.parquet",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "features" / "application_features.parquet",
    )
    args = parser.parse_args()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    app_df = read_raw_parquet(args.application_path)
    prev_df = read_raw_parquet(args.previous_application_path)

    features = generate_application_features(app_df, prev_df).reset_index()
    features.to_parquet(args.output_path, index=False)

    print(f"Признаки application сохранены: {args.output_path}")
    print(f"Размер таблицы признаков: {features.shape}")


if __name__ == "__main__":
    main()
