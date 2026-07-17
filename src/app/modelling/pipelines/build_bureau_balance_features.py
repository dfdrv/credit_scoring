import sys
from pathlib import Path
import argparse

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from app.modelling.features.bureau_balance_features import (
    generate_bureau_balance_features,
)


def read_raw_parquet(path: Path) -> pd.DataFrame:
    """
    Читает parquet-файл и приводит названия колонок к верхнему регистру.
    """
    df = pd.read_parquet(path)
    df.columns = df.columns.str.upper()
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Расчет признаков по домену bureau_balance из raw parquet-файлов"
    )
    parser.add_argument(
        "--bureau-balance-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "bureau_balance.parquet",
    )
    parser.add_argument(
        "--bureau-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "bureau.parquet",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "features" / "bureau_balance_features.parquet",
    )
    args = parser.parse_args()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    bureau_balance_df = read_raw_parquet(args.bureau_balance_path)
    bureau_df = read_raw_parquet(args.bureau_path)

    bureau_key_df = bureau_df[["SK_ID_BUREAU", "SK_ID_CURR"]].copy()

    merged_df = bureau_balance_df.merge(
        bureau_key_df,
        on="SK_ID_BUREAU",
        how="left",
    )

    features = generate_bureau_balance_features(merged_df).reset_index()
    features.to_parquet(args.output_path, index=False)

    print(f"Признаки bureau_balance сохранены: {args.output_path}")
    print(f"Размер таблицы признаков: {features.shape}")


if __name__ == "__main__":
    main()
