from __future__ import annotations

import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from app.core.feature_frame import (  # noqa: E402
    build_feature_frame,
    load_raw_tables,
    make_train_validation_split,
)
from app.core.model import GradientBoostingScoringModel  # noqa: E402
from app.core.validation import (  # noqa: E402
    build_policy_summary,
    build_probability_band_summary,
    score_validation_frame,
)


def main() -> None:
    """Собирает все артефакты для ядра скоринга поверх HW6"""

    artifacts_dir = PROJECT_ROOT / "artifacts" / "hw6_core"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    raw_data_dir = PROJECT_ROOT / "data" / "raw"
    model_source_path = (
        PROJECT_ROOT / "notebooks" / "hw6" / "models" / "gradient_boosting.pkl"
    )
    model_target_path = artifacts_dir / "gradient_boosting.pkl"

    # Копируем pickle отдельно в папку артефактов,
    # чтобы итог одного запуска лежал в одном месте
    shutil.copy2(model_source_path, model_target_path)

    application_df, bureau_df, previous_application_df = load_raw_tables(raw_data_dir)
    feature_frame = build_feature_frame(
        application_df=application_df,
        bureau_df=bureau_df,
        previous_application_df=previous_application_df,
    )

    feature_frame_path = artifacts_dir / "application_scoring_features.parquet"
    feature_frame.to_parquet(feature_frame_path, index=False)

    _, validation_frame = make_train_validation_split(
        feature_frame=feature_frame,
        test_size=0.2,
        random_state=42,
    )

    validation_features_path = artifacts_dir / "validation_features.parquet"
    validation_frame.to_parquet(validation_features_path, index=False)

    model_wrapper = GradientBoostingScoringModel(model_target_path)

    scored_validation = score_validation_frame(
        model_wrapper=model_wrapper,
        validation_frame=validation_frame,
    )

    scored_validation_path = artifacts_dir / "validation_scored.parquet"
    scored_validation.to_parquet(scored_validation_path, index=False)

    policy_summary = build_policy_summary(
        scored_validation=scored_validation,
        model_wrapper=model_wrapper,
    )
    policy_summary_path = artifacts_dir / "policy_summary.csv"
    policy_summary.to_csv(policy_summary_path, index=False)

    probability_band_summary = build_probability_band_summary(
        scored_validation=scored_validation,
        model_wrapper=model_wrapper,
    )
    probability_band_summary_path = artifacts_dir / "probability_band_summary.csv"
    probability_band_summary.to_csv(probability_band_summary_path, index=False)

    print("Артефакты успешно подготовлены:")
    print(f"  pickle модели: {model_target_path}")
    print(f"  полный DataFrame признаков: {feature_frame_path}")
    print(f"  признаки для validation: {validation_features_path}")
    print(f"  результат скоринга validation: {scored_validation_path}")
    print(f"  сводка по policy: {policy_summary_path}")
    print(f"  сводка по probability-бэндам: {probability_band_summary_path}")


if __name__ == "__main__":
    main()
