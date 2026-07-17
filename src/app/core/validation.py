from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.feature_frame import MODEL_FEATURE_COLUMNS


DISPLAY_COLUMNS = [
    "SK_ID_CURR",
    "TARGET",
    "DATASET_SOURCE",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "credit_to_income_ratio",
    "annuity_to_income_ratio",
]


def score_validation_frame(
    model_wrapper,
    validation_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Скорит validation и проверяет совпадение wrapper с raw model"""

    validation_frame = validation_frame.reset_index(drop=True).copy()

    # Здесь считаем вероятности в лоб через исходную модель.
    direct_proba = model_wrapper.raw_model.predict_proba(
        validation_frame[MODEL_FEATURE_COLUMNS]
    )[:, 1]

    # Идея в том, что proba должна совпасть,
    # а поверх неё уже добавятся решение и сумма.
    wrapped_scored = model_wrapper.score_batch(validation_frame[MODEL_FEATURE_COLUMNS])

    result = pd.concat(
        [
            validation_frame[DISPLAY_COLUMNS].reset_index(drop=True),
            wrapped_scored.reset_index(drop=True),
        ],
        axis=1,
    )

    result["direct_proba"] = direct_proba
    result["abs_diff"] = (result["proba"] - result["direct_proba"]).abs()

    if not np.allclose(result["proba"], result["direct_proba"], atol=1e-12):
        raise AssertionError(
            "Вероятности из ядра не совпали с прямым predict_proba() модели."
        )

    margin_rate = model_wrapper.calculator.policy.margin_rate

    # при дефолте теряем всю выданную сумму, при возврате зарабатываем margin_rate.
    result["realized_income"] = np.where(
        result["approved_amount"].eq(0),
        0.0,
        np.where(
            result["TARGET"].eq(1),
            -result["approved_amount"],
            result["approved_amount"] * margin_rate,
        ),
    )

    return result


def build_policy_summary(
    scored_validation: pd.DataFrame,
    model_wrapper,
) -> pd.DataFrame:
    """Строит короткую сводку по validation и policy"""

    approved_mask = scored_validation["approved_amount"] > 0
    defaulted_mask = approved_mask & scored_validation["TARGET"].eq(1)
    returned_mask = approved_mask & scored_validation["TARGET"].eq(0)

    returned_principal = float(
        scored_validation.loc[returned_mask, "approved_amount"].sum()
    )
    not_returned_amount = float(
        scored_validation.loc[defaulted_mask, "approved_amount"].sum()
    )
    profit_on_returned = returned_principal * model_wrapper.calculator.policy.margin_rate
    total_realized_income = profit_on_returned - not_returned_amount

    rows = [
        ("validation_rows", int(len(scored_validation))),
        ("approved_rows", int(approved_mask.sum())),
        ("approved_share", float(approved_mask.mean())),
        (
            "approved_default_rate",
            float(scored_validation.loc[approved_mask, "TARGET"].mean()),
        ),
        ("max_abs_diff", float(scored_validation["abs_diff"].max())),
        (
            "break_even_probability",
            model_wrapper.calculator.policy.break_even_probability(),
        ),
        (
            "low_risk_probability_threshold",
            model_wrapper.calculator.policy.low_risk_probability,
        ),
        (
            "max_approval_probability_threshold",
            model_wrapper.calculator.policy.max_approval_probability,
        ),
        ("not_returned_amount", not_returned_amount),
        ("returned_principal", returned_principal),
        ("profit_on_returned", profit_on_returned),
        ("total_realized_income", total_realized_income),
        (
            "average_realized_income",
            float(scored_validation["realized_income"].mean()),
        ),
    ]

    return pd.DataFrame(rows, columns=["metric", "value"])


def build_probability_band_summary(
    scored_validation: pd.DataFrame,
    model_wrapper,
) -> pd.DataFrame:
    """Показывает, как ведут себя разные диапазоны вероятности дефолта"""

    low = model_wrapper.calculator.policy.low_risk_probability
    high = model_wrapper.calculator.policy.max_approval_probability

    result = scored_validation.copy()
    result["probability_band"] = pd.cut(
        result["proba"],
        bins=[0.0, low, high, 1.0],
        labels=[
            f"[0.00, {low:.2f})",
            f"[{low:.2f}, {high:.2f})",
            f"[{high:.2f}, 1.00]",
        ],
        right=False,
        include_lowest=True,
    )

    def summarize(group: pd.DataFrame) -> pd.Series:
        """Считает агрегаты внутри одного probability-диапазона"""

        approved_mask = group["approved_amount"] > 0
        defaulted_mask = approved_mask & group["TARGET"].eq(1)
        returned_mask = approved_mask & group["TARGET"].eq(0)

        returned_principal = float(group.loc[returned_mask, "approved_amount"].sum())
        not_returned_amount = float(group.loc[defaulted_mask, "approved_amount"].sum())
        profit_on_returned = (
            returned_principal * model_wrapper.calculator.policy.margin_rate
        )
        total_realized_income = profit_on_returned - not_returned_amount

        return pd.Series(
            {
                "rows": int(len(group)),
                "default_rate": float(group["TARGET"].mean()),
                "approved_rows": int(approved_mask.sum()),
                "approved_share_inside_band": float(approved_mask.mean()),
                "approved_amount_sum": float(group["approved_amount"].sum()),
                "not_returned_amount": not_returned_amount,
                "profit_on_returned": profit_on_returned,
                "total_realized_income": total_realized_income,
            }
        )

    summary = (
        result.groupby("probability_band", observed=False)
        .apply(summarize)
        .reset_index()
    )

    return summary
