from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd


class ScoringDecision(str, Enum):
    """Итоговое решение по заявке"""

    ACCEPTED = "ACCEPTED"
    DECLINED = "DECLINED"


@dataclass(slots=True, frozen=True)
class ScoringResult:
    """Результат скоринга для одного клиента"""

    proba: float
    threshold: float
    decision: ScoringDecision
    approved_amount: int
    expected_loss: float
    expected_margin_income: float
    expected_total_income: float
    reason: str


def ensure_dataframe(
    features: Mapping[str, Any] | pd.Series | pd.DataFrame,
) -> pd.DataFrame:
    """Приводит признаки к DataFrame"""

    if isinstance(features, pd.DataFrame):
        if features.empty:
            raise ValueError("В DataFrame с признаками нет ни одной строки.")
        return features.copy()

    if isinstance(features, pd.Series):
        # Один объект превращаем в таблицу из одной строки,
        # чтобы дальше код работал одинаково для batch,для single-case.
        return features.to_frame().T

    if isinstance(features, Mapping):
        return pd.DataFrame([dict(features)])

    raise TypeError(
        "features должен быть pandas.Series или pandas.DataFrame"
    )
