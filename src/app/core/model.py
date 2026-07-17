from __future__ import annotations

import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.api import ScoringDecision, ScoringResult
from app.core.calculator import AmountCalculator
from app.core.feature_frame import MODEL_FEATURE_COLUMNS


class GradientBoostingScoringModel:
    """Обёртка над сохранённой моделью и логикой выдачи суммы"""

    def __init__(
        self,
        model_path: str | Path,
        calculator: AmountCalculator | None = None,
    ) -> None:
        model_path = Path(model_path)

        with model_path.open("rb") as file:
            self._model = pickle.load(file)

        self._calculator = calculator or AmountCalculator()

    @property
    def raw_model(self) -> Any:
        """Возвращает исходный sklearn pipeline без обёртки"""

        return self._model

    @property
    def calculator(self) -> AmountCalculator:
        """Возвращает объект с бизнес-логикой сумм"""

        return self._calculator

    @property
    def threshold(self) -> float:
        """Возвращает верхний порог вероятности для одобрения"""

        return self._calculator.policy.max_approval_probability

    @staticmethod
    def _to_dataframe(
        features: pd.DataFrame | pd.Series | Mapping[str, Any],
    ) -> pd.DataFrame:
        """Приводит входные признаки к формату, который ожидает sklearn pipeline"""

        if isinstance(features, pd.DataFrame):
            if features.empty:
                raise ValueError("В DataFrame с признаками нет ни одной строки")
            return features.copy()

        if isinstance(features, pd.Series):
            return features.to_frame().T

        if isinstance(features, Mapping):
            return pd.DataFrame([dict(features)])

        raise TypeError(
            "features должен быть dict, pandas.Series или pandas.DataFrame"
        )

    @staticmethod
    def _ensure_single_row(
        features: pd.DataFrame | pd.Series | Mapping[str, Any],
    ) -> pd.Series:
        """Приводит одну заявку к Series"""

        frame = GradientBoostingScoringModel._to_dataframe(features)
        if len(frame) != 1:
            raise ValueError(
                "Скоринг ожидает ровно одну клиентскую заявку. "
                "Используется score_batch() только для аналитики"
            )

        return frame.iloc[0]

    def predict_proba(
        self,
        features: pd.DataFrame | pd.Series | Mapping[str, Any],
    ) -> float:
        """Считает вероятность дефолта для одной клиентской заявки"""

        feature_row = self._ensure_single_row(features)
        frame = feature_row.to_frame().T
        proba = self._model.predict_proba(frame[MODEL_FEATURE_COLUMNS])[:, 1]
        return float(proba[0])

    def _score_with_proba(
        self,
        feature_row: pd.Series,
        current_proba: float,
    ) -> ScoringResult:
        """Строит итоговый скоринг одной заявки по уже посчитанной вероятности"""

        approved_amount = self._calculator.pick_amount(
            proba=current_proba,
            features=feature_row,
        )
        expected_loss = self._calculator.expected_loss(
            amount=approved_amount,
            proba=current_proba,
        )
        expected_margin_income = self._calculator.expected_margin_income(
            amount=approved_amount,
            proba=current_proba,
        )
        expected_total_income = self._calculator.expected_total_income(
            amount=approved_amount,
            proba=current_proba,
        )
        decision = (
            ScoringDecision.ACCEPTED
            if approved_amount > 0
            else ScoringDecision.DECLINED
        )

        return ScoringResult(
            proba=current_proba,
            threshold=self.threshold,
            decision=decision,
            approved_amount=approved_amount,
            expected_loss=expected_loss,
            expected_margin_income=expected_margin_income,
            expected_total_income=expected_total_income,
            reason=self._calculator.reason(
                proba=current_proba,
                approved_amount=approved_amount,
            ),
        )

    def score(
        self,
        features: pd.DataFrame | pd.Series | Mapping[str, Any],
    ) -> ScoringResult:
        """Скорит одну клиентскую заявку"""

        feature_row = self._ensure_single_row(features)
        current_proba = self.predict_proba(feature_row)
        return self._score_with_proba(feature_row, current_proba)

    def score_one(
        self,
        features: pd.DataFrame | pd.Series | Mapping[str, Any],
    ) -> ScoringResult:

        return self.score(features)

    @staticmethod
    def _result_to_row(result: ScoringResult) -> dict[str, Any]:
        """Превращает результат одного скоринга в строку"""

        return {
            "proba": result.proba,
            "threshold": result.threshold,
            "decision": result.decision.value,
            "approved_amount": result.approved_amount,
            "expected_loss": result.expected_loss,
            "expected_margin_income": result.expected_margin_income,
            "expected_total_income": result.expected_total_income,
            "reason": result.reason,
        }

    def score_batch(
        self,
        features: pd.DataFrame | pd.Series | Mapping[str, Any],
    ) -> pd.DataFrame:
        """Аналитический batch-scoring через скоринг каждой отдельной заявки"""

        frame = self._to_dataframe(features).reset_index(drop=True)
        proba = self._model.predict_proba(frame[MODEL_FEATURE_COLUMNS])[:, 1]
        rows = [
            self._result_to_row(self._score_with_proba(feature_row, float(proba[idx])))
            for idx, (_, feature_row) in enumerate(frame.iterrows())
        ]

        return pd.DataFrame(rows)
