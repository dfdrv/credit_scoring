from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True, slots=True)
class ApprovalPolicy:
    """сколько можно одобрить"""

    low_risk_probability: float = 0.03
    max_approval_probability: float = 0.05

    low_risk_amount: int = 300_000
    medium_risk_amount: int = 100_000

    min_amount: int = 50_000
    amount_step: int = 50_000

    income_multiplier_cap: float = 4.0

    # при дефолте считаем, что потеряли всю выданную сумму,
    # при возврате заработали 5% от суммы займа.
    margin_rate: float = 0.05
    default_loss_rate: float = 1.0

    def break_even_probability(self) -> float:
        """Возвращает вероятность дефолта на границе безубыточности"""

        return self.margin_rate / (self.margin_rate + self.default_loss_rate)


class AmountCalculator:
    """Подбирает сумму выдачи по риску и простым ограничениям"""

    def __init__(self, policy: ApprovalPolicy | None = None) -> None:
        self.policy = policy or ApprovalPolicy()

    @staticmethod
    def _to_series(features: Mapping[str, Any] | pd.Series) -> pd.Series:
        """Приводит признаки одной заявки к Series"""

        if isinstance(features, pd.Series):
            return features
        return pd.Series(dict(features))

    @staticmethod
    def _to_float(value: Any) -> float | None:
        """Переводит значение в float"""

        if value is None or pd.isna(value):
            return None
        return float(value)

    def _round_down_to_step(self, amount: float) -> int:
        """Округляет сумму вниз до заданного шага"""

        return int(amount // self.policy.amount_step * self.policy.amount_step)

    def _affordability_cap(self, features: Mapping[str, Any] | pd.Series) -> int:
        """Считает верхнюю границу суммы по доходу и запрошенному кредиту"""

        row = self._to_series(features)

        requested_credit = self._to_float(row.get("AMT_CREDIT"))
        income_total = self._to_float(row.get("AMT_INCOME_TOTAL"))

        caps: list[float] = []

        if requested_credit is not None and requested_credit > 0:
            caps.append(requested_credit)

        if income_total is not None and income_total > 0:
            caps.append(income_total * self.policy.income_multiplier_cap)

        if not caps:
            return 0

        # Берём самое консервативное ограничение.
        cap = min(caps)
        cap = self._round_down_to_step(cap)

        if cap < self.policy.min_amount:
            return 0

        return cap

    def _base_amount_by_probability(self, proba: float) -> int:
        """Возвращает базовую сумму только по уровню риска"""

        if proba < self.policy.low_risk_probability:
            return self.policy.low_risk_amount

        if proba < self.policy.max_approval_probability:
            return self.policy.medium_risk_amount

        return 0

    def pick_amount(
        self,
        proba: float,
        features: Mapping[str, Any] | pd.Series,
    ) -> int:
        """Выбирает итоговую сумму выдачи для клиента"""

        base_amount = self._base_amount_by_probability(proba)
        if base_amount == 0:
            return 0

        affordability_cap = self._affordability_cap(features)
        if affordability_cap == 0:
            return 0

        approved_amount = min(base_amount, affordability_cap)
        approved_amount = self._round_down_to_step(approved_amount)

        if approved_amount < self.policy.min_amount:
            return 0

        return approved_amount

    def expected_loss(self, amount: int, proba: float) -> float:
        """Считает ожидаемый убыток от дефолта"""

        if amount == 0:
            return 0.0
        return amount * proba * self.policy.default_loss_rate

    def expected_margin_income(self, amount: int, proba: float) -> float:
        """Считает ожидаемый доход на возвращённых займах"""

        if amount == 0:
            return 0.0
        return amount * (1.0 - proba) * self.policy.margin_rate

    def expected_total_income(self, amount: int, proba: float) -> float:
        """Считает ожидаемый финансовый результат по заявке"""

        return self.expected_margin_income(amount, proba) - self.expected_loss(
            amount, proba
        )

    def reason(self, proba: float, approved_amount: int) -> str:
        """Возвращает короткое пояснение к решению"""

        if approved_amount == 0:
            return "declined_by_probability_or_affordability"

        if proba < self.policy.low_risk_probability:
            return "approved_low_risk_band"

        return "approved_medium_risk_band"
