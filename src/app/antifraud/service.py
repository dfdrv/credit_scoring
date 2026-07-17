from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd


class AntifraudService:
    ACCEPTED = "accepted"
    RULE_1 = "AF001_low_external_score_pair"
    RULE_2 = "AF002_young_high_credit_to_income"
    RULE_3 = "AF003_low_skill_laborers"

    @staticmethod
    def _value(features: Mapping[str, Any] | pd.Series, name: str) -> Any:
        return features.get(name)

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value is None or pd.isna(value):
            return None
        return float(value)

    def _rule_1(self, features: Mapping[str, Any] | pd.Series) -> bool:
        ext2 = self._to_float(self._value(features, "EXT_SOURCE_2"))
        ext3 = self._to_float(self._value(features, "EXT_SOURCE_3"))
        return ext2 is not None and ext3 is not None and ext2 < 0.20 and ext3 < 0.20

    def _rule_2(self, features: Mapping[str, Any] | pd.Series) -> bool:
        days_birth = self._to_float(self._value(features, "DAYS_BIRTH"))
        ratio = self._to_float(self._value(features, "credit_to_income_ratio"))
        if days_birth is None or ratio is None:
            return False

        age_years = -days_birth / 365.25
        return age_years < 25 and ratio > 3.0

    def _rule_3(self, features: Mapping[str, Any] | pd.Series) -> bool:
        return self._value(features, "OCCUPATION_TYPE") == "Low-skill Laborers"

    def check(self, features: Mapping[str, Any] | pd.Series) -> str:
        if self._rule_1(features):
            return self.RULE_1
        if self._rule_2(features):
            return self.RULE_2
        if self._rule_3(features):
            return self.RULE_3
        return self.ACCEPTED
