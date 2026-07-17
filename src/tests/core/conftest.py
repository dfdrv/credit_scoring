import pytest

from app.core.calculator import AmountCalculator, ApprovalPolicy


# Базовая заявка, на которой удобно проверять риск.
# Кредит большой, доход нормальный, поэтому лимиты не мешают модели
# одобрять стандартные суммы: 300 000 или 100 000.
DEFAULT_FEATURES = {
    "AMT_CREDIT": 500_000,
    "AMT_INCOME_TOTAL": 100_000,
}


@pytest.fixture()
def default_features() -> dict[str, int]:
    return DEFAULT_FEATURES.copy()


@pytest.fixture()
def calculator() -> AmountCalculator:
    # Обычный калькулятор с дефолтной бизнес-политикой
    return AmountCalculator()


@pytest.fixture()
def custom_calculator() -> AmountCalculator:
    policy = ApprovalPolicy(
        low_risk_probability=0.10,
        max_approval_probability=0.20,
        low_risk_amount=400_000,
        medium_risk_amount=150_000,
        min_amount=25_000,
        amount_step=25_000,
        income_multiplier_cap=3.0,
        margin_rate=0.10,
        default_loss_rate=0.50,
    )
    return AmountCalculator(policy=policy)
