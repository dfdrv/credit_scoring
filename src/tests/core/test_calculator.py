import math

import pandas as pd
import pytest

from app.core.calculator import ApprovalPolicy


@pytest.mark.parametrize(
    ("proba", "expected_amount"),
    [
        pytest.param(0.00, 300_000, id="zero-risk"),
        pytest.param(0.02, 300_000, id="low-risk"),
        pytest.param(0.029999, 300_000, id="before-low-border"),
        pytest.param(0.03, 100_000, id="low-border"),
        pytest.param(0.04, 100_000, id="medium-risk"),
        pytest.param(0.049999, 100_000, id="before-decline-border"),
        pytest.param(0.05, 0, id="decline-border"),
        pytest.param(0.90, 0, id="high-risk"),
    ],
)
def test_pick_amount_by_probability_band(
    calculator,
    default_features,
    proba,
    expected_amount,
):
    # Здесь проверяем только влияние вероятности дефолта.
    # Лимиты заявки специально выбраны такими, чтобы не резать сумму
    assert calculator.pick_amount(
        proba=proba,
        features=default_features,
    ) == expected_amount


@pytest.mark.parametrize(
    ("features", "expected_amount"),
    [
        pytest.param(
            {"AMT_CREDIT": 180_000, "AMT_INCOME_TOTAL": 1_000_000},
            150_000,
            id="credit-cap-round-down",
        ),
        pytest.param(
            {"AMT_CREDIT": 50_000, "AMT_INCOME_TOTAL": 1_000_000},
            50_000,
            id="exact-min-amount",
        ),
        pytest.param(
            {"AMT_CREDIT": 49_999, "AMT_INCOME_TOTAL": 1_000_000},
            0,
            id="below-min-after-rounding",
        ),
        pytest.param(
            {"AMT_CREDIT": 500_000, "AMT_INCOME_TOTAL": 60_000},
            200_000,
            id="income-cap",
        ),
        pytest.param(
            {"AMT_CREDIT": 500_000, "AMT_INCOME_TOTAL": 10_000},
            0,
            id="income-cap-too-small",
        ),
        pytest.param(
            {"AMT_CREDIT": 125_000},
            100_000,
            id="only-credit-limit",
        ),
        pytest.param(
            {"AMT_INCOME_TOTAL": 80_000},
            300_000,
            id="only-income-limit",
        ),
        pytest.param(
            {"AMT_CREDIT": None, "AMT_INCOME_TOTAL": 0},
            0,
            id="no-positive-limits",
        ),
        pytest.param(
            {"AMT_CREDIT": -100_000, "AMT_INCOME_TOTAL": -50_000},
            0,
            id="negative-values",
        ),
        pytest.param(
            {"AMT_CREDIT": math.nan, "AMT_INCOME_TOTAL": 100_000},
            300_000,
            id="nan-credit",
        ),
        pytest.param(
            {"AMT_CREDIT": "250000", "AMT_INCOME_TOTAL": "100000"},
            250_000,
            id="string-numbers",
        ),
    ],
)
def test_pick_amount_by_affordability_limits(
    calculator,
    features,
    expected_amount,
):
    # Здесь риск всегда низкий, чтобы проверять именно ограничения заявки
    assert calculator.pick_amount(
        proba=0.02,
        features=features,
    ) == expected_amount


def test_pick_amount_supports_pandas_series(calculator, default_features):
    # Признаки могут прийти не только словарем, но и строкой pandas
    features = pd.Series(default_features)

    assert calculator.pick_amount(
        proba=0.02,
        features=features,
    ) == 300_000


def test_pick_amount_uses_custom_policy(
    custom_calculator,
    default_features,
):
    # Проверяем три зоны риска на нестандартной policy
    assert custom_calculator.pick_amount(
        proba=0.05,
        features=default_features,
    ) == 300_000
    assert custom_calculator.pick_amount(
        proba=0.15,
        features=default_features,
    ) == 150_000
    assert custom_calculator.pick_amount(
        proba=0.20,
        features=default_features,
    ) == 0


def test_pick_amount_uses_custom_step_and_min_amount(custom_calculator):
    # При шаге 25 000 сумма 74 999 должна округлиться вниз до 50 000.
    features = {
        "AMT_CREDIT": 74_999,
        "AMT_INCOME_TOTAL": 1_000_000,
    }

    assert custom_calculator.pick_amount(
        proba=0.05,
        features=features,
    ) == 50_000


def test_break_even_probability_default_policy():
    # Для дефолтной политики точка безубыточности:
    # margin / (margin + loss) = 0.05 / 1.05
    policy = ApprovalPolicy()

    assert policy.break_even_probability() == pytest.approx(0.05 / 1.05)


def test_break_even_probability_custom_policy(custom_calculator):
    # Та же формула, но уже для кастомных ставок.
    assert custom_calculator.policy.break_even_probability() == pytest.approx(
        0.10 / 0.60
    )


@pytest.mark.parametrize(
    ("amount", "proba", "expected_loss"),
    [
        pytest.param(0, 0.02, 0.0, id="zero-amount"),
        pytest.param(100_000, 0.00, 0.0, id="zero-risk"),
        pytest.param(100_000, 0.02, 2_000.0, id="default-case"),
        pytest.param(250_000, 0.05, 12_500.0, id="large-amount"),
    ],
)
def test_expected_loss(calculator, amount, proba, expected_loss):
    # Ожидаемый убыток считаем как сумма * вероятность дефолта.
    assert calculator.expected_loss(
        amount=amount,
        proba=proba,
    ) == pytest.approx(expected_loss)


@pytest.mark.parametrize(
    ("amount", "proba", "expected_income"),
    [
        pytest.param(0, 0.02, 0.0, id="zero-amount"),
        pytest.param(100_000, 0.00, 5_000.0, id="zero-risk"),
        pytest.param(100_000, 0.02, 4_900.0, id="default-case"),
        pytest.param(250_000, 0.05, 11_875.0, id="large-amount"),
    ],
)
def test_expected_margin_income(
    calculator,
    amount,
    proba,
    expected_income,
):
    # Маржинальный доход получаем только на недефолтной части.
    assert calculator.expected_margin_income(
        amount=amount,
        proba=proba,
    ) == pytest.approx(expected_income)


@pytest.mark.parametrize(
    ("amount", "proba", "expected_total_income"),
    [
        pytest.param(0, 0.02, 0.0, id="zero-amount"),
        pytest.param(100_000, 0.00, 5_000.0, id="zero-risk"),
        pytest.param(100_000, 0.02, 2_900.0, id="positive-result"),
        pytest.param(100_000, 0.05, -250.0, id="negative-result"),
    ],
)
def test_expected_total_income(
    calculator,
    amount,
    proba,
    expected_total_income,
):
    # Итоговая экономика заявки получается как доход минус ожидаемый убыток.
    assert calculator.expected_total_income(
        amount=amount,
        proba=proba,
    ) == pytest.approx(expected_total_income)


def test_expected_metrics_use_custom_policy(custom_calculator):
    # Экономические формулы берут ставки из policy
    amount = 100_000
    proba = 0.10

    assert custom_calculator.expected_loss(
        amount=amount,
        proba=proba,
    ) == pytest.approx(5_000.0)
    assert custom_calculator.expected_margin_income(
        amount=amount,
        proba=proba,
    ) == pytest.approx(9_000.0)
    assert custom_calculator.expected_total_income(
        amount=amount,
        proba=proba,
    ) == pytest.approx(4_000.0)


@pytest.mark.parametrize(
    ("proba", "approved_amount", "expected_reason"),
    [
        pytest.param(
            0.01,
            0,
            "declined_by_probability_or_affordability",
            id="declined",
        ),
        pytest.param(
            0.01,
            100_000,
            "approved_low_risk_band",
            id="low-risk-approved",
        ),
        pytest.param(
            0.03,
            100_000,
            "approved_medium_risk_band",
            id="low-border-approved",
        ),
        pytest.param(
            0.04,
            100_000,
            "approved_medium_risk_band",
            id="medium-risk-approved",
        ),
    ],
)
def test_reason(calculator, proba, approved_amount, expected_reason):
    # Граница 0.03 уже считается средним риском
    assert calculator.reason(
        proba=proba,
        approved_amount=approved_amount,
    ) == expected_reason
