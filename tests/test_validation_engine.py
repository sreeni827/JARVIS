from copy import deepcopy

import pandas as pd
import pytest

from backtesting.validation_engine import ValidationEngine


@pytest.fixture
def engine():

    instance = ValidationEngine.__new__(
        ValidationEngine
    )

    instance.config = {
        "safety": {
            "historical_simulation_only": True,
            "live_trading_enabled": False,
            "automatic_strategy_activation": False,
            "require_forward_validation": True
        }
    }

    instance.warmup = 250

    instance.symbols = ["TEST"]

    instance.periods = {
        "development": {
            "start_date": "2025-01-01",
            "end_date": "2025-06-30"
        },
        "validation": {
            "start_date": "2025-07-01",
            "end_date": "2025-12-31"
        }
    }

    instance.qualification = {
        "minimum_trades": 30,
        "minimum_profit_factor": 1.5,
        "minimum_expectancy": 0.0,
        "minimum_return_percent": 0.0,
        "maximum_drawdown_percent": 5.0,
        "maximum_risk_breach_percent": 10.0
    }

    instance.history_cache = {}

    return instance


@pytest.fixture
def history():

    dates = pd.date_range(
        "2023-01-01",
        "2026-01-31",
        freq="B",
        tz="America/New_York"
    )

    return pd.DataFrame(
        {
            "close": 100.0,
            "atr": 2.0
        },
        index=dates
    )


@pytest.fixture
def valid_metrics():

    return {
        "total_trades": 40,
        "profit_factor": 1.8,
        "expectancy": 0.10,
        "total_return": 3.0,
        "max_drawdown": 2.0,
        "risk_breach_percent": 5.0
    }


def test_valid_configuration(engine):

    engine.validate_configuration()


def test_live_trading_blocked(engine):

    engine.config["safety"][
        "live_trading_enabled"
    ] = True

    with pytest.raises(RuntimeError):

        engine.validate_configuration()


def test_historical_mode_required(engine):

    engine.config["safety"][
        "historical_simulation_only"
    ] = False

    with pytest.raises(RuntimeError):

        engine.validate_configuration()


def test_automatic_activation_blocked(engine):

    engine.config["safety"][
        "automatic_strategy_activation"
    ] = True

    with pytest.raises(RuntimeError):

        engine.validate_configuration()


def test_forward_validation_required(engine):

    engine.config["safety"][
        "require_forward_validation"
    ] = False

    with pytest.raises(RuntimeError):

        engine.validate_configuration()


def test_insufficient_warmup_configuration(engine):

    engine.warmup = 100

    with pytest.raises(ValueError):

        engine.validate_configuration()


def test_duplicate_symbols_rejected(engine):

    engine.symbols = ["TEST", "TEST"]

    with pytest.raises(ValueError):

        engine.validate_configuration()


def test_overlapping_periods_rejected(engine):

    engine.periods["validation"][
        "start_date"
    ] = "2025-06-01"

    with pytest.raises(ValueError):

        engine.validate_configuration()


def test_invalid_period_rejected(engine):

    engine.periods["development"][
        "end_date"
    ] = "2024-01-01"

    with pytest.raises(ValueError):

        engine.validate_configuration()


def test_period_contains_correct_warmup(
    engine,
    history
):

    engine.history_cache["TEST"] = history

    result = engine.prepare_period(
        "TEST",
        "2025-01-01",
        "2025-06-30"
    )

    first_trading_date = (
        result.index[engine.warmup].date()
    )

    assert first_trading_date == (
        pd.Timestamp("2025-01-01").date()
    )


def test_period_excludes_future_data(
    engine,
    history
):

    engine.history_cache["TEST"] = history

    result = engine.prepare_period(
        "TEST",
        "2025-01-01",
        "2025-06-30"
    )

    assert result.index.max().date() <= (
        pd.Timestamp("2025-06-30").date()
    )


def test_insufficient_historical_warmup(
    engine,
    history
):

    engine.history_cache["TEST"] = history

    with pytest.raises(ValueError):

        engine.prepare_period(
            "TEST",
            "2023-02-01",
            "2023-03-01"
        )


def test_missing_period_data(engine, history):

    engine.history_cache["TEST"] = history

    with pytest.raises(ValueError):

        engine.prepare_period(
            "TEST",
            "2030-01-01",
            "2030-06-30"
        )


def test_history_cache_not_modified(
    engine,
    history
):

    engine.history_cache["TEST"] = history

    result = engine.load_history("TEST")

    result.iloc[0, 0] = 999.0

    assert (
        engine.history_cache["TEST"].iloc[0, 0]
        == 100.0
    )


def test_no_trade_risk_breaches(engine):

    count, rate = (
        engine.calculate_risk_breaches([])
    )

    assert count == 0
    assert rate == 0.0


def test_risk_breach_calculation(engine):

    trades = [
        {
            "planned_risk": 0.50,
            "pnl": -0.75
        },
        {
            "planned_risk": 0.50,
            "pnl": -0.40
        },
        {
            "planned_risk": 0.50,
            "pnl": 1.00
        }
    ]

    count, rate = (
        engine.calculate_risk_breaches(trades)
    )

    assert count == 1

    assert rate == pytest.approx(
        33.3333333333
    )


def test_invalid_planned_risk(engine):

    trades = [
        {
            "planned_risk": 0.0,
            "pnl": -0.50
        }
    ]

    with pytest.raises(ValueError):

        engine.calculate_risk_breaches(trades)


def test_historical_qualification(
    engine,
    valid_metrics
):

    status, reason = engine.qualify(
        valid_metrics
    )

    assert status == "HISTORICAL_PASS"

    assert "Forward validation" in reason


@pytest.mark.parametrize(
    "field,value",
    [
        ("total_trades", 10),
        ("profit_factor", 1.0),
        ("expectancy", -0.10),
        ("total_return", -1.0),
        ("max_drawdown", 6.0),
        ("risk_breach_percent", 15.0)
    ]
)
def test_qualification_failures(
    engine,
    valid_metrics,
    field,
    value
):

    metrics = deepcopy(valid_metrics)

    metrics[field] = value

    status, reason = engine.qualify(
        metrics
    )

    assert status == "NOT_QUALIFIED"

    assert reason