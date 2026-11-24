from copy import deepcopy
from dataclasses import replace
from decimal import Decimal

import pytest

from portfolio.portfolio_config import PortfolioConfig
from risk.risk_config import RiskConfig


# ============================================================
# TEST FIXTURES
# ============================================================


@pytest.fixture
def guardian():

    return RiskConfig(

        starting_capital=Decimal("100"),

        max_daily_loss_percent=Decimal("1"),

        max_drawdown_percent=Decimal("5"),

        risk_per_trade_percent=Decimal("0.5"),

        max_position_percent=Decimal("20"),

        max_total_exposure_percent=Decimal("60"),

        min_order_value=Decimal("1"),

        max_open_positions=3,

        enabled=True,

        emergency_stop=False,

        live_trading_enabled=False,

        margin_enabled=False,

        short_selling_enabled=False,

        trading_mode="paper"

    )


@pytest.fixture
def baseline(guardian):

    return PortfolioConfig(
        risk_config=guardian
    )


@pytest.fixture
def config(baseline):

    return baseline.config


# ============================================================
# BASELINE VALIDATION
# ============================================================


def test_valid_configuration(baseline):

    assert baseline.validate() is True


def test_starting_capital(baseline):

    assert (
        baseline.config["account"]["initial_capital"]
        == 100
    )


def test_trading_universe(baseline):

    symbols = baseline.config["universe"]["symbols"]

    assert len(symbols) == 12

    assert len(symbols) == len(set(symbols))


def test_portfolio_risk_limits(baseline):

    risk = baseline.get_section("risk")

    assert risk["maximum_open_positions"] == 3

    assert risk["maximum_position_percent"] == 20

    assert risk["maximum_total_exposure_percent"] == 60

    assert risk["maximum_aggregate_open_risk_percent"] == 1


# ============================================================
# CONFIGURATION STRUCTURE
# ============================================================


@pytest.mark.parametrize(
    "section",
    sorted(PortfolioConfig.REQUIRED_SECTIONS)
)
def test_missing_required_section(
    config,
    guardian,
    section
):

    del config[section]

    with pytest.raises(
        ValueError,
        match="missing keys"
    ):

        PortfolioConfig(
            config=config,
            risk_config=guardian
        )


@pytest.mark.parametrize(
    "section",
    [
        "account",
        "historical_data",
        "universe",
        "risk",
        "execution",
        "opportunity_selection",
        "strategy",
        "benchmarks",
        "reporting",
        "safety"
    ]
)
def test_invalid_section_type(
    config,
    guardian,
    section
):

    config[section] = []

    with pytest.raises(ValueError):

        PortfolioConfig(
            config=config,
            risk_config=guardian
        )


def test_missing_nested_setting(config, guardian):

    del config["risk"]["maximum_open_positions"]

    with pytest.raises(
        ValueError,
        match="missing keys"
    ):

        PortfolioConfig(
            config=config,
            risk_config=guardian
        )


# ============================================================
# CONFIGURATION PROTECTION
# ============================================================


def test_config_returns_defensive_copy(baseline):

    external = baseline.config

    external["risk"]["maximum_open_positions"] = 999

    assert (
        baseline.config["risk"]["maximum_open_positions"]
        == 3
    )


def test_section_returns_defensive_copy(baseline):

    external = baseline.get_section("safety")

    external["live_trading_enabled"] = True

    assert (
        baseline.config["safety"]["live_trading_enabled"]
        is False
    )


def test_original_input_is_protected(config, guardian):

    instance = PortfolioConfig(
        config=config,
        risk_config=guardian
    )

    config["account"]["initial_capital"] = 999

    assert (
        instance.config["account"]["initial_capital"]
        == 100
    )


def test_nested_list_is_protected(baseline):

    external = baseline.config

    external["universe"]["symbols"].append("TEST")

    assert (
        "TEST"
        not in baseline.config["universe"]["symbols"]
    )


# ============================================================
# ACCOUNT VALIDATION
# ============================================================


@pytest.mark.parametrize(
    "value",
    [
        0,
        -100,
        float("inf"),
        float("nan"),
        True,
        "100"
    ]
)
def test_invalid_initial_capital(
    config,
    guardian,
    value
):

    config["account"]["initial_capital"] = value

    with pytest.raises(ValueError):

        PortfolioConfig(
            config=config,
            risk_config=guardian
        )


@pytest.mark.parametrize(
    "key,value",
    [
        ("currency", "EUR"),
        ("fractional_shares", False),
        ("quantity_precision", 0),
        ("quantity_precision", 7),
        ("quantity_precision", 2.5),
        ("quantity_precision", True),
        ("monthly_contribution", -100),
        ("simulate_contributions", True),
        ("margin_enabled", True),
        ("short_selling_enabled", True)
    ]
)
def test_invalid_account_settings(
    config,
    guardian,
    key,
    value
):

    config["account"][key] = value

    with pytest.raises(ValueError):

        PortfolioConfig(
            config=config,
            risk_config=guardian
        )


# ============================================================
# HISTORICAL CONFIGURATION
# ============================================================


@pytest.mark.parametrize(
    "key,value",
    [
        ("historical_days", 100),
        ("historical_days", True),
        ("warmup_candles", 199),
        ("warmup_candles", 200.5),
        ("start_date", "2026-02-30"),
        ("start_date", "2024/11/22"),
        ("start_date", "2027-01-01"),
        ("end_date", "2020-01-01"),
        ("timeframe", "MINUTE"),
        ("require_aligned_sessions", False)
    ]
)
def test_invalid_historical_settings(
    config,
    guardian,
    key,
    value
):

    config["historical_data"][key] = value

    with pytest.raises(ValueError):

        PortfolioConfig(
            config=config,
            risk_config=guardian
        )


def test_equal_historical_dates(config, guardian):

    config["historical_data"]["end_date"] = (
        config["historical_data"]["start_date"]
    )

    with pytest.raises(ValueError):

        PortfolioConfig(
            config=config,
            risk_config=guardian
        )


# ============================================================
# SYMBOL VALIDATION
# ============================================================


@pytest.mark.parametrize(
    "symbols",
    [
        [],
        ["AAPL", "AAPL"],
        ["aapl"],
        ["AAPL "],
        [""],
        [123],
        ["INVALID SYMBOL"],
        "AAPL"
    ]
)
def test_invalid_trading_universe(
    config,
    guardian,
    symbols
):

    config["universe"]["symbols"] = symbols

    with pytest.raises(ValueError):

        PortfolioConfig(
            config=config,
            risk_config=guardian
        )


# ============================================================
# RISK VALIDATION
# ============================================================


@pytest.mark.parametrize(
    "key,value",
    [
        ("risk_per_trade_percent", 0),
        ("risk_per_trade_percent", -1),
        ("risk_per_trade_percent", 2),
        ("maximum_daily_loss_percent", 0),
        ("maximum_drawdown_percent", 101),
        ("maximum_position_percent", 70),
        ("maximum_total_exposure_percent", 150),
        ("maximum_aggregate_open_risk_percent", 5),
        ("maximum_open_positions", 0),
        ("maximum_open_positions", True),
        ("maximum_open_positions", 2.5),
        ("minimum_order_value", 0),
        ("minimum_order_value", 21),
        ("use_starting_capital_ceiling", False),
        ("allow_duplicate_positions", True),
        ("allow_pyramiding", True),
        ("halt_new_entries_on_daily_loss", False),
        ("halt_on_maximum_drawdown", False),
        ("liquidate_on_drawdown_breach", False)
    ]
)
def test_invalid_risk_settings(
    config,
    guardian,
    key,
    value
):

    config["risk"][key] = value

    with pytest.raises(ValueError):

        PortfolioConfig(
            config=config,
            risk_config=guardian
        )


# ============================================================
# EXECUTION VALIDATION
# ============================================================


@pytest.mark.parametrize(
    "key,value",
    [
        ("slippage_percent", -0.1),
        ("slippage_percent", 11),
        ("slippage_percent", float("nan")),
        ("transaction_fee_percent", -1),
        ("transaction_fee_percent", 11),
        ("entry_timing", "SAME_SESSION_CLOSE"),
        ("stop_target_conflict", "TARGET_FIRST"),
        ("maximum_holding_sessions", 0),
        ("liquidate_at_end", False),
        ("allow_same_session_reentry", True)
    ]
)
def test_invalid_execution_settings(
    config,
    guardian,
    key,
    value
):

    config["execution"][key] = value

    with pytest.raises(ValueError):

        PortfolioConfig(
            config=config,
            risk_config=guardian
        )


def test_zero_execution_costs(config, guardian):

    config["execution"]["slippage_percent"] = 0

    config["execution"]["transaction_fee_percent"] = 0

    instance = PortfolioConfig(
        config=config,
        risk_config=guardian
    )

    assert instance.validate() is True


# ============================================================
# OPPORTUNITY SELECTION
# ============================================================


@pytest.mark.parametrize(
    "key,value",
    [
        ("minimum_score", -1),
        ("minimum_score", 101),
        ("minimum_score", True),
        ("ranking_method", "RANDOM"),
        ("ranking_direction", "ASCENDING"),
        ("tie_breaker", "RANDOM"),
        ("maximum_new_positions_per_session", 0),
        ("maximum_new_positions_per_session", 4)
    ]
)
def test_invalid_opportunity_settings(
    config,
    guardian,
    key,
    value
):

    config["opportunity_selection"][key] = value

    with pytest.raises(ValueError):

        PortfolioConfig(
            config=config,
            risk_config=guardian
        )


@pytest.mark.parametrize(
    "score",
    [0, 55, 65, 75, 100]
)
def test_valid_score_boundaries(
    config,
    guardian,
    score
):

    config["opportunity_selection"]["minimum_score"] = score

    instance = PortfolioConfig(
        config=config,
        risk_config=guardian
    )

    assert instance.validate() is True


# ============================================================
# STRATEGY VALIDATION
# ============================================================


@pytest.mark.parametrize(
    "key,value",
    [
        ("momentum_enabled", "true"),
        ("breakout_enabled", 1),
        ("pullback_enabled", None),
        ("stop_atr_multiplier", 0),
        ("stop_atr_multiplier", -1),
        ("risk_reward_ratio", 0.5),
        ("risk_reward_ratio", float("inf")),
        ("research_only", False)
    ]
)
def test_invalid_strategy_settings(
    config,
    guardian,
    key,
    value
):

    config["strategy"][key] = value

    with pytest.raises(ValueError):

        PortfolioConfig(
            config=config,
            risk_config=guardian
        )


def test_all_strategies_disabled_allowed(
    config,
    guardian
):

    config["strategy"]["momentum_enabled"] = False

    config["strategy"]["breakout_enabled"] = False

    config["strategy"]["pullback_enabled"] = False

    instance = PortfolioConfig(
        config=config,
        risk_config=guardian
    )

    assert instance.validate() is True


# ============================================================
# BENCHMARK VALIDATION
# ============================================================


@pytest.mark.parametrize(
    "key,value",
    [
        ("enabled", "true"),
        ("include_cash_benchmark", 1),
        ("symbols", []),
        ("symbols", ["SPY", "SPY"]),
        ("symbols", ["spy"])
    ]
)
def test_invalid_benchmark_settings(
    config,
    guardian,
    key,
    value
):

    config["benchmarks"][key] = value

    with pytest.raises(ValueError):

        PortfolioConfig(
            config=config,
            risk_config=guardian
        )


# ============================================================
# REPORTING VALIDATION
# ============================================================


@pytest.mark.parametrize(
    "key",
    [
        "save_trades",
        "save_equity_curve",
        "save_daily_snapshots",
        "save_rejected_opportunities",
        "save_risk_events",
        "save_position_history"
    ]
)
def test_mandatory_reporting(
    config,
    guardian,
    key
):

    config["reporting"][key] = False

    with pytest.raises(ValueError):

        PortfolioConfig(
            config=config,
            risk_config=guardian
        )


# ============================================================
# SAFETY VALIDATION
# ============================================================


@pytest.mark.parametrize(
    "key,value",
    [
        ("historical_simulation_only", False),
        ("live_trading_enabled", True),
        ("broker_order_submission_enabled", True),
        ("automatic_strategy_activation", True),
        ("require_forward_validation", False),
        ("fail_on_missing_market_data", False),
        ("fail_on_accounting_mismatch", False)
    ]
)
def test_unsafe_configuration(
    config,
    guardian,
    key,
    value
):

    config["safety"][key] = value

    with pytest.raises(ValueError):

        PortfolioConfig(
            config=config,
            risk_config=guardian
        )


@pytest.mark.parametrize(
    "value",
    [
        "false",
        "true",
        0,
        1,
        None
    ]
)
def test_strict_safety_boolean(
    config,
    guardian,
    value
):

    config["safety"]["live_trading_enabled"] = value

    with pytest.raises(ValueError):

        PortfolioConfig(
            config=config,
            risk_config=guardian
        )


# ============================================================
# RISK GUARDIAN SYNCHRONIZATION
# ============================================================


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("account", "initial_capital", 200),
        ("risk", "risk_per_trade_percent", 0.25),
        ("risk", "maximum_daily_loss_percent", 2),
        ("risk", "maximum_drawdown_percent", 6),
        ("risk", "maximum_position_percent", 25),
        ("risk", "maximum_total_exposure_percent", 70),
        ("risk", "maximum_open_positions", 4),
        ("risk", "minimum_order_value", 2)
    ]
)
def test_risk_guardian_conflict(
    config,
    guardian,
    section,
    key,
    value
):

    config[section][key] = value

    with pytest.raises(
        ValueError,
        match="Risk configuration conflict"
    ):

        PortfolioConfig(
            config=config,
            risk_config=guardian
        )


@pytest.mark.parametrize(
    "key,value",
    [
        ("trading_mode", "live"),
        ("enabled", False),
        ("emergency_stop", True),
        ("live_trading_enabled", True),
        ("margin_enabled", True),
        ("short_selling_enabled", True)
    ]
)
def test_unsafe_risk_guardian(
    config,
    guardian,
    key,
    value
):

    unsafe_guardian = replace(
        guardian,
        **{key: value}
    )

    with pytest.raises(ValueError):

        PortfolioConfig(
            config=config,
            risk_config=unsafe_guardian
        )


def test_matching_modified_risk_configuration(
    config,
    guardian
):

    config["risk"]["maximum_open_positions"] = 2

    config["opportunity_selection"][
        "maximum_new_positions_per_session"
    ] = 2

    updated_guardian = replace(
        guardian,
        max_open_positions=2
    )

    instance = PortfolioConfig(
        config=config,
        risk_config=updated_guardian
    )

    assert instance.validate() is True


# ============================================================
# SUMMARY AND SAFETY OUTPUT
# ============================================================


def test_configuration_summary(baseline, capsys):

    baseline.display_summary()

    output = capsys.readouterr().out

    assert "Starting capital: $100.00" in output

    assert "Maximum position value: $20.00" in output

    assert "Maximum total exposure: $60.00" in output

    assert "Live trading: DISABLED" in output

    assert "Broker order submission: DISABLED" in output

    assert "PORTFOLIO CONFIGURATION VALIDATED" in output