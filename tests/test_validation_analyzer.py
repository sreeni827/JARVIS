import pandas as pd
import pytest

from backtesting.validation_analyzer import (
    ValidationAnalyzer
)


@pytest.fixture
def analyzer():

    instance = ValidationAnalyzer.__new__(
        ValidationAnalyzer
    )

    instance.symbols = ["TEST"]

    instance.periods = [
        "development",
        "validation",
        "historical_test"
    ]

    instance.scenarios = [
        "baseline",
        "moderate",
        "stressed"
    ]

    summaries = []

    trades = []

    returns = {
        "baseline": 1.0,
        "moderate": 0.5,
        "stressed": -0.25
    }

    for period in instance.periods:

        for scenario in instance.scenarios:

            pnl = returns[scenario]

            summaries.append({
                "symbol": "TEST",
                "period": period,
                "scenario": scenario,
                "total_trades": 1,
                "total_return": pnl,
                "net_profit": pnl,
                "profit_factor": (
                    float("inf")
                    if pnl > 0
                    else 0.0
                ),
                "expectancy": pnl,
                "max_drawdown": 1.0,
                "risk_breaches": 0,
                "qualification": "NOT_QUALIFIED"
            })

            trades.append({
                "symbol": "TEST",
                "strategy": "MOMENTUM",
                "period": period,
                "scenario": scenario,
                "pnl": pnl,
                "planned_risk": 0.50,
                "entry_date": "2025-01-02",
                "exit_date": "2025-01-03"
            })

    instance.summary = pd.DataFrame(
        summaries
    )

    instance.trades = pd.DataFrame(
        trades
    )

    return instance


def test_valid_reports(analyzer):

    analyzer.validate_reports()


def test_duplicate_simulations_rejected(
    analyzer
):

    analyzer.summary = pd.concat(
        [
            analyzer.summary,
            analyzer.summary.iloc[[0]]
        ],
        ignore_index=True
    )

    with pytest.raises(ValueError):

        analyzer.validate_reports()


def test_missing_simulation_rejected(
    analyzer
):

    analyzer.summary = (
        analyzer.summary.iloc[1:].copy()
    )

    with pytest.raises(ValueError):

        analyzer.validate_reports()


def test_missing_summary_column(analyzer):

    analyzer.summary = (
        analyzer.summary.drop(
            columns=["net_profit"]
        )
    )

    with pytest.raises(ValueError):

        analyzer.validate_reports()


def test_trade_count_mismatch(analyzer):

    analyzer.summary.loc[
        0, "total_trades"
    ] = 2

    with pytest.raises(ValueError):

        analyzer.validate_reports()


def test_trade_pnl_mismatch(analyzer):

    analyzer.summary.loc[
        0, "net_profit"
    ] = 999.0

    with pytest.raises(ValueError):

        analyzer.validate_reports()


def test_invalid_planned_risk(analyzer):

    analyzer.trades.loc[
        0, "planned_risk"
    ] = 0.0

    with pytest.raises(ValueError):

        analyzer.validate_reports()


def test_period_analysis(analyzer):

    results = analyzer.analyze_periods()

    assert len(results) == 9

    assert results["simulations"].sum() == 9


def test_symbol_consistency(analyzer):

    results = analyzer.analyze_symbols()

    row = results.iloc[0]

    assert row["simulations"] == 9

    assert row["profitable_simulations"] == 6

    assert row[
        "profitability_consistency"
    ] == pytest.approx(66.66666667)

    assert row["baseline_trades"] == 3

    assert row["status"] == "MIXED_PERFORMANCE"


def test_cost_sensitivity(analyzer):

    results = (
        analyzer.analyze_cost_sensitivity()
    )

    assert len(results) == 3

    assert results[
        "moderate_cost_impact"
    ].mean() == pytest.approx(0.50)

    assert results[
        "stressed_cost_impact"
    ].mean() == pytest.approx(1.25)


def test_profitability_lost_under_stress(
    analyzer
):

    results = (
        analyzer.analyze_cost_sensitivity()
    )

    assert results[
        "profitability_lost_under_stress"
    ].sum() == 3


def test_strategy_analysis(analyzer):

    results = analyzer.analyze_strategies()

    assert len(results) == 9

    assert (
        results["total_trades"] == 1
    ).all()

    assert (
        results["research_status"]
        == "INSUFFICIENT_DATA"
    ).all()


def test_no_strategy_trading_authorization(
    analyzer
):

    results = analyzer.analyze_strategies()

    assert not results[
        "trading_authorized"
    ].any()


def test_no_symbol_trading_authorization(
    analyzer
):

    results = analyzer.analyze_symbols()

    assert not results[
        "trading_authorized"
    ].any()


def test_performance_deterioration(
    analyzer
):

    results = (
        analyzer.analyze_deterioration()
    )

    assert results.iloc[0][
        "validation_change"
    ] == pytest.approx(0.0)

    assert results.iloc[0][
        "historical_test_change"
    ] == pytest.approx(0.0)