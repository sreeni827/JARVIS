from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from backtesting.walk_forward_analyzer import WalkForwardAnalyzer


@pytest.fixture
def analyzer():

    instance = WalkForwardAnalyzer.__new__(WalkForwardAnalyzer)

    instance.symbols = ["TEST"]

    instance.windows = pd.DataFrame([{"window_id": "W001"}])

    instance.training = pd.DataFrame(
        [
            {
                "window_id": "W001",
                "candidate_id": "C001",
                "minimum_score": 55,
                "stop_atr_multiplier": 1.5,
                "risk_reward_ratio": 2.0,
                "total_trades": 40,
                "average_return": 2.0,
                "profit_factor": 1.5,
                "expectancy": 0.10,
                "maximum_drawdown": 2.0,
                "risk_breach_percent": 0.0,
                "qualified": True,
                "data_role": "TRAINING",
                "trading_authorized": False,
            }
        ]
    )

    instance.selections = pd.DataFrame(
        [
            {
                "window_id": "W001",
                "candidate_id": "C001",
                "status": "SELECTED_FOR_HISTORICAL_TEST",
                "qualified_candidates": 1,
                "trading_authorized": False,
            }
        ]
    )

    instance.testing = pd.DataFrame(
        [
            {
                "window_id": "W001",
                "candidate_id": "C001",
                "symbol": "TEST",
                "total_trades": 2,
                "initial_capital": 100.0,
                "final_capital": 101.0,
                "net_profit": 1.0,
                "total_return": 1.0,
                "max_drawdown": 100 / 102,
                "period_start": "2025-01-02",
                "period_end": "2025-01-31",
                "data_role": "TESTING",
                "trading_authorized": False,
            }
        ]
    )

    instance.trades = pd.DataFrame(
        [
            {
                "window_id": "W001",
                "candidate_id": "C001",
                "symbol": "TEST",
                "strategy": "MOMENTUM",
                "pnl": 2.0,
                "planned_risk": 0.50,
                "entry_date": "2025-01-02T15:00:00Z",
                "exit_date": "2025-01-03T15:00:00Z",
                "exit_reason": "TAKE_PROFIT",
                "data_role": "TESTING",
                "trading_authorized": False,
            },
            {
                "window_id": "W001",
                "candidate_id": "C001",
                "symbol": "TEST",
                "strategy": "MOMENTUM",
                "pnl": -1.0,
                "planned_risk": 0.50,
                "entry_date": "2025-01-06T15:00:00Z",
                "exit_date": "2025-01-07T15:00:00Z",
                "exit_reason": "STOP_GAP",
                "data_role": "TESTING",
                "trading_authorized": False,
            },
        ]
    )

    instance.equity = pd.DataFrame(
        [
            {
                "window_id": "W001",
                "candidate_id": "C001",
                "symbol": "TEST",
                "timestamp": "2025-01-02T15:00:00Z",
                "equity": 100.0,
                "data_role": "TESTING",
                "trading_authorized": False,
            },
            {
                "window_id": "W001",
                "candidate_id": "C001",
                "symbol": "TEST",
                "timestamp": "2025-01-03T15:00:00Z",
                "equity": 102.0,
                "data_role": "TESTING",
                "trading_authorized": False,
            },
            {
                "window_id": "W001",
                "candidate_id": "C001",
                "symbol": "TEST",
                "timestamp": "2025-01-07T15:00:00Z",
                "equity": 101.0,
                "data_role": "TESTING",
                "trading_authorized": False,
            },
        ]
    )

    candidate = {
        "candidate_id": "C001",
        "minimum_score": 55,
        "stop_atr_multiplier": 1.5,
        "risk_reward_ratio": 2.0,
    }

    instance.selector = SimpleNamespace(
        generate_candidates=lambda: [candidate],
        qualify_candidate=lambda row: (True, "Qualified"),
        select=lambda evaluations: {
            "status": "SELECTED_FOR_HISTORICAL_TEST",
            "selected_candidate": candidate,
        },
    )

    return instance


def test_valid_reports(analyzer):

    analyzer.validate_reports()


def test_boolean_conversion():

    values = pd.Series([True, False, "TRUE", "false"])

    result = WalkForwardAnalyzer.boolean_series(values)

    assert result.tolist() == [True, False, True, False]


def test_invalid_boolean_rejected():

    with pytest.raises(ValueError):
        WalkForwardAnalyzer.boolean_series(pd.Series(["unknown"]))


def test_missing_columns_rejected():

    with pytest.raises(ValueError):
        WalkForwardAnalyzer.require_columns(
            pd.DataFrame({"a": [1]}), ["a", "b"], "TEST"
        )


def test_duplicate_windows_rejected(analyzer):

    analyzer.windows = pd.concat(
        [analyzer.windows, analyzer.windows], ignore_index=True
    )

    with pytest.raises(ValueError):
        analyzer.validate_reports()


def test_missing_training_evaluation(analyzer):

    analyzer.training = analyzer.training.iloc[0:0]

    with pytest.raises(ValueError):
        analyzer.validate_reports()


def test_duplicate_training_evaluation(analyzer):

    analyzer.training = pd.concat(
        [analyzer.training, analyzer.training], ignore_index=True
    )

    with pytest.raises(ValueError):
        analyzer.validate_reports()


def test_missing_testing_record(analyzer):

    analyzer.testing = analyzer.testing.iloc[0:0]

    with pytest.raises(ValueError):
        analyzer.validate_reports()


def test_duplicate_testing_record(analyzer):

    analyzer.testing = pd.concat(
        [analyzer.testing, analyzer.testing], ignore_index=True
    )

    with pytest.raises(ValueError):
        analyzer.validate_reports()


def test_missing_selection(analyzer):

    analyzer.selections = analyzer.selections.iloc[0:0]

    with pytest.raises(ValueError):
        analyzer.validate_reports()


@pytest.mark.parametrize(
    "attribute", ["training", "selections", "testing", "trades", "equity"]
)
def test_trading_authorization_rejected(analyzer, attribute):

    frame = getattr(analyzer, attribute)

    frame.loc[frame.index[0], "trading_authorized"] = True

    with pytest.raises(RuntimeError):
        analyzer.validate_reports()


def test_invalid_training_role(analyzer):

    analyzer.training.loc[0, "data_role"] = "TESTING"

    with pytest.raises(ValueError):
        analyzer.validate_reports()


@pytest.mark.parametrize("attribute", ["testing", "trades", "equity"])
def test_invalid_testing_role(analyzer, attribute):

    frame = getattr(analyzer, attribute)

    frame.loc[0, "data_role"] = "TRAINING"

    with pytest.raises(ValueError):
        analyzer.validate_reports()


def test_invalid_training_metric(analyzer):

    analyzer.training.loc[0, "average_return"] = np.nan

    with pytest.raises(ValueError):
        analyzer.validate_reports()


def test_infinite_profit_factor_allowed(analyzer):

    analyzer.training.loc[0, "profit_factor"] = np.inf

    analyzer.validate_reports()


def test_negative_profit_factor_rejected(analyzer):

    analyzer.training.loc[0, "profit_factor"] = -1.0

    with pytest.raises(ValueError):
        analyzer.validate_reports()


def test_candidate_parameter_mismatch(analyzer):

    analyzer.training.loc[0, "minimum_score"] = 75

    with pytest.raises(ValueError):
        analyzer.validate_reports()


def test_qualification_mismatch(analyzer):

    analyzer.training.loc[0, "qualified"] = False

    with pytest.raises(ValueError):
        analyzer.validate_reports()


def test_selection_status_mismatch(analyzer):

    analyzer.selections.loc[0, "status"] = "NO_SELECTION"

    with pytest.raises(ValueError):
        analyzer.validate_reports()


def test_selected_candidate_mismatch(analyzer):

    analyzer.selections.loc[0, "candidate_id"] = "C999"

    with pytest.raises(ValueError):
        analyzer.validate_reports()


def test_qualified_candidate_count_mismatch(analyzer):

    analyzer.selections.loc[0, "qualified_candidates"] = 2

    with pytest.raises(ValueError):
        analyzer.validate_reports()


def test_testing_candidate_mismatch(analyzer):

    analyzer.testing.loc[0, "candidate_id"] = "C999"

    with pytest.raises(ValueError):
        analyzer.validate_reports()


def test_trade_count_mismatch(analyzer):

    analyzer.testing.loc[0, "total_trades"] = 3

    with pytest.raises(ValueError):
        analyzer.validate_reports()


def test_invalid_planned_risk(analyzer):

    analyzer.trades.loc[0, "planned_risk"] = 0.0

    with pytest.raises(ValueError):
        analyzer.validate_reports()


def test_nonfinite_trade_pnl(analyzer):

    analyzer.trades.loc[0, "pnl"] = np.inf

    with pytest.raises(ValueError):
        analyzer.validate_reports()


def test_pnl_mismatch(analyzer):

    analyzer.testing.loc[0, "net_profit"] = 5.0

    with pytest.raises(ValueError):
        analyzer.validate_reports()


def test_final_capital_mismatch(analyzer):

    analyzer.testing.loc[0, "final_capital"] = 105.0

    with pytest.raises(ValueError):
        analyzer.validate_reports()


def test_final_equity_mismatch(analyzer):

    analyzer.equity.loc[2, "equity"] = 105.0

    with pytest.raises(ValueError):
        analyzer.validate_reports()


def test_return_mismatch(analyzer):

    analyzer.testing.loc[0, "total_return"] = 5.0

    with pytest.raises(ValueError):
        analyzer.validate_reports()


def test_drawdown_mismatch(analyzer):

    analyzer.testing.loc[0, "max_drawdown"] = 5.0

    with pytest.raises(ValueError):
        analyzer.validate_reports()


def test_invalid_equity(analyzer):

    analyzer.equity.loc[0, "equity"] = -100.0

    with pytest.raises(ValueError):
        analyzer.validate_reports()


def test_missing_equity_coverage(analyzer):

    analyzer.equity = analyzer.equity.iloc[0:0]

    with pytest.raises(ValueError):
        analyzer.validate_reports()


def test_duplicate_equity_timestamp(analyzer):

    analyzer.equity = pd.concat(
        [analyzer.equity, analyzer.equity.iloc[[0]]], ignore_index=True
    )

    with pytest.raises(ValueError):
        analyzer.validate_reports()


def test_trade_before_testing_period(analyzer):

    analyzer.trades.loc[0, "entry_date"] = "2024-12-31T15:00:00Z"

    with pytest.raises(ValueError):
        analyzer.validate_reports()


def test_trade_after_testing_period(analyzer):

    analyzer.trades.loc[1, "exit_date"] = "2025-02-03T15:00:00Z"

    with pytest.raises(ValueError):
        analyzer.validate_reports()


def test_exit_before_entry(analyzer):

    analyzer.trades.loc[1, "exit_date"] = "2025-01-03T15:00:00Z"

    with pytest.raises(ValueError):
        analyzer.validate_reports()


def test_unexpected_trade_symbol(analyzer):

    analyzer.trades.loc[0, "symbol"] = "UNKNOWN"

    with pytest.raises(ValueError):
        analyzer.validate_reports()


def test_trade_metrics(analyzer):

    metrics = analyzer.calculate_trade_metrics(analyzer.trades)

    assert metrics["total_trades"] == 2
    assert metrics["winning_trades"] == 1
    assert metrics["losing_trades"] == 1
    assert metrics["win_rate"] == 50.0
    assert metrics["net_profit"] == 1.0
    assert metrics["profit_factor"] == 2.0
    assert metrics["expectancy"] == 0.5
    assert metrics["risk_breaches"] == 1
    assert metrics["risk_breach_percent"] == 50.0


def test_empty_trade_metrics(analyzer):

    metrics = analyzer.calculate_trade_metrics(analyzer.trades.iloc[0:0])

    assert metrics["total_trades"] == 0
    assert metrics["net_profit"] == 0.0
    assert metrics["profit_factor"] == 0.0
    assert metrics["expectancy"] == 0.0
    assert metrics["risk_breaches"] == 0


def test_all_winning_trades(analyzer):

    trades = analyzer.trades.iloc[[0]]

    metrics = analyzer.calculate_trade_metrics(trades)

    assert metrics["profit_factor"] == np.inf
    assert metrics["win_rate"] == 100.0


def test_all_losing_trades(analyzer):

    trades = analyzer.trades.iloc[[1]]

    metrics = analyzer.calculate_trade_metrics(trades)

    assert metrics["profit_factor"] == 0.0
    assert metrics["win_rate"] == 0.0


def test_risk_breach_boundary(analyzer):

    trades = analyzer.trades.copy()

    trades.loc[1, "pnl"] = -0.50

    metrics = analyzer.calculate_trade_metrics(trades)

    assert metrics["risk_breaches"] == 0


def test_window_analysis(analyzer):

    result = analyzer.analyze_windows().iloc[0]

    assert result["training_average_return"] == 2.0
    assert result["testing_average_return"] == 1.0
    assert result["performance_deterioration"] == 1.0
    assert result["total_trades"] == 2
    assert not bool(result["trading_authorized"])


def test_parameter_stability(analyzer):

    result = analyzer.analyze_parameter_stability().iloc[0]

    assert result["windows_evaluated"] == 1
    assert result["qualified_windows"] == 1
    assert result["qualification_rate"] == 100.0
    assert result["selected_windows"] == 1


def test_parameter_effectiveness_observed(analyzer):

    second = analyzer.training.copy()

    second.loc[0, "candidate_id"] = "C002"
    second.loc[0, "minimum_score"] = 75
    second.loc[0, "average_return"] = 3.0

    analyzer.training = pd.concat([analyzer.training, second], ignore_index=True)

    result = analyzer.analyze_parameter_effectiveness()

    score = result[result["parameter"] == "minimum_score"].iloc[0]

    assert score["unique_returns"] == 2
    assert score["status"] == "EFFECT_OBSERVED"


def test_parameter_no_effect_observed(analyzer):

    second = analyzer.training.copy()

    second.loc[0, "candidate_id"] = "C002"
    second.loc[0, "minimum_score"] = 75

    analyzer.training = pd.concat([analyzer.training, second], ignore_index=True)

    result = analyzer.analyze_parameter_effectiveness()

    score = result[result["parameter"] == "minimum_score"].iloc[0]

    assert score["unique_returns"] == 1
    assert score["status"] == "NO_EFFECT_OBSERVED"


def test_strategy_analysis(analyzer):

    result = analyzer.analyze_strategies().iloc[0]

    assert result["strategy"] == "MOMENTUM"
    assert result["total_trades"] == 2
    assert result["net_profit"] == 1.0
    assert not bool(result["trading_authorized"])


def test_empty_strategy_analysis(analyzer):

    analyzer.trades = analyzer.trades.iloc[0:0]

    result = analyzer.analyze_strategies()

    assert result.empty
    assert "strategy" in result.columns
    assert "trading_authorized" in result.columns


def make_cash_only(analyzer):

    analyzer.training.loc[0, "qualified"] = False

    analyzer.selector.qualify_candidate = lambda row: (False, "Rejected")

    analyzer.selector.select = lambda evaluations: {
        "status": "NO_SELECTION",
        "selected_candidate": None,
    }

    analyzer.selections.loc[0, "candidate_id"] = "NONE"

    analyzer.selections.loc[0, "status"] = "NO_SELECTION"

    analyzer.selections.loc[0, "qualified_candidates"] = 0

    analyzer.testing.loc[0, "candidate_id"] = "NONE"

    analyzer.testing.loc[0, "total_trades"] = 0

    analyzer.testing.loc[0, "final_capital"] = 100.0

    analyzer.testing.loc[0, "net_profit"] = 0.0

    analyzer.testing.loc[0, "total_return"] = 0.0

    analyzer.testing.loc[0, "max_drawdown"] = 0.0

    analyzer.trades = analyzer.trades.iloc[0:0].copy()

    analyzer.equity["candidate_id"] = "NONE"
    analyzer.equity["equity"] = 100.0


def test_valid_cash_only_window(analyzer):

    make_cash_only(analyzer)

    analyzer.validate_reports()


def test_invalid_cash_only_equity(analyzer):

    make_cash_only(analyzer)

    analyzer.equity.loc[1, "equity"] = 101.0

    analyzer.testing.loc[0, "max_drawdown"] = 100 / 101

    with pytest.raises(ValueError):
        analyzer.validate_reports()


def test_cash_only_window_analysis(analyzer):

    make_cash_only(analyzer)

    result = analyzer.analyze_windows().iloc[0]

    assert result["candidate_id"] == "NONE"
    assert result["total_trades"] == 0
    assert result["testing_average_return"] == 0.0

    assert pd.isna(result["training_average_return"])

    assert pd.isna(result["performance_deterioration"])
