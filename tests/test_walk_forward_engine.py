from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest

from backtesting.walk_forward_engine import WalkForwardEngine
from backtesting.execution_model import ExecutionModel


@pytest.fixture
def engine():

    instance = WalkForwardEngine.__new__(
        WalkForwardEngine
    )

    instance.config = {
        "research_start_date": "2024-11-22",
        "research_end_date": "2026-11-23",

        "windows": {
            "training_months": 6,
            "testing_months": 3,
            "step_months": 3
        },

        "execution": {
            "slippage_percent": 0.10,
            "transaction_fee_percent": 0.05
        }
    }

    instance.symbols = ["TEST"]

    instance.warmup = 250

    instance.initial_capital = 100.0

    instance.history_cache = {}

    instance.original_backtest_config = {
        "stop_atr_multiplier": 1.5,
        "risk_reward_ratio": 2.0
    }

    instance.original_strategy_config = {
        "settings": {
            "minimum_score": 55,
            "stop_atr_multiplier": 1.5,
            "minimum_risk_reward": 2.0
        }
    }

    instance.original_execution = ExecutionModel()

    instance.original_prepare_data = (
        lambda symbol: instance.history_cache[symbol].copy()
    )

    instance.engine = SimpleNamespace(
        config=deepcopy(
            instance.original_backtest_config
        ),

        strategy=SimpleNamespace(
            config=deepcopy(
                instance.original_strategy_config
            )
        ),

        execution=instance.original_execution,

        prepare_data=instance.original_prepare_data
    )

    return instance


@pytest.fixture
def history():

    dates = pd.date_range(
        "2022-01-01",
        "2026-12-31",
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
def candidate():

    return {
        "candidate_id": "C012",
        "minimum_score": 75,
        "stop_atr_multiplier": 2.0,
        "risk_reward_ratio": 3.0
    }


def test_valid_configuration(engine):

    engine.validate_configuration()


def test_six_research_windows(engine):

    assert len(engine.generate_windows()) == 6


def test_first_window_dates(engine):

    window = engine.generate_windows()[0]

    assert window["window_id"] == "W001"
    assert window["training_start"] == "2024-11-22"
    assert window["training_end"] == "2025-05-21"
    assert window["testing_start"] == "2025-05-22"
    assert window["testing_end"] == "2025-08-21"


def test_last_window_dates(engine):

    window = engine.generate_windows()[-1]

    assert window["window_id"] == "W006"
    assert window["testing_end"] == "2026-11-21"


def test_training_precedes_testing(engine):

    for window in engine.generate_windows():

        assert (
            pd.Timestamp(window["training_end"])
            < pd.Timestamp(window["testing_start"])
        )


def test_testing_windows_do_not_overlap(engine):

    windows = engine.generate_windows()

    for previous, current in zip(
        windows,
        windows[1:]
    ):

        assert (
            pd.Timestamp(previous["testing_end"])
            < pd.Timestamp(current["testing_start"])
        )


def test_research_end_respected(engine):

    end = pd.Timestamp(
        engine.config["research_end_date"]
    )

    for window in engine.generate_windows():

        assert pd.Timestamp(window["testing_end"]) <= end


def test_overlapping_testing_configuration_rejected(
    engine
):

    engine.config["windows"]["step_months"] = 1

    with pytest.raises(ValueError):

        engine.validate_configuration()


def test_invalid_date_range_rejected(engine):

    engine.config["research_end_date"] = "2020-01-01"

    with pytest.raises(ValueError):

        engine.validate_configuration()


def test_invalid_initial_capital(engine):

    engine.initial_capital = 0.0

    with pytest.raises(ValueError):

        engine.validate_configuration()


def test_invalid_warmup(engine):

    engine.warmup = 100

    with pytest.raises(ValueError):

        engine.validate_configuration()


def test_training_period_boundaries(engine, history):

    engine.history_cache["TEST"] = history

    window = engine.generate_windows()[0]

    result = engine.prepare_period(
        "TEST",
        window["training_start"],
        window["training_end"]
    )

    assert result.index[engine.warmup].date() == (
        pd.Timestamp(window["training_start"]).date()
    )

    assert result.index[-1].date() <= (
        pd.Timestamp(window["training_end"]).date()
    )


def test_testing_period_boundaries(engine, history):

    engine.history_cache["TEST"] = history

    window = engine.generate_windows()[0]

    result = engine.prepare_period(
        "TEST",
        window["testing_start"],
        window["testing_end"]
    )

    assert result.index[engine.warmup].date() == (
        pd.Timestamp(window["testing_start"]).date()
    )

    assert result.index[-1].date() <= (
        pd.Timestamp(window["testing_end"]).date()
    )


def test_training_excludes_testing_candles(
    engine,
    history
):

    engine.history_cache["TEST"] = history

    window = engine.generate_windows()[0]

    result = engine.prepare_period(
        "TEST",
        window["training_start"],
        window["training_end"]
    )

    assert result.index[-1].date() < (
        pd.Timestamp(window["testing_start"]).date()
    )


def test_insufficient_warmup_rejected(
    engine,
    history
):

    engine.history_cache["TEST"] = history

    with pytest.raises(ValueError):

        engine.prepare_period(
            "TEST",
            "2022-02-01",
            "2022-03-01"
        )


def test_missing_period_rejected(engine, history):

    engine.history_cache["TEST"] = history

    with pytest.raises(ValueError):

        engine.prepare_period(
            "TEST",
            "2030-01-01",
            "2030-03-01"
        )


def test_history_cache_returns_copy(engine, history):

    engine.history_cache["TEST"] = history

    result = engine.load_history("TEST")

    result.iloc[0, 0] = 999.0

    assert engine.history_cache["TEST"].iloc[0, 0] == 100.0


def test_candidate_parameters_applied(
    engine,
    candidate
):

    engine.apply_candidate(candidate)

    assert engine.engine.config[
        "stop_atr_multiplier"
    ] == 2.0

    assert engine.engine.config[
        "risk_reward_ratio"
    ] == 3.0

    assert engine.engine.strategy.config[
        "settings"
    ]["minimum_score"] == 75


def test_execution_costs_applied(engine, candidate):

    engine.apply_candidate(candidate)

    assert engine.engine.execution.slippage == (
        pytest.approx(0.001)
    )

    assert engine.engine.execution.fee == (
        pytest.approx(0.0005)
    )


def test_original_parameters_restored(
    engine,
    candidate
):

    engine.apply_candidate(candidate)

    engine.restore_engine()

    assert (
        engine.engine.config
        == engine.original_backtest_config
    )

    assert (
        engine.engine.strategy.config
        == engine.original_strategy_config
    )

    assert (
        engine.engine.execution
        is engine.original_execution
    )


def test_original_config_not_modified(
    engine,
    candidate
):

    original = deepcopy(
        engine.original_strategy_config
    )

    engine.apply_candidate(candidate)

    assert engine.original_strategy_config == original


def test_valid_trade_dates(engine):

    trades = [
        {
            "entry_date": "2025-05-22T04:00:00+00:00",
            "exit_date": "2025-05-23T04:00:00+00:00"
        }
    ]

    engine.verify_trade_dates(
        trades,
        "2025-05-22",
        "2025-08-21"
    )


def test_trade_before_period_rejected(engine):

    trades = [
        {
            "entry_date": "2025-05-21T04:00:00+00:00",
            "exit_date": "2025-05-23T04:00:00+00:00"
        }
    ]

    with pytest.raises(RuntimeError):

        engine.verify_trade_dates(
            trades,
            "2025-05-22",
            "2025-08-21"
        )


def test_trade_after_period_rejected(engine):

    trades = [
        {
            "entry_date": "2025-08-20T04:00:00+00:00",
            "exit_date": "2025-08-22T04:00:00+00:00"
        }
    ]

    with pytest.raises(RuntimeError):

        engine.verify_trade_dates(
            trades,
            "2025-05-22",
            "2025-08-21"
        )


def test_cash_only_results(engine, history):

    engine.history_cache["TEST"] = history

    window = engine.generate_windows()[0]

    summaries, trades, curves = (
        engine.create_cash_results(window)
    )

    assert len(summaries) == 1
    assert trades == []
    assert len(curves) == 1

    summary = summaries[0]

    assert summary["candidate_id"] == "NONE"
    assert summary["research_status"] == "CASH_ONLY"
    assert summary["final_capital"] == 100.0
    assert summary["total_return"] == 0.0
    assert summary["total_trades"] == 0
    assert summary["trading_authorized"] is False

    assert (curves[0]["equity"] == 100.0).all()


def test_invalid_data_role_rejected(
    engine,
    candidate
):

    window = engine.generate_windows()[0]

    with pytest.raises(ValueError):

        engine.simulate_candidate(
            candidate,
            window,
            "INVALID"
        )


def test_restoration_after_simulation_failure(
    engine,
    candidate,
    history
):

    engine.history_cache["TEST"] = history

    def failing_run(symbol):

        raise RuntimeError("Synthetic simulation failure")

    engine.engine.run = failing_run

    window = engine.generate_windows()[0]

    with pytest.raises(RuntimeError):

        engine.simulate_candidate(
            candidate,
            window,
            "TRAINING"
        )

    assert (
        engine.engine.config
        == engine.original_backtest_config
    )

    assert (
        engine.engine.strategy.config
        == engine.original_strategy_config
    )

    assert (
        engine.engine.execution
        is engine.original_execution
    )

    assert (
        engine.engine.prepare_data
        is engine.original_prepare_data
    )