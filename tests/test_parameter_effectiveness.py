from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest

from strategies.strategy_engine import StrategyEngine
from backtesting.walk_forward_engine import WalkForwardEngine
from backtesting.execution_model import ExecutionModel


@pytest.fixture
def strategy():

    return StrategyEngine()


@pytest.fixture
def bullish_row():

    return pd.Series({

        "symbol": "TEST",

        "close": 100.0,

        "score": 70.0,

        "rsi": 60.0,

        "atr": 2.0,

        "momentum": 3.0,

        "relative_volume": 1.2,

        "macd": 1.5,

        "macd_signal": 1.0,

        "ema_9": 99.0,

        "ema_21": 97.0,

        "sma_20": 96.0,

        "sma_50": 92.0,

        "sma_200": 85.0,

        "support": 90.0,

        "resistance": 110.0,

        "trend": "BULLISH"

    })


@pytest.fixture
def engine(strategy):

    instance = WalkForwardEngine.__new__(
        WalkForwardEngine
    )

    instance.config = {

        "execution": {

            "slippage_percent": 0.10,

            "transaction_fee_percent": 0.05

        }

    }

    instance.original_backtest_config = {

        "stop_atr_multiplier": 1.5,

        "risk_reward_ratio": 2.0

    }

    instance.original_strategy_config = deepcopy(
        strategy.config
    )

    instance.original_execution = ExecutionModel()

    instance.original_prepare_data = (
        lambda symbol: None
    )

    instance.engine = SimpleNamespace(

        config=deepcopy(
            instance.original_backtest_config
        ),

        strategy=strategy,

        execution=instance.original_execution,

        prepare_data=instance.original_prepare_data

    )

    return instance


def make_candidate(
    minimum_score=55,
    stop_multiplier=1.5,
    risk_reward=2.0
):

    return {

        "candidate_id": "TEST",

        "minimum_score": minimum_score,

        "stop_atr_multiplier": stop_multiplier,

        "risk_reward_ratio": risk_reward

    }


def test_settings_reference_current_config(strategy):

    assert (
        strategy.settings
        is strategy.config["settings"]
    )


def test_settings_follow_replaced_config(strategy):

    strategy.config = deepcopy(strategy.config)

    strategy.config["settings"]["minimum_score"] = 75

    assert strategy.settings["minimum_score"] == 75


@pytest.mark.parametrize(
    "minimum_score",
    [55, 65, 75]
)
def test_effective_candidate_score(
    engine,
    minimum_score
):

    engine.apply_candidate(
        make_candidate(minimum_score)
    )

    assert (
        engine.engine.strategy.settings["minimum_score"]
        == minimum_score
    )


def test_settings_reference_after_candidate_application(
    engine
):

    engine.apply_candidate(
        make_candidate(75)
    )

    assert (
        engine.engine.strategy.settings
        is engine.engine.strategy.config["settings"]
    )


def test_score_70_passes_threshold_55(
    engine,
    bullish_row
):

    engine.apply_candidate(
        make_candidate(55)
    )

    signal = engine.engine.strategy.generate_signal(
        bullish_row
    )

    assert signal["signal"] == "BUY"


def test_score_70_passes_threshold_65(
    engine,
    bullish_row
):

    engine.apply_candidate(
        make_candidate(65)
    )

    signal = engine.engine.strategy.generate_signal(
        bullish_row
    )

    assert signal["signal"] == "BUY"


def test_score_70_fails_threshold_75(
    engine,
    bullish_row
):

    engine.apply_candidate(
        make_candidate(75)
    )

    signal = engine.engine.strategy.generate_signal(
        bullish_row
    )

    assert signal["signal"] == "NO_TRADE"

    assert "score below minimum" in (
        signal["reason"].lower()
    )


def test_threshold_equality_allowed(
    engine,
    bullish_row
):

    bullish_row["score"] = 75.0

    engine.apply_candidate(
        make_candidate(75)
    )

    signal = engine.engine.strategy.generate_signal(
        bullish_row
    )

    assert signal["signal"] == "BUY"


def test_atr_multiplier_effective(engine):

    engine.apply_candidate(
        make_candidate(
            stop_multiplier=2.0
        )
    )

    assert (
        engine.engine.strategy.settings[
            "stop_atr_multiplier"
        ]
        == 2.0
    )

    assert (
        engine.engine.config["stop_atr_multiplier"]
        == 2.0
    )


def test_risk_reward_effective(engine):

    engine.apply_candidate(
        make_candidate(
            risk_reward=3.0
        )
    )

    assert (
        engine.engine.strategy.settings[
            "minimum_risk_reward"
        ]
        == 3.0
    )

    assert (
        engine.engine.config["risk_reward_ratio"]
        == 3.0
    )


def test_candidate_changes_stop_distance(
    engine,
    bullish_row
):

    engine.apply_candidate(
        make_candidate(
            stop_multiplier=1.5
        )
    )

    first = engine.engine.strategy.generate_signal(
        bullish_row
    )

    engine.apply_candidate(
        make_candidate(
            stop_multiplier=2.0
        )
    )

    second = engine.engine.strategy.generate_signal(
        bullish_row
    )

    assert first["signal"] == "BUY"
    assert second["signal"] == "BUY"

    assert first["stop_loss"] == pytest.approx(97.0)
    assert second["stop_loss"] == pytest.approx(96.0)


def test_candidate_changes_profit_target(
    engine,
    bullish_row
):

    engine.apply_candidate(
        make_candidate(
            risk_reward=2.0
        )
    )

    first = engine.engine.strategy.generate_signal(
        bullish_row
    )

    engine.apply_candidate(
        make_candidate(
            risk_reward=3.0
        )
    )

    second = engine.engine.strategy.generate_signal(
        bullish_row
    )

    assert first["signal"] == "BUY"
    assert second["signal"] == "BUY"

    assert first["take_profit"] == pytest.approx(106.0)
    assert second["take_profit"] == pytest.approx(109.0)


def test_sequential_candidates_do_not_leak_settings(
    engine
):

    for minimum_score in [75, 55, 65, 75, 55]:

        engine.apply_candidate(
            make_candidate(minimum_score)
        )

        assert (
            engine.engine.strategy.settings[
                "minimum_score"
            ]
            == minimum_score
        )


def test_effective_settings_restored(engine):

    original = deepcopy(
        engine.original_strategy_config["settings"]
    )

    engine.apply_candidate(
        make_candidate(
            minimum_score=75,
            stop_multiplier=2.0,
            risk_reward=3.0
        )
    )

    engine.restore_engine()

    assert engine.engine.strategy.settings == original

    assert (
        engine.engine.strategy.settings
        is engine.engine.strategy.config["settings"]
    )


def test_signal_behavior_restored(
    engine,
    bullish_row
):

    engine.apply_candidate(
        make_candidate(75)
    )

    rejected = engine.engine.strategy.generate_signal(
        bullish_row
    )

    assert rejected["signal"] == "NO_TRADE"

    engine.restore_engine()

    restored = engine.engine.strategy.generate_signal(
        bullish_row
    )

    assert restored["signal"] == "BUY"


def test_original_configuration_preserved(engine):

    original = deepcopy(
        engine.original_strategy_config
    )

    engine.apply_candidate(
        make_candidate(
            minimum_score=75,
            stop_multiplier=2.0,
            risk_reward=3.0
        )
    )

    assert engine.original_strategy_config == original
    