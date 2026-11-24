from decimal import Decimal
from types import SimpleNamespace

import pandas as pd
import pytest

from backtesting.backtest_engine import BacktestEngine
from backtesting.execution_model import ExecutionModel


D = Decimal


class FakeRanker:

    def calculate_score(self, row):

        return 80, {}, 2.0


class FakeStrategy:

    def generate_signal(self, row):

        return {
            "signal": "BUY",
            "strategy": "TEST_STRATEGY"
        }


class FakeGuardian:

    def __init__(
        self,
        approved=True,
        drawdown_limit="5"
    ):

        self.approved = approved

        self.config = SimpleNamespace(

            min_order_value=D("1"),

            max_daily_loss_percent=D("1"),

            max_drawdown_percent=D(
                drawdown_limit
            )

        )

        self.requests = []

    def evaluate(self, account, trade):

        self.requests.append(
            (account, trade)
        )

        return SimpleNamespace(

            approved=self.approved,

            quantity=D("0.1"),

            estimated_risk=D("0.5")

        )


def make_data(candles):

    data = pd.DataFrame(

        candles,

        columns=[
            "open",
            "high",
            "low",
            "close"
        ]

    )

    data["atr"] = 2.0

    data["symbol"] = "TEST"

    data.index = pd.date_range(

        "2025-01-02",

        periods=len(data),

        freq="B",

        tz="UTC"

    )

    return data


def make_engine(
    candles,
    approved=True,
    drawdown_limit="5"
):

    engine = BacktestEngine.__new__(
        BacktestEngine
    )

    engine.config = {

        "warmup_candles": 1,

        "initial_capital": 100.0,

        "stop_atr_multiplier": 1.5,

        "risk_reward_ratio": 2.0,

        "maximum_holding_days": 20

    }

    engine.ranker = FakeRanker()

    engine.strategy = FakeStrategy()

    engine.guardian = FakeGuardian(

        approved=approved,

        drawdown_limit=drawdown_limit

    )

    engine.execution = ExecutionModel(

        slippage_percent=0.05,

        fee_percent=0.02

    )

    data = make_data(candles)

    engine.prepare_data = lambda symbol: data

    return engine


def test_profitable_trade():

    engine = make_engine([

        [100, 101, 99, 100],

        [100, 108, 99, 107],

        [107, 108, 106, 107]

    ])

    metrics, trades, curve = engine.run(
        "TEST"
    )

    assert len(trades) >= 1

    assert trades[0]["pnl"] > 0

    assert (
        trades[0]["exit_reason"]
        == "TAKE_PROFIT"
    )


def test_stop_loss_trade():

    engine = make_engine([

        [100, 101, 99, 100],

        [100, 102, 95, 96],

        [96, 97, 95, 96]

    ])

    metrics, trades, curve = engine.run(
        "TEST"
    )

    assert trades[0]["pnl"] < 0

    assert (
        trades[0]["exit_reason"]
        == "STOP_LOSS"
    )


def test_risk_rejection_blocks_trades():

    engine = make_engine(

        [

            [100, 101, 99, 100],

            [100, 108, 99, 107],

            [107, 108, 106, 107]

        ],

        approved=False

    )

    metrics, trades, curve = engine.run(
        "TEST"
    )

    assert len(trades) == 0

    assert metrics["total_trades"] == 0

    assert metrics["final_capital"] == (
        pytest.approx(100)
    )


def test_overnight_gap_exceeds_planned_risk():

    engine = make_engine([

        [100, 101, 99, 100],

        [100, 102, 99, 101],

        [90, 92, 88, 91]

    ])

    metrics, trades, curve = engine.run(
        "TEST"
    )

    trade = trades[0]

    assert trade["exit_reason"] == "STOP_GAP"

    assert (
        abs(trade["pnl"])
        > trade["planned_risk"]
    )


def test_normal_stop_matches_planned_risk():

    engine = make_engine([

        [100, 101, 99, 100],

        [100, 102, 95, 96],

        [96, 97, 95, 96]

    ])

    metrics, trades, curve = engine.run(
        "TEST"
    )

    trade = trades[0]

    assert abs(trade["pnl"]) == pytest.approx(

        trade["planned_risk"],

        abs=0.000001

    )


def test_equity_reconciliation():

    engine = make_engine([

        [100, 101, 99, 100],

        [100, 108, 99, 107],

        [107, 108, 106, 107]

    ])

    metrics, trades, curve = engine.run(
        "TEST"
    )

    expected_equity = (

        100

        + sum(
            trade["pnl"]
            for trade in trades
        )

    )

    assert metrics["final_capital"] == (
        pytest.approx(expected_equity)
    )

    assert curve["equity"].iloc[-1] == (
        pytest.approx(expected_equity)
    )


def test_drawdown_halts_future_entries():

    engine = make_engine(

        [

            [100, 101, 99, 100],

            [100, 101, 99, 100],

            [80, 82, 79, 81],

            [81, 90, 80, 89]

        ],

        drawdown_limit="1"

    )

    metrics, trades, curve = engine.run(
        "TEST"
    )

    assert metrics["trading_halted"]

    assert metrics["drawdown_events"] == 1

    assert len(trades) == 1

    assert len(engine.guardian.requests) == 1


def test_daily_loss_event_detected():

    engine = make_engine([

        [100, 101, 99, 100],

        [100, 101, 99, 100],

        [80, 82, 79, 81]

    ])

    metrics, trades, curve = engine.run(
        "TEST"
    )

    assert metrics["daily_loss_events"] >= 1


def test_final_session_liquidation():

    engine = make_engine([

        [100, 101, 99, 100],

        [100, 102, 99, 101],

        [101, 103, 100, 102]

    ])

    metrics, trades, curve = engine.run(
        "TEST"
    )

    assert len(trades) == 1

    assert (

        trades[0]["exit_reason"]

        == "END_OF_BACKTEST"

    )


def test_execution_version():

    engine = make_engine([

        [100, 101, 99, 100],

        [100, 102, 99, 101],

        [101, 103, 100, 102]

    ])

    metrics, trades, curve = engine.run(
        "TEST"
    )

    assert metrics["execution_version"] == (
        "0.8.5B"
    )


def test_drawdown_liquidation_next_session():

    engine = make_engine(

        [

            [100, 101, 99, 100],

            [100, 101, 98, 98],

            [99, 101, 98, 100],

            [100, 102, 99, 101]

        ],

        drawdown_limit="0.1"

    )

    metrics, trades, curve = engine.run(
        "TEST"
    )

    assert metrics["trading_halted"]

    assert metrics["drawdown_events"] == 1

    assert len(trades) == 1

    trade = trades[0]

    assert trade["exit_reason"] == (
        "DRAWDOWN_EXIT"
    )

    assert trade["exit_price"] == (
        pytest.approx(
            engine.execution.sell_price(99)
        )
    )

    data = engine.prepare_data("TEST")

    assert trade["exit_date"] == str(
        data.index[2]
    )

    assert len(engine.guardian.requests) == 1
    