import pandas as pd

from backtesting.regime_strategy_analyzer import (
    align_trades_with_regimes
)


def sample_regimes():

    return pd.DataFrame({

        "session_date": [
            "2026-11-20",
            "2026-11-23",
            "2026-11-24"
        ],

        "regime": [
            "BULLISH",
            "BEARISH",
            "HIGH_VOLATILITY"
        ]

    })


def test_previous_session_used():

    trades = pd.DataFrame({

        "entry_date": [
            "2026-11-24 05:00:00+00:00"
        ]

    })

    result = align_trades_with_regimes(
        trades,
        sample_regimes()
    )

    assert result.iloc[0]["regime"] == "BEARISH"


def test_weekend_alignment():

    trades = pd.DataFrame({

        "entry_date": [
            "2026-11-23 05:00:00+00:00"
        ]

    })

    result = align_trades_with_regimes(
        trades,
        sample_regimes()
    )

    assert result.iloc[0]["regime"] == "BULLISH"


def test_no_future_data():

    trades = pd.DataFrame({

        "entry_date": [
            "2026-11-19 05:00:00+00:00"
        ]

    })

    result = align_trades_with_regimes(
        trades,
        sample_regimes()
    )

    assert result.iloc[0]["regime"] == "UNKNOWN"


def test_stale_data_rejected():

    trades = pd.DataFrame({

        "entry_date": [
            "2026-12-15 05:00:00+00:00"
        ]

    })

    result = align_trades_with_regimes(
        trades,
        sample_regimes()
    )

    assert result.iloc[0]["regime"] == "UNKNOWN"


def test_trade_count_preserved():

    trades = pd.DataFrame({

        "entry_date": [

            "2026-11-23 05:00:00+00:00",

            "2026-11-24 05:00:00+00:00"

        ]

    })

    result = align_trades_with_regimes(
        trades,
        sample_regimes()
    )

    assert len(result) == len(trades)


def test_unknown_regime_preserved():

    regimes = pd.DataFrame({

        "session_date": [
            "2026-11-20",
            "2026-11-23"
        ],

        "regime": [
            "BULLISH",
            "UNKNOWN"
        ]

    })

    trades = pd.DataFrame({

        "entry_date": [
            "2026-11-24 05:00:00+00:00"
        ]

    })

    result = align_trades_with_regimes(
        trades,
        regimes
    )

    assert result.iloc[0]["regime"] == "UNKNOWN"