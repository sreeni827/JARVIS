import pandas as pd
import pytest

from intelligence.market_regime import MarketRegimeEngine


@pytest.fixture
def engine():

    return MarketRegimeEngine(market=object())


def test_bullish_market(engine):

    assert engine.classify_regime(85, 1.2) == "BULLISH"


def test_bearish_market(engine):

    assert engine.classify_regime(20, 1.5) == "BEARISH"


def test_sideways_market(engine):

    assert engine.classify_regime(50, 1.4) == "SIDEWAYS"


def test_high_volatility(engine):

    assert engine.classify_regime(80, 3.5) == "HIGH_VOLATILITY"


def test_bullish_boundary(engine):

    assert engine.classify_regime(70, 1.0) == "BULLISH"


def test_bearish_boundary(engine):

    assert engine.classify_regime(30, 1.0) == "BEARISH"


def test_volatility_boundary(engine):

    assert engine.classify_regime(50, 2.5) == "HIGH_VOLATILITY"


def test_volatility_priority(engine):

    assert engine.classify_regime(10, 4.0) == "HIGH_VOLATILITY"


def test_invalid_score(engine):

    assert engine.classify_regime(float("nan"), 1.0) == "UNKNOWN"


def test_invalid_volatility(engine):

    assert engine.classify_regime(50, -1) == "UNKNOWN"


def test_bullish_indicator_score(engine):

    row = pd.Series(
        {
            "close": 110,
            "SMA_20": 108,
            "SMA_50": 105,
            "SMA_200": 100,
            "ROC_10": 5,
            "RSI_14": 70,
            "ATR_14": 2,
        }
    )

    score, volatility = engine.calculate_score(row)

    assert score == pytest.approx(100)

    assert volatility == pytest.approx(2 / 110 * 100)


def test_bearish_indicator_score(engine):

    row = pd.Series(
        {
            "close": 90,
            "SMA_20": 92,
            "SMA_50": 95,
            "SMA_200": 100,
            "ROC_10": -5,
            "RSI_14": 30,
            "ATR_14": 2,
        }
    )

    score, _ = engine.calculate_score(row)

    assert score == pytest.approx(0)


def test_missing_benchmark(engine, monkeypatch):

    def unavailable(symbol):

        raise RuntimeError("Data unavailable")

    monkeypatch.setattr(engine, "analyze_benchmark", unavailable)

    result = engine.analyze_market()

    assert result["regime"] == "UNKNOWN"

    assert len(result["errors"]) == 4


def test_mismatched_dates(engine, monkeypatch):

    def benchmark(symbol):

        return {
            "symbol": symbol,
            "session_date": ("2026-11-20" if symbol == "SPY" else "2026-11-23"),
            "score": 80,
            "atr_percent": 1.5,
            "weight": engine.benchmarks[symbol],
        }

    monkeypatch.setattr(engine, "analyze_benchmark", benchmark)

    result = engine.analyze_market()

    assert result["regime"] == "UNKNOWN"
