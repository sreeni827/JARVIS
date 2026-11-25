from datetime import date, datetime
from decimal import Decimal as D

import numpy as np
import pandas as pd
import pytest

from market.indicators import TechnicalIndicators
from portfolio.historical_data import HistoricalDataEngine
from portfolio.portfolio_config import PortfolioConfig
from portfolio.position_manager import PositionMarketBar

# ============================================================
# OFFLINE MARKET DATA PROVIDER
# ============================================================


class OfflineMarket:
    def __init__(self, history=None):

        self.history = history or {}
        self.calls = []

    def get_historical_data(self, symbol, days=1500):

        self.calls.append((symbol, days))

        if symbol not in self.history:
            raise RuntimeError(f"No offline data for {symbol}")

        return self.history[symbol].copy(deep=True)


# ============================================================
# FIXTURES
# ============================================================


@pytest.fixture
def config():

    settings = PortfolioConfig().config

    settings["universe"]["symbols"] = ["AAPL", "NVDA"]

    settings["benchmarks"]["symbols"] = ["SPY"]

    settings["historical_data"]["start_date"] = "2024-11-22"

    settings["historical_data"]["end_date"] = "2024-12-06"

    return PortfolioConfig(config=settings)


@pytest.fixture
def engine(config):

    return HistoricalDataEngine(config=config, market=OfflineMarket())


def make_history(sessions, symbol="AAPL", offset=0):

    timestamps = (
        pd.DatetimeIndex(pd.to_datetime(list(sessions)))
        .tz_localize("America/New_York")
        .tz_convert("UTC")
    )

    x = np.arange(len(sessions), dtype=float)

    close = 100 + offset + x * 0.08 + np.sin(x / 5)

    opening = close - 0.25

    frame = pd.DataFrame(
        {
            "open": opening,
            "high": close + 1.0,
            "low": opening - 1.0,
            "close": close,
            "volume": 1000000 + x * 100,
        },
        index=timestamps,
    )

    frame.index.name = "timestamp"

    frame["symbol"] = symbol

    frame = frame.set_index("symbol", append=True).reorder_levels(
        ["symbol", "timestamp"]
    )

    return frame


@pytest.fixture
def raw_history(engine):

    warmup, research = engine.build_expected_sessions()

    sessions = warmup + research

    return {
        symbol: make_history(sessions, symbol=symbol, offset=index * 20)
        for index, symbol in enumerate(engine.required_symbols)
    }


@pytest.fixture
def loaded_engine(engine, raw_history):

    engine.load(raw_history)

    return engine


# ============================================================
# INITIALIZATION
# ============================================================


def test_initialization(engine):

    assert engine.loaded is False
    assert engine.sessions == ()
    assert engine.warmup_sessions == ()
    assert engine.loaded_symbols == ()
    assert engine.total_sessions == 0


def test_configured_symbols(engine):

    assert engine.symbols == ("AAPL", "NVDA")


def test_benchmark_symbols(engine):

    assert engine.benchmark_symbols == ("SPY",)


def test_required_symbols(engine):

    assert engine.required_symbols == ("AAPL", "NVDA", "SPY")


def test_timezone(engine):

    assert engine.TIMEZONE == "America/New_York"


def test_warmup_configuration(engine):

    assert engine.warmup_candles == 250


@pytest.mark.parametrize("invalid", [{}, [], True, 123, "invalid"])
def test_invalid_configuration(invalid):

    with pytest.raises(ValueError):
        HistoricalDataEngine(config=invalid, market=OfflineMarket())


def test_benchmark_deduplication(config):

    settings = config.config

    settings["benchmarks"]["symbols"] = ["AAPL", "SPY"]

    engine = HistoricalDataEngine(
        config=PortfolioConfig(config=settings), market=OfflineMarket()
    )

    assert engine.required_symbols == ("AAPL", "NVDA", "SPY")


def test_disabled_benchmarks(config):

    settings = config.config

    settings["benchmarks"]["enabled"] = False

    engine = HistoricalDataEngine(
        config=PortfolioConfig(config=settings), market=OfflineMarket()
    )

    assert engine.benchmark_symbols == ()

    assert engine.required_symbols == ("AAPL", "NVDA")


# ============================================================
# EXCHANGE CALENDAR
# ============================================================


def test_thanksgiving_excluded(engine):

    sessions = engine.get_calendar_sessions(date(2026, 11, 23), date(2026, 11, 30))

    assert len(sessions) == 5

    assert date(2026, 11, 26) not in sessions


def test_early_close_is_valid_session(engine):

    sessions = engine.get_calendar_sessions(date(2026, 11, 27), date(2026, 11, 27))

    assert sessions == (date(2026, 11, 27),)


def test_weekends_excluded(engine):

    sessions = engine.get_calendar_sessions(date(2026, 11, 21), date(2026, 11, 22))

    assert sessions == ()


def test_reversed_calendar_dates(engine):

    with pytest.raises(ValueError):
        engine.get_calendar_sessions(date(2026, 11, 30), date(2026, 11, 23))


@pytest.mark.parametrize(
    "invalid", [None, "2026-11-23", 123, True, datetime(2026, 11, 23)]
)
def test_invalid_calendar_start(engine, invalid):

    with pytest.raises(ValueError):
        engine.get_calendar_sessions(invalid, date(2026, 11, 30))


def test_expected_warmup_count(engine):

    warmup, research = engine.build_expected_sessions()

    assert len(warmup) == 250

    assert len(research) > 0


def test_warmup_precedes_research(engine):

    warmup, research = engine.build_expected_sessions()

    assert warmup[-1] < research[0]

    assert not set(warmup).intersection(research)


def test_expected_sessions_sorted(engine):

    warmup, research = engine.build_expected_sessions()

    sessions = warmup + research

    assert sessions == tuple(sorted(sessions))

    assert len(sessions) == len(set(sessions))


def test_first_research_session(engine):

    _, research = engine.build_expected_sessions()

    assert research[0] == date(2024, 11, 22)


def test_last_research_session(engine):

    _, research = engine.build_expected_sessions()

    assert research[-1] == date(2024, 12, 6)


# ============================================================
# NORMALIZATION
# ============================================================


def test_normalize_multiindex(engine, raw_history):

    frame = engine.normalize_history("AAPL", raw_history["AAPL"])

    assert not isinstance(frame.index, pd.MultiIndex)

    assert frame.index.name == "session"

    assert type(frame.index[0]) is date


def test_normalize_datetime_index(engine, raw_history):

    raw = raw_history["AAPL"].droplevel("symbol")

    frame = engine.normalize_history("AAPL", raw)

    assert type(frame.index[0]) is date


def test_normalization_preserves_input(engine, raw_history):

    raw = raw_history["AAPL"]

    original = raw.copy(deep=True)

    engine.normalize_history("AAPL", raw)

    pd.testing.assert_frame_equal(raw, original)


def test_unsorted_data_is_sorted(engine, raw_history):

    raw = raw_history["AAPL"].iloc[::-1]

    frame = engine.normalize_history("AAPL", raw)

    assert frame.index.is_monotonic_increasing


def test_timezone_conversion(engine):

    raw = make_history([date(2024, 11, 22)])

    frame = engine.normalize_history("AAPL", raw)

    assert frame.index[0] == date(2024, 11, 22)


@pytest.mark.parametrize("invalid", [None, {}, [], 123, "invalid"])
def test_invalid_history_type(engine, invalid):

    with pytest.raises(ValueError):
        engine.normalize_history("AAPL", invalid)


def test_empty_history(engine):

    with pytest.raises(ValueError):
        engine.normalize_history("AAPL", pd.DataFrame())


def test_wrong_symbol(engine, raw_history):

    with pytest.raises(ValueError):
        engine.normalize_history("AAPL", raw_history["NVDA"])


def test_mixed_symbols(engine, raw_history):

    raw = pd.concat([raw_history["AAPL"], raw_history["NVDA"]])

    with pytest.raises(ValueError):
        engine.normalize_history("AAPL", raw)


def test_missing_symbol_index(engine, raw_history):

    raw = raw_history["AAPL"].copy()

    raw.index = raw.index.set_names(["ticker", "timestamp"])

    with pytest.raises(ValueError):
        engine.normalize_history("AAPL", raw)


def test_invalid_timestamp_index(engine, raw_history):

    raw = raw_history["AAPL"].reset_index(drop=True)

    with pytest.raises(ValueError):
        engine.normalize_history("AAPL", raw)


def test_timezone_naive_rejected(engine, raw_history):

    raw = raw_history["AAPL"].droplevel("symbol")

    raw.index = raw.index.tz_localize(None)

    with pytest.raises(ValueError):
        engine.normalize_history("AAPL", raw)


def test_missing_timestamp(engine, raw_history):

    raw = raw_history["AAPL"].droplevel("symbol")

    timestamps = list(raw.index)

    timestamps[0] = pd.NaT

    raw.index = pd.DatetimeIndex(timestamps)

    with pytest.raises(ValueError):
        engine.normalize_history("AAPL", raw)


def test_duplicate_timestamp(engine, raw_history):

    raw = raw_history["AAPL"]

    raw = pd.concat([raw, raw.iloc[[0]]])

    with pytest.raises(ValueError):
        engine.normalize_history("AAPL", raw)


def test_duplicate_session(engine, raw_history):

    raw = raw_history["AAPL"].droplevel("symbol")

    extra = raw.iloc[[0]].copy()

    extra.index = extra.index + pd.Timedelta(hours=1)

    raw = pd.concat([raw, extra])

    with pytest.raises(ValueError):
        engine.normalize_history("AAPL", raw)


@pytest.mark.parametrize("column", ["open", "high", "low", "close", "volume"])
def test_missing_required_column(engine, raw_history, column):

    raw = raw_history["AAPL"].drop(columns=column)

    with pytest.raises(ValueError):
        engine.normalize_history("AAPL", raw)


@pytest.mark.parametrize("column", ["open", "high", "low", "close", "volume"])
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf, True, "invalid"])
def test_invalid_numeric_values(engine, raw_history, column, invalid):

    raw = raw_history["AAPL"].copy()

    raw[column] = raw[column].astype(object)

    raw.iloc[0, raw.columns.get_loc(column)] = invalid

    with pytest.raises(ValueError):
        engine.normalize_history("AAPL", raw)


@pytest.mark.parametrize("column", ["open", "high", "low", "close"])
@pytest.mark.parametrize("invalid", [0, -1])
def test_nonpositive_prices(engine, raw_history, column, invalid):

    raw = raw_history["AAPL"].copy()

    raw.iloc[0, raw.columns.get_loc(column)] = invalid

    with pytest.raises(ValueError):
        engine.normalize_history("AAPL", raw)


def test_negative_volume(engine, raw_history):

    raw = raw_history["AAPL"].copy()

    raw.iloc[0, raw.columns.get_loc("volume")] = -1

    with pytest.raises(ValueError):
        engine.normalize_history("AAPL", raw)


def test_zero_volume_normalization(engine, raw_history):

    raw = raw_history["AAPL"].copy()

    raw.iloc[0, raw.columns.get_loc("volume")] = 0

    frame = engine.normalize_history("AAPL", raw)

    assert frame.iloc[0]["volume"] == 0


def test_invalid_high(engine, raw_history):

    raw = raw_history["AAPL"].copy()

    raw.iloc[0, raw.columns.get_loc("high")] = 1

    with pytest.raises(ValueError):
        engine.normalize_history("AAPL", raw)


def test_invalid_low(engine, raw_history):

    raw = raw_history["AAPL"].copy()

    raw.iloc[0, raw.columns.get_loc("low")] = 10000

    with pytest.raises(ValueError):
        engine.normalize_history("AAPL", raw)


# ============================================================
# SESSION ALIGNMENT
# ============================================================


def test_complete_alignment(engine, raw_history):

    warmup, research = engine.build_expected_sessions()

    expected = warmup + research

    frame = engine.normalize_history("AAPL", raw_history["AAPL"])

    aligned = engine.align_history("AAPL", frame, expected)

    assert tuple(aligned.index) == expected


@pytest.mark.parametrize("location", ["first", "middle", "last"])
def test_missing_session(engine, raw_history, location):

    warmup, research = engine.build_expected_sessions()

    expected = warmup + research

    frame = engine.normalize_history("AAPL", raw_history["AAPL"])

    positions = {"first": 0, "middle": len(expected) // 2, "last": -1}

    damaged = frame.drop(index=expected[positions[location]])

    with pytest.raises(ValueError, match="missing"):
        engine.align_history("AAPL", damaged, expected)


def test_unexpected_weekend_session(engine, raw_history):

    warmup, research = engine.build_expected_sessions()

    expected = warmup + research

    frame = engine.normalize_history("AAPL", raw_history["AAPL"])

    extra = frame.iloc[[0]].copy()

    extra.index = pd.Index([date(2024, 11, 23)], name="session")

    frame = pd.concat([frame, extra])

    with pytest.raises(ValueError, match="unexpected"):
        engine.align_history("AAPL", frame, expected)


def test_empty_expected_sessions(engine, raw_history):

    frame = engine.normalize_history("AAPL", raw_history["AAPL"])

    with pytest.raises(ValueError):
        engine.align_history("AAPL", frame, ())


def test_duplicate_expected_sessions(engine, raw_history):

    warmup, research = engine.build_expected_sessions()

    expected = warmup + research

    frame = engine.normalize_history("AAPL", raw_history["AAPL"])

    with pytest.raises(ValueError):
        engine.align_history("AAPL", frame, expected + (expected[-1],))


def test_unsorted_expected_sessions(engine, raw_history):

    warmup, research = engine.build_expected_sessions()

    expected = warmup + research

    frame = engine.normalize_history("AAPL", raw_history["AAPL"])

    with pytest.raises(ValueError):
        engine.align_history("AAPL", frame, expected[::-1])


def test_outside_research_data_trimmed(engine, raw_history):

    warmup, research = engine.build_expected_sessions()

    expected = warmup + research

    frame = engine.normalize_history("AAPL", raw_history["AAPL"])

    extra = frame.iloc[[0]].copy()

    extra.index = pd.Index([date(2020, 1, 2)], name="session")

    extended = pd.concat([extra, frame])

    aligned = engine.align_history("AAPL", extended, expected)

    assert tuple(aligned.index) == expected


# ============================================================
# HISTORICAL LOADING
# ============================================================


def test_successful_load(loaded_engine):

    assert loaded_engine.loaded

    assert len(loaded_engine.loaded_symbols) == 3

    assert len(loaded_engine.warmup_sessions) == 250


def test_load_returns_session_count(engine, raw_history):

    result = engine.load(raw_history)

    assert result == len(engine.sessions)


def test_injected_data_requires_no_market_calls(engine, raw_history):

    engine.load(raw_history)

    assert engine.market.calls == []


def test_offline_provider_loading(config, raw_history):

    market = OfflineMarket(raw_history)

    engine = HistoricalDataEngine(config=config, market=market)

    engine.load()

    assert len(market.calls) == 3

    assert market.calls == [
        (symbol, engine.historical_days) for symbol in engine.required_symbols
    ]


@pytest.mark.parametrize("symbol", ["AAPL", "NVDA", "SPY"])
def test_missing_symbol_data(engine, raw_history, symbol):

    del raw_history[symbol]

    with pytest.raises(ValueError):
        engine.load(raw_history)

    assert not engine.loaded

    assert engine.loaded_symbols == ()


@pytest.mark.parametrize("invalid", [[], 123, True, "invalid"])
def test_invalid_history_mapping(engine, invalid):

    with pytest.raises(ValueError):
        engine.load(invalid)


def test_failed_initial_load_is_atomic(engine, raw_history):

    raw_history["SPY"] = pd.DataFrame()

    with pytest.raises(ValueError):
        engine.load(raw_history)

    assert not engine.loaded
    assert engine.sessions == ()
    assert engine.loaded_symbols == ()


def test_failed_reload_preserves_previous_data(loaded_engine, raw_history):

    original = loaded_engine.get_history("AAPL")

    raw_history["SPY"] = pd.DataFrame()

    with pytest.raises(ValueError):
        loaded_engine.load(raw_history)

    assert loaded_engine.loaded

    pd.testing.assert_frame_equal(loaded_engine.get_history("AAPL"), original)


def test_all_symbols_share_identical_sessions(loaded_engine):

    expected = loaded_engine.warmup_sessions + loaded_engine.sessions

    for symbol in loaded_engine.loaded_symbols:
        frame = loaded_engine.get_history(symbol)

        assert tuple(frame.index) == expected


# ============================================================
# ACCESS VALIDATION
# ============================================================


def test_unloaded_history_rejected(engine):

    with pytest.raises(RuntimeError):
        engine.get_history("AAPL")


def test_unloaded_signals_rejected(engine):

    with pytest.raises(RuntimeError):
        engine.get_session_signals(date(2024, 11, 22))


def test_unloaded_market_bars_rejected(engine):

    with pytest.raises(RuntimeError):
        engine.get_session_bars(date(2024, 11, 22))


def test_unknown_symbol(loaded_engine):

    with pytest.raises(ValueError):
        loaded_engine.get_history("UNKNOWN")


def test_unknown_research_session(loaded_engine):

    with pytest.raises(ValueError):
        loaded_engine.get_session_bars(date(2020, 1, 2))


def test_warmup_session_not_executable(loaded_engine):

    with pytest.raises(ValueError):
        loaded_engine.get_session_bars(loaded_engine.warmup_sessions[-1])


def test_history_defensive_copy(loaded_engine):

    frame = loaded_engine.get_history("AAPL")

    original = frame.iloc[0]["close"]

    frame.iloc[0, frame.columns.get_loc("close")] = 9999

    assert loaded_engine.get_history("AAPL").iloc[0]["close"] == original


# ============================================================
# MARKET BARS
# ============================================================


def test_market_bar_type(loaded_engine):

    session = loaded_engine.sessions[0]

    bar = loaded_engine.get_market_bar("AAPL", session)

    assert isinstance(bar, PositionMarketBar)

    assert bar.symbol == "AAPL"

    assert bar.session == session


def test_market_bar_prices(loaded_engine):

    session = loaded_engine.sessions[0]

    row = loaded_engine.get_history("AAPL").loc[session]

    bar = loaded_engine.get_market_bar("AAPL", session)

    assert bar.opening == D(str(row["open"]))
    assert bar.high == D(str(row["high"]))
    assert bar.low == D(str(row["low"]))
    assert bar.close == D(str(row["close"]))


def test_session_bars_include_benchmarks(loaded_engine):

    bars = loaded_engine.get_session_bars(loaded_engine.sessions[0])

    assert set(bars) == {"AAPL", "NVDA", "SPY"}


# ============================================================
# SIGNAL TIMING
# ============================================================


def test_first_signal_uses_warmup(loaded_engine):

    first = loaded_engine.sessions[0]

    previous = loaded_engine.get_previous_session(first)

    assert previous == loaded_engine.warmup_sessions[-1]


def test_subsequent_signal_uses_previous_session(loaded_engine):

    sessions = loaded_engine.sessions

    for index in range(1, len(sessions)):
        assert (
            loaded_engine.get_previous_session(sessions[index]) == sessions[index - 1]
        )


def test_signal_date_precedes_execution(loaded_engine):

    for session in loaded_engine.sessions:
        signals = loaded_engine.get_session_signals(session)

        for row in signals.values():
            assert row["signal_date"] < session


def test_signal_row_matches_previous_history(loaded_engine):

    session = loaded_engine.sessions[0]

    previous = loaded_engine.get_previous_session(session)

    row = loaded_engine.get_signal_row("AAPL", session)

    history = loaded_engine.get_history("AAPL")

    assert row["close"] == history.loc[previous, "close"]

    assert row["signal_date"] == previous

    assert row["symbol"] == "AAPL"


def test_signals_exclude_benchmark_only_symbols(loaded_engine):

    signals = loaded_engine.get_session_signals(loaded_engine.sessions[0])

    assert set(signals) == {"AAPL", "NVDA"}


def test_signal_defensive_copy(loaded_engine):

    session = loaded_engine.sessions[0]

    row = loaded_engine.get_signal_row("AAPL", session)

    original = row["close"]

    row["close"] = 9999

    assert loaded_engine.get_signal_row("AAPL", session)["close"] == original


# ============================================================
# TECHNICAL INDICATORS
# ============================================================


@pytest.mark.parametrize("symbol", ["AAPL", "NVDA", "SPY"])
def test_required_indicators(loaded_engine, symbol):

    frame = loaded_engine.get_history(symbol)

    assert set(loaded_engine.INDICATOR_COLUMNS).issubset(frame.columns)


@pytest.mark.parametrize("symbol", ["AAPL", "NVDA", "SPY"])
def test_research_indicators_finite(loaded_engine, symbol):

    frame = loaded_engine.get_history(symbol)

    required = frame.loc[list(loaded_engine.sessions)]

    columns = [
        column for column in loaded_engine.INDICATOR_COLUMNS if column != "TREND"
    ]

    assert np.isfinite(required[columns].to_numpy(dtype=float)).all()


@pytest.mark.parametrize("symbol", ["AAPL", "NVDA", "SPY"])
def test_valid_trend_classifications(loaded_engine, symbol):

    frame = loaded_engine.get_history(symbol)

    required = frame.loc[list(loaded_engine.sessions)]

    assert required["TREND"].isin(["BULLISH", "BEARISH", "NEUTRAL"]).all()


# ============================================================
# LOOKAHEAD PROTECTION
# ============================================================


@pytest.mark.parametrize("symbol", ["AAPL", "NVDA", "SPY"])
@pytest.mark.parametrize("location", ["first", "middle", "last"])
def test_indicator_prefix_consistency(loaded_engine, symbol, location):

    sessions = loaded_engine.sessions

    selected = {
        "first": sessions[0],
        "middle": sessions[len(sessions) // 2],
        "last": sessions[-1],
    }[location]

    full = loaded_engine.get_history(symbol)

    truncated = full.loc[
        full.index <= selected, list(loaded_engine.REQUIRED_COLUMNS)
    ].copy()

    recalculated = TechnicalIndicators.calculate(truncated)

    columns = [
        column for column in loaded_engine.INDICATOR_COLUMNS if column != "TREND"
    ]

    np.testing.assert_allclose(
        full.loc[selected, columns].to_numpy(dtype=float),
        recalculated.loc[selected, columns].to_numpy(dtype=float),
        rtol=1e-10,
        atol=1e-10,
    )

    assert full.loc[selected, "TREND"] == recalculated.loc[selected, "TREND"]


def test_future_price_changes_do_not_change_signal(config, loaded_engine, raw_history):

    session = loaded_engine.sessions[0]

    original = loaded_engine.get_signal_row("AAPL", session)

    raw = raw_history["AAPL"].copy()

    timestamps = raw.index.get_level_values("timestamp")

    dates = timestamps.tz_convert("America/New_York").date

    future_mask = dates >= session

    for column in ["open", "high", "low", "close"]:
        raw.loc[future_mask, column] *= 2

    raw_history["AAPL"] = raw

    changed_engine = HistoricalDataEngine(config=config, market=OfflineMarket())

    changed_engine.load(raw_history)

    changed = changed_engine.get_signal_row("AAPL", session)

    pd.testing.assert_series_equal(original, changed)


# ============================================================
# SUMMARY
# ============================================================


def test_summary(loaded_engine, capsys):

    loaded_engine.display_summary()

    output = capsys.readouterr().out

    assert "HISTORICAL DATA ALIGNMENT ENGINE" in output

    assert "Missing required sessions: NONE" in output

    assert "Duplicate sessions: NONE" in output

    assert "PREVIOUS SESSION" in output

    assert "Broker orders: DISABLED" in output

    assert "Live trading: DISABLED" in output
