from collections.abc import Mapping
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from market.indicators import TechnicalIndicators
from market.market_data import MarketDataEngine
from portfolio.models import validate_session
from portfolio.portfolio_config import PortfolioConfig
from portfolio.position_manager import PositionMarketBar


class HistoricalDataEngine:
    REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")

    INDICATOR_COLUMNS = (
        "SMA_20",
        "SMA_50",
        "SMA_200",
        "EMA_9",
        "EMA_21",
        "RSI_14",
        "MACD",
        "MACD_SIGNAL",
        "ATR_14",
        "RELATIVE_VOLUME",
        "ROC_10",
        "SUPPORT",
        "RESISTANCE",
        "TREND",
    )

    TIMEZONE = "America/New_York"

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(self, config=None, market=None):

        if config is None:
            config = PortfolioConfig()

        if not isinstance(config, PortfolioConfig):
            raise ValueError("Invalid portfolio configuration.")

        config.validate()

        self._config = config

        settings = config.config

        historical = settings["historical_data"]

        self.historical_days = historical["historical_days"]

        self.warmup_candles = historical["warmup_candles"]

        self.start_date = date.fromisoformat(historical["start_date"])

        self.end_date = date.fromisoformat(historical["end_date"])

        self.symbols = tuple(settings["universe"]["symbols"])

        self.benchmark_symbols = (
            tuple(settings["benchmarks"]["symbols"])
            if settings["benchmarks"]["enabled"]
            else ()
        )

        self.required_symbols = tuple(
            dict.fromkeys(self.symbols + self.benchmark_symbols)
        )

        self.calendar = mcal.get_calendar("NYSE")

        self.market = market if market is not None else MarketDataEngine()

        self._history = {}

        self._sessions = ()

        self._warmup_sessions = ()

        self._loaded = False

    # ========================================================
    # PUBLIC PROPERTIES
    # ========================================================

    @property
    def loaded(self):

        return self._loaded

    @property
    def sessions(self):

        return self._sessions

    @property
    def warmup_sessions(self):

        return self._warmup_sessions

    @property
    def total_sessions(self):

        return len(self._sessions)

    @property
    def loaded_symbols(self):

        return tuple(self._history.keys())

    # ========================================================
    # EXCHANGE CALENDAR
    # ========================================================

    def get_calendar_sessions(self, start, end):

        validate_session(start, "start")
        validate_session(end, "end")

        if start > end:
            raise ValueError("Invalid calendar date range.")

        schedule = self.calendar.schedule(start_date=start, end_date=end)

        return tuple(timestamp.date() for timestamp in schedule.index)

    def build_expected_sessions(self):

        research_sessions = self.get_calendar_sessions(self.start_date, self.end_date)

        if not research_sessions:
            raise ValueError("No research trading sessions.")

        lookback_days = max(365, self.warmup_candles * 3)

        warmup_start = self.start_date - timedelta(days=lookback_days)

        warmup_end = self.start_date - timedelta(days=1)

        prior_sessions = self.get_calendar_sessions(warmup_start, warmup_end)

        if len(prior_sessions) < self.warmup_candles:
            raise ValueError("Insufficient exchange calendar warmup sessions.")

        warmup_sessions = prior_sessions[-self.warmup_candles :]

        return warmup_sessions, research_sessions

    # ========================================================
    # MARKET DATA NORMALIZATION
    # ========================================================

    def normalize_history(self, symbol, data):

        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"{symbol}: historical data must be a DataFrame.")

        if data.empty:
            raise ValueError(f"{symbol}: historical data is empty.")

        frame = data.copy(deep=True)

        if isinstance(frame.index, pd.MultiIndex):
            if "symbol" not in frame.index.names:
                raise ValueError(f"{symbol}: missing symbol index.")

            available_symbols = set(frame.index.get_level_values("symbol"))

            if available_symbols != {symbol}:
                raise ValueError(f"{symbol}: unexpected historical symbols.")

            frame = frame.droplevel("symbol")

        if not isinstance(frame.index, pd.DatetimeIndex):
            raise ValueError(f"{symbol}: invalid timestamp index.")

        if frame.index.tz is None:
            raise ValueError(f"{symbol}: timestamps must be timezone-aware.")

        if frame.index.hasnans:
            raise ValueError(f"{symbol}: missing timestamps.")

        if frame.index.has_duplicates:
            raise ValueError(f"{symbol}: duplicate timestamps.")

        missing_columns = set(self.REQUIRED_COLUMNS) - set(frame.columns)

        if missing_columns:
            raise ValueError(f"{symbol}: missing columns {sorted(missing_columns)}")

        frame = frame.sort_index()

        frame.index = pd.Index(
            frame.index.tz_convert(self.TIMEZONE).date, name="session"
        )

        if frame.index.has_duplicates:
            raise ValueError(f"{symbol}: duplicate trading sessions.")

        for column in self.REQUIRED_COLUMNS:
            if (
                frame[column]
                .map(lambda value: isinstance(value, (bool, np.bool_)))
                .any()
            ):
                raise ValueError(f"{symbol}: boolean values in {column}.")

            frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)

        values = frame[list(self.REQUIRED_COLUMNS)].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(f"{symbol}: non-finite market data.")

        price_columns = ["open", "high", "low", "close"]

        if (frame[price_columns] <= 0).any().any():
            raise ValueError(f"{symbol}: nonpositive market prices.")

        if (frame["volume"] < 0).any():
            raise ValueError(f"{symbol}: negative volume.")

        if (frame["high"] < frame[["open", "close"]].max(axis=1)).any():
            raise ValueError(f"{symbol}: invalid market high.")

        if (frame["low"] > frame[["open", "close"]].min(axis=1)).any():
            raise ValueError(f"{symbol}: invalid market low.")

        if (frame["low"] > frame["high"]).any():
            raise ValueError(f"{symbol}: market low exceeds high.")

        return frame

    # ========================================================
    # SESSION ALIGNMENT
    # ========================================================

    def align_history(self, symbol, frame, expected_sessions):

        if not expected_sessions:
            raise ValueError("Expected sessions cannot be empty.")

        expected = tuple(expected_sessions)

        if len(expected) != len(set(expected)):
            raise ValueError("Duplicate expected sessions.")

        if expected != tuple(sorted(expected)):
            raise ValueError("Expected sessions must be sorted.")

        start = expected[0]

        end = expected[-1]

        frame = frame.loc[(frame.index >= start) & (frame.index <= end)].copy()

        expected_set = set(expected)

        actual_set = set(frame.index)

        missing = sorted(expected_set - actual_set)

        unexpected = sorted(actual_set - expected_set)

        if missing:
            raise ValueError(
                f"{symbol}: missing {len(missing)} "
                f"trading sessions. Examples: {missing[:5]}"
            )

        if unexpected:
            raise ValueError(
                f"{symbol}: unexpected trading sessions. Examples: {unexpected[:5]}"
            )

        frame = frame.loc[list(expected)].copy()

        if tuple(frame.index) != expected:
            raise ValueError(f"{symbol}: session alignment failed.")

        return frame

    # ========================================================
    # TECHNICAL INDICATORS
    # ========================================================

    def prepare_indicators(self, symbol, frame, research_sessions):

        calculated = TechnicalIndicators.calculate(frame.copy(deep=True))

        if tuple(calculated.index) != tuple(frame.index):
            raise ValueError(f"{symbol}: indicators changed session alignment.")

        missing_columns = set(self.INDICATOR_COLUMNS) - set(calculated.columns)

        if missing_columns:
            raise ValueError(f"{symbol}: missing indicators {sorted(missing_columns)}")

        # Include the last warmup session because it
        # supplies the first research session's signal.

        first_research = research_sessions[0]

        first_index = calculated.index.get_loc(first_research)

        if first_index < 1:
            raise ValueError(f"{symbol}: missing previous signal session.")

        required = calculated.iloc[first_index - 1 :]

        numeric_columns = [
            column for column in self.INDICATOR_COLUMNS if column != "TREND"
        ]

        values = required[numeric_columns].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(f"{symbol}: invalid research indicators.")

        if not required["TREND"].isin(["BULLISH", "BEARISH", "NEUTRAL"]).all():
            raise ValueError(f"{symbol}: invalid trend classification.")

        return calculated

    # ========================================================
    # LOAD HISTORICAL DATA
    # ========================================================

    def load(self, historical_data=None):

        if historical_data is not None:
            if not isinstance(historical_data, Mapping):
                raise ValueError("Historical data must be a mapping.")

        warmup, research = self.build_expected_sessions()

        expected = warmup + research

        prepared = {}

        for symbol in self.required_symbols:
            if historical_data is None:
                raw = self.market.get_historical_data(symbol, days=self.historical_days)

            else:
                if symbol not in historical_data:
                    raise ValueError(f"Missing historical data: {symbol}")

                raw = historical_data[symbol]

            normalized = self.normalize_history(symbol, raw)

            aligned = self.align_history(symbol, normalized, expected)

            calculated = self.prepare_indicators(symbol, aligned, research)

            prepared[symbol] = calculated

        # Publish only after every required symbol
        # has passed validation.

        self._history = prepared

        self._warmup_sessions = warmup

        self._sessions = research

        self._loaded = True

        return self.total_sessions

    # ========================================================
    # ACCESS VALIDATION
    # ========================================================

    def require_loaded(self):

        if not self._loaded:
            raise RuntimeError("Historical data has not been loaded.")

    def require_symbol(self, symbol):

        self.require_loaded()

        if symbol not in self._history:
            raise ValueError(f"Unknown historical symbol: {symbol}")

    def require_research_session(self, session):

        self.require_loaded()

        validate_session(session, "session")

        if session not in self._sessions:
            raise ValueError(f"Unknown research session: {session}")

    # ========================================================
    # HISTORICAL MARKET BARS
    # ========================================================

    def get_market_bar(self, symbol, session):

        self.require_symbol(symbol)

        self.require_research_session(session)

        row = self._history[symbol].loc[session]

        return PositionMarketBar(
            session=session,
            symbol=symbol,
            opening=str(row["open"]),
            high=str(row["high"]),
            low=str(row["low"]),
            close=str(row["close"]),
        )

    def get_session_bars(self, session):

        self.require_research_session(session)

        return {
            symbol: self.get_market_bar(symbol, session)
            for symbol in self.required_symbols
        }

    # ========================================================
    # PREVIOUS-SESSION SIGNALS
    # ========================================================

    def get_previous_session(self, session):

        self.require_research_session(session)

        index = self._sessions.index(session)

        if index == 0:
            return self._warmup_sessions[-1]

        return self._sessions[index - 1]

    def get_signal_row(self, symbol, session):

        self.require_symbol(symbol)

        previous = self.get_previous_session(session)

        row = self._history[symbol].loc[previous].copy(deep=True)

        row["symbol"] = symbol

        row["signal_date"] = previous

        return row

    def get_session_signals(self, session):

        self.require_research_session(session)

        return {symbol: self.get_signal_row(symbol, session) for symbol in self.symbols}

    # ========================================================
    # HISTORICAL RESEARCH ACCESS
    # ========================================================

    def get_history(self, symbol):

        self.require_symbol(symbol)

        return self._history[symbol].copy(deep=True)

    # ========================================================
    # SUMMARY
    # ========================================================

    def display_summary(self):

        self.require_loaded()

        print("=" * 70)
        print("JARVIS HISTORICAL DATA ALIGNMENT ENGINE")
        print("VERSION 0.8.6.6")
        print("=" * 70)

        print("Exchange calendar: NYSE")

        print("Timezone:", self.TIMEZONE)

        print("Trading symbols:", len(self.symbols))

        print("Loaded symbols:", len(self.loaded_symbols))

        print("Warmup sessions:", len(self.warmup_sessions))

        print("Research sessions:", self.total_sessions)

        print("First warmup session:", self.warmup_sessions[0])

        print("First research session:", self.sessions[0])

        print("Last research session:", self.sessions[-1])

        print()

        for symbol in self.loaded_symbols:
            frame = self._history[symbol]

            print(f"{symbol:6} | {len(frame):4} candles | ALIGNED")

        print()

        print("Missing required sessions: NONE")
        print("Duplicate sessions: NONE")
        print("Indicator validation: PASSED")
        print("Signal timing: PREVIOUS SESSION")
        print("Historical simulation only")
        print("Broker orders: DISABLED")
        print("Live trading: DISABLED")


if __name__ == "__main__":
    engine = HistoricalDataEngine()

    print("Loading historical portfolio data...")
    print()

    engine.load()

    engine.display_summary()
