from dataclasses import replace
from datetime import date, datetime
from decimal import Decimal as D

import numpy as np
import pandas as pd
import pytest

from portfolio.capital_manager import CapitalManager
from portfolio.historical_data import HistoricalDataEngine
from portfolio.models import (
    PortfolioExecutionResult,
    PortfolioOpportunity,
    PortfolioRejection,
)
from portfolio.opportunity_competition import (
    CompetitionResult,
    OpportunityBatch,
    OpportunityCompetitionEngine,
)
from portfolio.portfolio_config import PortfolioConfig

SESSION = date(2026, 11, 23)
SIGNAL_DATE = date(2026, 11, 20)


# ============================================================
# OFFLINE TEST PROVIDERS
# ============================================================


class FakeRanker:
    def __init__(self, score=85):

        self.score = score
        self.calls = []

    def calculate_score(self, row):

        self.calls.append(row.copy())

        return self.score, {}, 2.0


class FakeStrategy:
    def __init__(self):

        self.config = {
            "settings": {},
            "momentum": {"enabled": True},
            "breakout": {"enabled": True},
            "pullback": {"enabled": True},
        }

        self.action = "BUY"
        self.strategy = "MOMENTUM"
        self.reason = "Synthetic strategy."
        self.calls = []
        self.response = None

    @property
    def settings(self):

        return self.config["settings"]

    def generate_signal(self, row):

        self.calls.append(row.copy())

        if self.response is not None:
            return self.response

        return {
            "symbol": row["symbol"],
            "signal": self.action,
            "strategy": self.strategy,
            "reason": self.reason,
        }


class OfflineMarket:
    def get_historical_data(self, *args, **kwargs):

        raise AssertionError("Automated tests must not call Alpaca.")


# ============================================================
# FIXTURES
# ============================================================


@pytest.fixture
def config():

    return PortfolioConfig()


@pytest.fixture
def ranker():

    return FakeRanker()


@pytest.fixture
def strategy():

    return FakeStrategy()


@pytest.fixture
def engine(config, ranker, strategy):

    return OpportunityCompetitionEngine(
        config=config, ranker=ranker, strategy_engine=strategy
    )


@pytest.fixture
def capital(config):

    manager = CapitalManager(config=config)

    manager.start_session(SESSION)

    return manager


def make_signal(symbol="AAPL", signal_date=SIGNAL_DATE):

    return pd.Series(
        {
            "symbol": symbol,
            "signal_date": signal_date,
            "close": 100.0,
            "SMA_20": 98.0,
            "SMA_50": 95.0,
            "SMA_200": 90.0,
            "EMA_9": 99.0,
            "EMA_21": 97.0,
            "RSI_14": 60.0,
            "MACD": 1.5,
            "MACD_SIGNAL": 1.0,
            "ATR_14": 2.0,
            "RELATIVE_VOLUME": 1.5,
            "ROC_10": 3.0,
            "SUPPORT": 90.0,
            "RESISTANCE": 105.0,
            "TREND": "BULLISH",
        }
    )


def make_opportunity(
    symbol="AAPL", score=85, strategy="MOMENTUM", signal_date=SIGNAL_DATE
):

    return PortfolioOpportunity(
        symbol=symbol,
        strategy=strategy,
        signal_date=signal_date,
        score=score,
        reference_price="100",
        atr="2",
        stop_atr_multiplier="1.5",
        risk_reward_ratio="2",
    )


def make_batch(engine, candidates=()):

    candidates = tuple(candidates)

    symbols = {item.symbol for item in candidates}

    rejections = tuple(
        PortfolioRejection(
            session=SESSION,
            symbol=symbol,
            strategy="NONE",
            score=0,
            reason="Synthetic exclusion.",
        )
        for symbol in engine.symbols
        if symbol not in symbols
    )

    return OpportunityBatch(
        session=SESSION,
        opportunities=engine.rank_opportunities(candidates),
        rejections=rejections,
        evaluated_symbols=len(engine.symbols),
    )


def make_prices(batch):

    return {item.symbol: D(100) for item in batch.opportunities}


# ============================================================
# INITIALIZATION
# ============================================================


def test_real_engine_initialization():

    engine = OpportunityCompetitionEngine()

    assert len(engine.symbols) == 12

    assert engine.stop_atr_multiplier == D("1.5")

    assert engine.risk_reward_ratio == D(2)


def test_initialization(engine):

    assert len(engine.symbols) == 12

    assert engine.minimum_score == D(55)

    assert engine.maximum_new_positions == 3


def test_enabled_strategies(engine):

    assert engine.enabled_strategies == {"MOMENTUM", "BREAKOUT", "PULLBACK"}


@pytest.mark.parametrize("invalid", [{}, [], True, 123, "invalid"])
def test_invalid_configuration(invalid):

    with pytest.raises(ValueError):
        OpportunityCompetitionEngine(config=invalid)


def test_strategy_settings_synchronized(engine):

    settings = engine.strategy_engine.settings

    assert settings["minimum_score"] == 55

    assert settings["stop_atr_multiplier"] == 1.5

    assert settings["minimum_risk_reward"] == 2


@pytest.mark.parametrize("name", ["momentum", "breakout", "pullback"])
def test_disabled_strategy_synchronized(config, ranker, strategy, name):

    settings = config.config

    settings["strategy"][f"{name}_enabled"] = False

    engine = OpportunityCompetitionEngine(
        config=PortfolioConfig(config=settings), ranker=ranker, strategy_engine=strategy
    )

    assert name.upper() not in engine.enabled_strategies

    assert strategy.config[name]["enabled"] is False


def test_custom_strategy_parameters(config, ranker, strategy):

    settings = config.config

    settings["strategy"]["stop_atr_multiplier"] = 2.0

    settings["strategy"]["risk_reward_ratio"] = 3.0

    settings["opportunity_selection"]["minimum_score"] = 75

    engine = OpportunityCompetitionEngine(
        config=PortfolioConfig(config=settings), ranker=ranker, strategy_engine=strategy
    )

    assert engine.minimum_score == D(75)

    assert engine.stop_atr_multiplier == D(2)

    assert engine.risk_reward_ratio == D(3)

    assert strategy.settings["minimum_score"] == 75

    assert strategy.settings["stop_atr_multiplier"] == 2

    assert strategy.settings["minimum_risk_reward"] == 3


# ============================================================
# SIGNAL NORMALIZATION
# ============================================================


def test_normalize_signal(engine):

    row = engine.normalize_signal("AAPL", make_signal(), SESSION)

    assert row["symbol"] == "AAPL"

    assert row["signal_date"] == SIGNAL_DATE

    assert row["sma_20"] == 98

    assert row["rsi"] == 60

    assert row["atr"] == 2

    assert row["trend"] == "BULLISH"


def test_all_required_fields_present(engine):

    row = engine.normalize_signal("AAPL", make_signal(), SESSION)

    assert set(engine.REQUIRED_FIELDS).issubset(row.index)


def test_original_signal_unchanged(engine):

    row = make_signal()

    original = row.copy(deep=True)

    engine.normalize_signal("AAPL", row, SESSION)

    pd.testing.assert_series_equal(row, original)


def test_already_normalized_signal(engine):

    row = make_signal().rename(index=engine.COLUMN_MAPPING)

    result = engine.normalize_signal("AAPL", row, SESSION)

    assert result["sma_20"] == 98


@pytest.mark.parametrize("invalid", [None, {}, [], 123, True, "invalid"])
def test_invalid_signal_type(engine, invalid):

    with pytest.raises(ValueError):
        engine.normalize_signal("AAPL", invalid, SESSION)


def test_unknown_symbol(engine):

    with pytest.raises(ValueError):
        engine.normalize_signal("UNKNOWN", make_signal("UNKNOWN"), SESSION)


def test_symbol_mismatch(engine):

    with pytest.raises(ValueError):
        engine.normalize_signal("AAPL", make_signal("NVDA"), SESSION)


@pytest.mark.parametrize(
    "invalid", [None, "2026-11-23", True, 123, datetime(2026, 11, 23)]
)
def test_invalid_execution_session(engine, invalid):

    with pytest.raises(ValueError):
        engine.normalize_signal("AAPL", make_signal(), invalid)


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        "2026-11-20",
        True,
        123,
        datetime(2026, 11, 20),
        SESSION,
        date(2026, 11, 24),
    ],
)
def test_invalid_signal_date(engine, invalid):

    with pytest.raises(ValueError):
        engine.normalize_signal("AAPL", make_signal(signal_date=invalid), SESSION)


@pytest.mark.parametrize(
    "field",
    [
        "symbol",
        "signal_date",
        "close",
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
    ],
)
def test_missing_signal_field(engine, field):

    row = make_signal().drop(field)

    with pytest.raises(ValueError):
        engine.normalize_signal("AAPL", row, SESSION)


def test_duplicate_signal_fields(engine):

    row = make_signal()

    row = pd.concat([row, pd.Series({"close": 101})])

    with pytest.raises(ValueError):
        engine.normalize_signal("AAPL", row, SESSION)


def test_conflicting_normalized_fields(engine):

    row = make_signal()

    row["sma_20"] = 99

    with pytest.raises(ValueError):
        engine.normalize_signal("AAPL", row, SESSION)


@pytest.mark.parametrize(
    "field",
    [
        "close",
        "SMA_20",
        "SMA_50",
        "SMA_200",
        "EMA_9",
        "EMA_21",
        "ATR_14",
        "SUPPORT",
        "RESISTANCE",
    ],
)
@pytest.mark.parametrize("invalid", [0, -1, np.nan, np.inf, True, "invalid"])
def test_invalid_positive_fields(engine, field, invalid):

    row = make_signal()

    row[field] = invalid

    with pytest.raises(ValueError):
        engine.normalize_signal("AAPL", row, SESSION)


@pytest.mark.parametrize("invalid", [-1, 101, np.nan, np.inf, True])
def test_invalid_rsi(engine, invalid):

    row = make_signal()

    row["RSI_14"] = invalid

    with pytest.raises(ValueError):
        engine.normalize_signal("AAPL", row, SESSION)


@pytest.mark.parametrize("value", [0, 30, 50, 70, 100])
def test_valid_rsi_boundaries(engine, value):

    row = make_signal()

    row["RSI_14"] = value

    normalized = engine.normalize_signal("AAPL", row, SESSION)

    assert normalized["rsi"] == value


@pytest.mark.parametrize("invalid", [-1, np.nan, np.inf, True])
def test_invalid_relative_volume(engine, invalid):

    row = make_signal()

    row["RELATIVE_VOLUME"] = invalid

    with pytest.raises(ValueError):
        engine.normalize_signal("AAPL", row, SESSION)


@pytest.mark.parametrize("trend", ["BULLISH", "BEARISH", "NEUTRAL"])
def test_valid_trends(engine, trend):

    row = make_signal()

    row["TREND"] = trend

    normalized = engine.normalize_signal("AAPL", row, SESSION)

    assert normalized["trend"] == trend


@pytest.mark.parametrize("trend", ["UNKNOWN", "bullish", "", None, 123])
def test_invalid_trend(engine, trend):

    row = make_signal()

    row["TREND"] = trend

    with pytest.raises(ValueError):
        engine.normalize_signal("AAPL", row, SESSION)


# ============================================================
# OPPORTUNITY EVALUATION
# ============================================================


def test_qualified_opportunity(engine):

    result = engine.evaluate_signal("AAPL", make_signal(), SESSION)

    assert isinstance(result, PortfolioOpportunity)

    assert result.symbol == "AAPL"

    assert result.strategy == "MOMENTUM"

    assert result.score == D(85)

    assert result.signal_date == SIGNAL_DATE

    assert result.reference_price == D(100)

    assert result.atr == D(2)


@pytest.mark.parametrize("score", [0, 10, 40, 54.99])
def test_low_score_rejected(engine, ranker, strategy, score):

    ranker.score = score

    result = engine.evaluate_signal("AAPL", make_signal(), SESSION)

    assert isinstance(result, PortfolioRejection)

    assert "Minimum opportunity score" in result.reason

    assert strategy.calls == []


@pytest.mark.parametrize("score", [55, 70, 85, 100])
def test_qualifying_score_boundary(engine, ranker, score):

    ranker.score = score

    result = engine.evaluate_signal("AAPL", make_signal(), SESSION)

    assert isinstance(result, PortfolioOpportunity)


@pytest.mark.parametrize("score", [-1, 101, np.nan, np.inf, True])
def test_invalid_ranker_score(engine, ranker, score):

    ranker.score = score

    with pytest.raises(ValueError):
        engine.evaluate_signal("AAPL", make_signal(), SESSION)


def test_no_trade_rejection(engine, strategy):

    strategy.action = "NO_TRADE"

    strategy.strategy = "NONE"

    strategy.reason = "No strategy conditions satisfied."

    result = engine.evaluate_signal("AAPL", make_signal(), SESSION)

    assert isinstance(result, PortfolioRejection)

    assert result.reason == strategy.reason


@pytest.mark.parametrize("name", ["MOMENTUM", "BREAKOUT", "PULLBACK"])
def test_supported_strategies(engine, strategy, name):

    strategy.strategy = name

    result = engine.evaluate_signal("AAPL", make_signal(), SESSION)

    assert result.strategy == name


@pytest.mark.parametrize("action", ["SELL", "SHORT", "HOLD", "", None])
def test_unsupported_action(engine, strategy, action):

    strategy.action = action

    with pytest.raises(ValueError):
        engine.evaluate_signal("AAPL", make_signal(), SESSION)


def test_unknown_strategy(engine, strategy):

    strategy.strategy = "UNKNOWN"

    with pytest.raises(ValueError):
        engine.evaluate_signal("AAPL", make_signal(), SESSION)


def test_invalid_strategy_response(engine, strategy):

    strategy.response = ["BUY"]

    with pytest.raises(ValueError):
        engine.evaluate_signal("AAPL", make_signal(), SESSION)


def test_strategy_symbol_mismatch(engine, strategy):

    strategy.response = {"symbol": "NVDA", "signal": "BUY", "strategy": "MOMENTUM"}

    with pytest.raises(ValueError):
        engine.evaluate_signal("AAPL", make_signal(), SESSION)


def test_disabled_strategy_rejected(config, ranker, strategy):

    settings = config.config

    settings["strategy"]["momentum_enabled"] = False

    engine = OpportunityCompetitionEngine(
        config=PortfolioConfig(config=settings), ranker=ranker, strategy_engine=strategy
    )

    result = engine.evaluate_signal("AAPL", make_signal(), SESSION)

    assert isinstance(result, PortfolioRejection)

    assert result.reason == "Strategy is disabled."


def test_score_passed_to_strategy(engine, strategy):

    engine.evaluate_signal("AAPL", make_signal(), SESSION)

    assert strategy.calls[0]["score"] == 85


# ============================================================
# DETERMINISTIC RANKING
# ============================================================


def test_empty_ranking(engine):

    assert engine.rank_opportunities([]) == ()


def test_single_opportunity(engine):

    item = make_opportunity()

    assert engine.rank_opportunities([item]) == (item,)


def test_descending_score_ranking(engine):

    items = [
        make_opportunity("AAPL", 70),
        make_opportunity("NVDA", 95),
        make_opportunity("MSFT", 85),
    ]

    ranked = engine.rank_opportunities(items)

    assert [item.symbol for item in ranked] == ["NVDA", "MSFT", "AAPL"]


def test_alphabetical_tie_breaker(engine):

    items = [
        make_opportunity("TSLA", 85),
        make_opportunity("NVDA", 85),
        make_opportunity("AAPL", 85),
    ]

    ranked = engine.rank_opportunities(items)

    assert [item.symbol for item in ranked] == ["AAPL", "NVDA", "TSLA"]


def test_ranking_independent_of_input_order(engine):

    items = [
        make_opportunity("AAPL", 85),
        make_opportunity("NVDA", 95),
        make_opportunity("MSFT", 85),
    ]

    assert engine.rank_opportunities(items) == engine.rank_opportunities(items[::-1])


def test_duplicate_opportunities_rejected(engine):

    items = [make_opportunity("AAPL", 85), make_opportunity("AAPL", 90)]

    with pytest.raises(ValueError):
        engine.rank_opportunities(items)


@pytest.mark.parametrize("invalid", [None, {}, True, 123, "invalid"])
def test_invalid_opportunity_item(engine, invalid):

    with pytest.raises(ValueError):
        engine.rank_opportunities([invalid])


# ============================================================
# HISTORICAL BATCH CONSTRUCTION
# ============================================================


@pytest.fixture
def historical(config, monkeypatch):

    historical = HistoricalDataEngine(config=config, market=OfflineMarket())

    monkeypatch.setattr(historical, "_loaded", True)

    monkeypatch.setattr(historical, "_sessions", (SESSION,))

    monkeypatch.setattr(historical, "_warmup_sessions", (SIGNAL_DATE,))

    monkeypatch.setattr(
        historical,
        "get_session_signals",
        lambda session: {symbol: make_signal(symbol) for symbol in historical.symbols},
    )

    return historical


def test_build_complete_batch(engine, historical):

    batch = engine.build_batch(historical, SESSION)

    assert isinstance(batch, OpportunityBatch)

    assert batch.session == SESSION

    assert batch.evaluated_symbols == 12

    assert batch.qualified_count == 12

    assert batch.rejected_count == 0


def test_batch_all_rejected(engine, historical, ranker):

    ranker.score = 10

    batch = engine.build_batch(historical, SESSION)

    assert batch.qualified_count == 0

    assert batch.rejected_count == 12


def test_batch_ranking(engine, historical):

    batch = engine.build_batch(historical, SESSION)

    assert [item.symbol for item in batch.opportunities] == sorted(engine.symbols)


def test_batch_signal_timing(engine, historical):

    batch = engine.build_batch(historical, SESSION)

    assert all(item.signal_date == SIGNAL_DATE for item in batch.opportunities)


def test_invalid_historical_engine(engine):

    with pytest.raises(ValueError):
        engine.build_batch({}, SESSION)


def test_unloaded_historical_engine(engine, config):

    historical = HistoricalDataEngine(config=config, market=OfflineMarket())

    with pytest.raises(RuntimeError):
        engine.build_batch(historical, SESSION)


def test_historical_universe_mismatch(engine, historical, monkeypatch):

    monkeypatch.setattr(historical, "symbols", ("AAPL",))

    with pytest.raises(ValueError):
        engine.build_batch(historical, SESSION)


def test_incomplete_signal_universe(engine, historical, monkeypatch):

    monkeypatch.setattr(
        historical, "get_session_signals", lambda session: {"AAPL": make_signal()}
    )

    with pytest.raises(ValueError):
        engine.build_batch(historical, SESSION)


def test_stale_signal_rejected(engine, historical, monkeypatch):

    signals = {symbol: make_signal(symbol) for symbol in engine.symbols}

    signals["AAPL"]["signal_date"] = date(2026, 11, 19)

    monkeypatch.setattr(historical, "get_session_signals", lambda session: signals)

    with pytest.raises(ValueError):
        engine.build_batch(historical, SESSION)


# ============================================================
# SHARED CAPITAL EXECUTION
# ============================================================


def test_empty_batch_execution(engine, capital):

    batch = make_batch(engine)

    result = engine.execute_batch(batch, capital, {})

    assert isinstance(result, CompetitionResult)

    assert result.executed_count == 0

    assert result.rejected_count == 12

    assert capital.state.cash == D(100)


def test_single_candidate_execution(engine, capital):

    batch = make_batch(engine, [make_opportunity()])

    result = engine.execute_batch(batch, capital, make_prices(batch))

    assert result.executed_count == 1

    assert result.executed_symbols == ("AAPL",)

    assert result.rejected_count == 11

    assert capital.state.open_positions == 1


def test_highest_score_receives_priority(engine, capital):

    batch = make_batch(
        engine,
        [
            make_opportunity("AAPL", 85),
            make_opportunity("NVDA", 95),
            make_opportunity("MSFT", 90),
        ],
    )

    result = engine.execute_batch(batch, capital, make_prices(batch))

    assert result.executed_symbols[0] == "NVDA"


def test_shared_risk_budget(engine, capital):

    batch = make_batch(
        engine,
        [
            make_opportunity("NVDA", 95),
            make_opportunity("AAPL", 90),
            make_opportunity("MSFT", 85),
            make_opportunity("AMD", 80),
            make_opportunity("TSLA", 75),
        ],
    )

    result = engine.execute_batch(batch, capital, make_prices(batch))

    assert result.executed_symbols == ("NVDA", "AAPL")

    assert result.executed_count == 2

    assert result.rejected_count == 10

    assert capital.state.aggregate_open_risk <= capital.maximum_aggregate_risk


def test_cash_and_exposure_limits(engine, capital):

    batch = make_batch(
        engine, [make_opportunity(symbol, 90) for symbol in engine.symbols]
    )

    engine.execute_batch(batch, capital, make_prices(batch))

    assert capital.state.cash >= 0

    assert capital.state.open_positions <= 3

    assert capital.state.total_exposure <= capital.maximum_exposure


def test_execution_accounting(engine, capital):

    batch = make_batch(
        engine, [make_opportunity("AAPL", 90), make_opportunity("NVDA", 95)]
    )

    result = engine.execute_batch(batch, capital, make_prices(batch))

    assert result.executed_count + result.rejected_count == batch.evaluated_symbols

    assert (
        capital.state.equity == capital.state.initial_capital + capital.state.total_pnl
    )


def test_rejected_candidate_does_not_block_next(engine, capital, monkeypatch):

    batch = make_batch(
        engine, [make_opportunity("NVDA", 95), make_opportunity("AAPL", 90)]
    )

    original = capital.open_position

    def open_position(opportunity, price):

        if opportunity.symbol == "NVDA":
            return PortfolioExecutionResult(
                executed=False, symbol="NVDA", reason="Synthetic risk rejection."
            )

        return original(opportunity, price)

    monkeypatch.setattr(capital, "open_position", open_position)

    result = engine.execute_batch(batch, capital, make_prices(batch))

    assert result.executed_symbols == ("AAPL",)

    assert result.rejected_count == 11


def test_competition_entry_limit(config, ranker, strategy):

    settings = config.config

    settings["opportunity_selection"]["maximum_new_positions_per_session"] = 1

    custom = PortfolioConfig(config=settings)

    engine = OpportunityCompetitionEngine(
        config=custom, ranker=ranker, strategy_engine=strategy
    )

    capital = CapitalManager(config=custom)

    capital.start_session(SESSION)

    batch = make_batch(
        engine, [make_opportunity("NVDA", 95), make_opportunity("AAPL", 90)]
    )

    result = engine.execute_batch(batch, capital, make_prices(batch))

    assert result.executed_count == 1

    assert result.executed_symbols == ("NVDA",)

    assert any(
        item.reason == "Competition entry limit reached." for item in result.rejections
    )


# ============================================================
# EXECUTION INPUT PROTECTION
# ============================================================


def test_invalid_batch(engine, capital):

    with pytest.raises(ValueError):
        engine.execute_batch({}, capital, {})


def test_invalid_capital_manager(engine):

    with pytest.raises(ValueError):
        engine.execute_batch(make_batch(engine), {}, {})


def test_capital_session_mismatch(engine, config):

    capital = CapitalManager(config=config)

    capital.start_session(date(2026, 11, 24))

    with pytest.raises(ValueError):
        engine.execute_batch(make_batch(engine), capital, {})


def test_configuration_mismatch(engine, config):

    settings = config.config

    settings["opportunity_selection"]["minimum_score"] = 75

    capital = CapitalManager(config=PortfolioConfig(config=settings))

    capital.start_session(SESSION)

    with pytest.raises(ValueError):
        engine.execute_batch(make_batch(engine), capital, {})


@pytest.mark.parametrize("invalid", [None, [], True, 123, "invalid"])
def test_invalid_price_mapping(engine, capital, invalid):

    with pytest.raises(ValueError):
        engine.execute_batch(make_batch(engine), capital, invalid)


def test_missing_price_rejected_before_execution(engine, capital):

    batch = make_batch(
        engine, [make_opportunity("NVDA", 95), make_opportunity("AAPL", 90)]
    )

    original = capital.state

    with pytest.raises(ValueError):
        engine.execute_batch(batch, capital, {"NVDA": D(100)})

    assert capital.state == original


@pytest.mark.parametrize("invalid", [0, -1, None, True, np.nan, np.inf, "invalid"])
def test_invalid_opening_price(engine, capital, invalid):

    batch = make_batch(engine, [make_opportunity()])

    original = capital.state

    with pytest.raises(ValueError):
        engine.execute_batch(batch, capital, {"AAPL": invalid})

    assert capital.state == original


def test_later_invalid_price_prevents_partial_execution(engine, capital):

    batch = make_batch(
        engine, [make_opportunity("NVDA", 95), make_opportunity("AAPL", 90)]
    )

    original = capital.state

    with pytest.raises(ValueError):
        engine.execute_batch(batch, capital, {"NVDA": 100, "AAPL": -1})

    assert capital.state == original


def test_unsorted_batch_rejected(engine, capital):

    batch = make_batch(
        engine, [make_opportunity("NVDA", 95), make_opportunity("AAPL", 90)]
    )

    batch = replace(batch, opportunities=batch.opportunities[::-1])

    with pytest.raises(ValueError):
        engine.execute_batch(batch, capital, make_prices(batch))


def test_incomplete_evaluated_count(engine, capital):

    batch = replace(make_batch(engine), evaluated_symbols=11)

    with pytest.raises(ValueError):
        engine.execute_batch(batch, capital, {})


def test_missing_rejection_record(engine, capital):

    batch = make_batch(engine)

    batch = replace(batch, rejections=batch.rejections[:-1])

    with pytest.raises(ValueError):
        engine.execute_batch(batch, capital, {})


def test_duplicate_rejection_record(engine, capital):

    batch = make_batch(engine)

    batch = replace(batch, rejections=(batch.rejections[:-1] + (batch.rejections[0],)))

    with pytest.raises(ValueError):
        engine.execute_batch(batch, capital, {})


def test_invalid_rejection_record(engine, capital):

    batch = make_batch(engine)

    batch = replace(batch, rejections=("invalid",))

    with pytest.raises(ValueError):
        engine.execute_batch(batch, capital, {})


def test_rejection_session_mismatch(engine, capital):

    batch = make_batch(engine)

    damaged = replace(batch.rejections[0], session=date(2026, 11, 24))

    batch = replace(batch, rejections=((damaged,) + batch.rejections[1:]))

    with pytest.raises(ValueError):
        engine.execute_batch(batch, capital, {})


@pytest.mark.parametrize("signal_date", [SESSION, date(2026, 11, 24)])
def test_invalid_execution_signal_timing(engine, capital, signal_date):

    batch = make_batch(engine, [make_opportunity(signal_date=signal_date)])

    with pytest.raises(ValueError):
        engine.execute_batch(batch, capital, make_prices(batch))


# ============================================================
# REPORTING
# ============================================================


def test_batch_summary(engine, capsys):

    batch = make_batch(engine, [make_opportunity()])

    engine.display_batch(batch)

    output = capsys.readouterr().out

    assert "JARVIS OPPORTUNITY COMPETITION" in output

    assert "AAPL" in output

    assert "Broker orders: DISABLED" in output

    assert "Live trading: DISABLED" in output


def test_empty_batch_summary(engine, capsys):

    engine.display_batch(make_batch(engine))

    output = capsys.readouterr().out

    assert "No qualifying opportunities." in output


def test_execution_summary(engine, capital, capsys):

    batch = make_batch(engine, [make_opportunity()])

    result = engine.execute_batch(batch, capital, make_prices(batch))

    engine.display_result(result)

    output = capsys.readouterr().out

    assert "PORTFOLIO COMPETITION RESULTS" in output

    assert "EXECUTED" in output

    assert "Broker orders: DISABLED" in output

    assert "Live trading: DISABLED" in output
