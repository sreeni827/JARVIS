from dataclasses import FrozenInstanceError, replace
from datetime import date
from decimal import Decimal as D

import pytest

from portfolio.capital_manager import CapitalManager
from portfolio.models import PortfolioOpportunity
from portfolio.position_manager import (
    PositionExitDecision,
    PositionManager,
    PositionMarketBar,
)

from portfolio.decimal_utils import normalize_money

ZERO = D(0)

SIGNAL_DATE = date(2026, 11, 19)
ENTRY_DATE = date(2026, 11, 20)
NEXT_SESSION = date(2026, 11, 23)
FINAL_SESSION = date(2026, 11, 24)


# ============================================================
# FIXTURES AND HELPERS
# ============================================================


@pytest.fixture
def capital():

    return CapitalManager()


@pytest.fixture
def manager(capital):

    return PositionManager(capital)


def make_opportunity(capital, symbol="AAPL", **overrides):

    values = {
        "symbol": symbol,
        "strategy": "MOMENTUM",
        "signal_date": SIGNAL_DATE,
        "score": D(90),
        "reference_price": D(100),
        "atr": D(2),
        "stop_atr_multiplier": capital.stop_atr_multiplier,
        "risk_reward_ratio": capital.risk_reward_ratio,
    }

    values.update(overrides)

    return PortfolioOpportunity(**values)


def make_bar(
    session=NEXT_SESSION,
    symbol="AAPL",
    opening="100",
    high="102",
    low="99",
    close="101",
):

    return PositionMarketBar(
        session=session,
        symbol=symbol,
        opening=D(opening),
        high=D(high),
        low=D(low),
        close=D(close),
    )


def open_position(capital, symbol="AAPL"):

    result = capital.open_position(make_opportunity(capital, symbol), D(100))

    assert result.executed

    return result.position


@pytest.fixture
def overnight_position(manager, capital):

    capital.start_session(ENTRY_DATE)

    manager.process_opening_exits({})

    open_position(capital)

    bars = {"AAPL": make_bar(session=ENTRY_DATE)}

    manager.process_intraday_exits(bars)
    manager.process_closing_exits(bars)
    manager.mark_closing_positions(bars)

    capital.start_session(NEXT_SESSION)

    return capital.state.positions[0]


def assert_accounting(capital):

    state = capital.state

    assert state.cash >= ZERO

    assert state.equity == (state.cash + state.market_value)

    assert state.equity == (state.initial_capital + state.total_pnl)

    assert state.realized_pnl == sum((trade.pnl for trade in capital.trades), ZERO)


# ============================================================
# INITIALIZATION
# ============================================================


def test_default_initialization():

    manager = PositionManager()

    assert isinstance(manager.capital, CapitalManager)


def test_shared_capital_instance(manager, capital):

    assert manager.capital is capital


def test_maximum_holding_sessions(manager):

    assert manager.maximum_holding_sessions == 20


def test_end_liquidation_enabled(manager):

    assert manager.liquidate_at_end is True


def test_drawdown_liquidation_enabled(manager):

    assert manager.liquidate_on_drawdown is True


@pytest.mark.parametrize("invalid", [{}, [], True, 123, "invalid"])
def test_invalid_capital_manager(invalid):

    with pytest.raises(ValueError):
        PositionManager(capital_manager=invalid)


@pytest.mark.parametrize("invalid", [{}, [], True, 123, "invalid"])
def test_invalid_configuration(invalid):

    with pytest.raises(ValueError):
        PositionManager(config=invalid)


# ============================================================
# MARKET BAR VALIDATION
# ============================================================


def test_valid_market_bar():

    bar = make_bar()

    assert bar.symbol == "AAPL"
    assert bar.session == NEXT_SESSION
    assert bar.opening == D(100)
    assert bar.high == D(102)
    assert bar.low == D(99)
    assert bar.close == D(101)


def test_market_bar_immutable():

    bar = make_bar()

    with pytest.raises(FrozenInstanceError):
        bar.close = D(200)


@pytest.mark.parametrize("field", ["opening", "high", "low", "close"])
@pytest.mark.parametrize(
    "invalid", [ZERO, D(-1), D("NaN"), D("Infinity"), True, None]
)
def test_invalid_market_price(field, invalid):

    values = {
        "session": NEXT_SESSION,
        "symbol": "AAPL",
        "opening": D(100),
        "high": D(102),
        "low": D(99),
        "close": D(101),
    }

    values[field] = invalid

    with pytest.raises(ValueError):
        PositionMarketBar(**values)


@pytest.mark.parametrize("symbol", ["", " ", "aapl", " AAPL", "AAPL ", None, 123])
def test_invalid_bar_symbol(symbol):

    with pytest.raises(ValueError):
        make_bar(symbol=symbol)


@pytest.mark.parametrize("session", [None, "2026-11-23", 123, True])
def test_invalid_bar_session(session):

    with pytest.raises(ValueError):
        make_bar(session=session)


def test_high_below_opening():

    with pytest.raises(ValueError):
        make_bar(opening="105", high="102")


def test_high_below_close():

    with pytest.raises(ValueError):
        make_bar(high="102", close="105")


def test_low_above_opening():

    with pytest.raises(ValueError):
        make_bar(opening="100", low="101")


def test_low_above_close():

    with pytest.raises(ValueError):
        make_bar(low="99", close="98")


def test_flat_market_bar():

    bar = make_bar(opening="100", high="100", low="100", close="100")

    assert bar.high == bar.low


# ============================================================
# EXIT DECISION VALIDATION
# ============================================================


def test_hold_decision(manager, overnight_position):

    decision = manager.hold(overnight_position)

    assert isinstance(decision, PositionExitDecision)

    assert not decision.triggered
    assert decision.reason == "HOLD"
    assert decision.reference_price == ZERO
    assert decision.phase == "NONE"


def test_decision_immutable(manager, overnight_position):

    decision = manager.hold(overnight_position)

    with pytest.raises(FrozenInstanceError):
        decision.triggered = True


def test_invalid_position(manager):

    with pytest.raises(ValueError):
        manager.evaluate_exit({}, make_bar())


def test_invalid_market_bar(manager, overnight_position):

    with pytest.raises(ValueError):
        manager.evaluate_exit(overnight_position, {})


def test_symbol_mismatch(manager, overnight_position):

    with pytest.raises(ValueError):
        manager.evaluate_exit(overnight_position, make_bar(symbol="NVDA"))


def test_bar_precedes_entry(manager, overnight_position):

    with pytest.raises(ValueError):
        manager.evaluate_exit(overnight_position, make_bar(session=SIGNAL_DATE))


@pytest.mark.parametrize("invalid", [None, 1, "true", [], {}])
def test_invalid_forced_exit_flag(manager, overnight_position, invalid):

    with pytest.raises(ValueError):
        manager.evaluate_exit(overnight_position, make_bar(), forced_exit=invalid)


@pytest.mark.parametrize("invalid", [None, 1, "true", [], {}])
def test_invalid_final_session_flag(manager, overnight_position, invalid):

    with pytest.raises(ValueError):
        manager.evaluate_exit(overnight_position, make_bar(), final_session=invalid)


# ============================================================
# OPENING EXIT EVALUATION
# ============================================================


def test_stop_gap(manager, overnight_position):

    bar = make_bar(opening="90", high="95", low="85", close="92")

    decision = manager.evaluate_opening_exit(overnight_position, bar)

    assert decision.triggered
    assert decision.reason == "STOP_GAP"
    assert decision.reference_price == D(90)
    assert decision.phase == "OPENING"


def test_opening_exactly_at_stop(manager, overnight_position):

    bar = make_bar(opening=str(overnight_position.stop_price), low="95")

    decision = manager.evaluate_opening_exit(overnight_position, bar)

    assert decision.reason == "STOP_GAP"

    assert decision.reference_price == (overnight_position.stop_price)


def test_target_gap(manager, overnight_position):

    bar = make_bar(opening="110", high="112", low="108", close="111")

    decision = manager.evaluate_opening_exit(overnight_position, bar)

    assert decision.reason == "TAKE_PROFIT_GAP"

    assert decision.reference_price == (overnight_position.target_price)


def test_opening_exactly_at_target(manager, overnight_position):

    bar = make_bar(
        opening=str(overnight_position.target_price), high="110", close="108"
    )

    decision = manager.evaluate_opening_exit(overnight_position, bar)

    assert decision.reason == "TAKE_PROFIT_GAP"


def test_normal_opening_hold(manager, overnight_position):

    decision = manager.evaluate_opening_exit(overnight_position, make_bar())

    assert not decision.triggered


def test_forced_opening_exit(manager, overnight_position):

    decision = manager.evaluate_opening_exit(
        overnight_position, make_bar(), forced_exit=True
    )

    assert decision.reason == "DRAWDOWN_EXIT"
    assert decision.reference_price == D(100)
    assert decision.phase == "OPENING"


def test_stop_gap_precedes_forced_exit(manager, overnight_position):

    decision = manager.evaluate_opening_exit(
        overnight_position,
        make_bar(opening="90", high="95", low="85", close="92"),
        forced_exit=True,
    )

    assert decision.reason == "STOP_GAP"


def test_target_gap_precedes_forced_exit(manager, overnight_position):

    decision = manager.evaluate_opening_exit(
        overnight_position,
        make_bar(opening="110", high="112", low="108", close="111"),
        forced_exit=True,
    )

    assert decision.reason == "TAKE_PROFIT_GAP"


def test_new_position_cannot_have_overnight_gap(manager, capital):

    capital.start_session(ENTRY_DATE)

    position = open_position(capital)

    bar = make_bar(session=ENTRY_DATE, opening="90", high="95", low="85", close="92")

    decision = manager.evaluate_opening_exit(position, bar)

    assert not decision.triggered


# ============================================================
# INTRADAY EXIT EVALUATION
# ============================================================


@pytest.mark.parametrize("low", ["95", "96", "97.100"])
def test_intraday_stop(manager, overnight_position, low):

    decision = manager.evaluate_intraday_exit(overnight_position, make_bar(low=low))

    assert decision.reason == "STOP_LOSS"
    assert decision.phase == "INTRADAY"

    assert decision.reference_price == (overnight_position.stop_price)


@pytest.mark.parametrize("high", ["106.100", "108", "110"])
def test_intraday_target(manager, overnight_position, high):

    decision = manager.evaluate_intraday_exit(overnight_position, make_bar(high=high))

    assert decision.reason == "TAKE_PROFIT"
    assert decision.phase == "INTRADAY"

    assert decision.reference_price == (overnight_position.target_price)


def test_stop_target_conflict(manager, overnight_position):

    decision = manager.evaluate_intraday_exit(
        overnight_position, make_bar(high="110", low="95")
    )

    assert decision.reason == "STOP_LOSS"


def test_intraday_hold(manager, overnight_position):

    decision = manager.evaluate_intraday_exit(overnight_position, make_bar())

    assert not decision.triggered


def test_entry_session_stop_uses_conservative_price(manager, capital):

    capital.start_session(ENTRY_DATE)

    position = open_position(capital)

    decision = manager.evaluate_intraday_exit(
        position,
        make_bar(session=ENTRY_DATE, opening="90", high="102", low="85", close="101"),
    )

    assert decision.reason == "STOP_LOSS"
    assert decision.reference_price == D(90)


# ============================================================
# CLOSING EXIT EVALUATION
# ============================================================


def test_time_exit(manager, overnight_position):

    position = replace(
        overnight_position, holding_sessions=manager.maximum_holding_sessions
    )

    decision = manager.evaluate_closing_exit(position, make_bar())

    assert decision.reason == "TIME_EXIT"
    assert decision.reference_price == D(101)
    assert decision.phase == "CLOSING"


def test_before_holding_limit(manager, overnight_position):

    position = replace(
        overnight_position, holding_sessions=manager.maximum_holding_sessions - 1
    )

    decision = manager.evaluate_closing_exit(position, make_bar())

    assert not decision.triggered


def test_final_session_liquidation(manager, overnight_position):

    decision = manager.evaluate_closing_exit(
        overnight_position, make_bar(), final_session=True
    )

    assert decision.reason == "END_OF_BACKTEST"
    assert decision.reference_price == D(101)


def test_time_exit_precedes_final_exit(manager, overnight_position):

    position = replace(
        overnight_position, holding_sessions=manager.maximum_holding_sessions
    )

    decision = manager.evaluate_closing_exit(position, make_bar(), final_session=True)

    assert decision.reason == "TIME_EXIT"


def test_normal_closing_hold(manager, overnight_position):

    decision = manager.evaluate_closing_exit(overnight_position, make_bar())

    assert not decision.triggered


# ============================================================
# COMPLETE EXIT PRIORITY
# ============================================================


@pytest.mark.parametrize(
    "opening,high,low,close,forced,final,expected",
    [
        ("90", "95", "85", "92", False, False, "STOP_GAP"),
        ("110", "112", "108", "111", False, False, "TAKE_PROFIT_GAP"),
        ("100", "102", "95", "99", False, False, "STOP_LOSS"),
        ("100", "110", "99", "108", False, False, "TAKE_PROFIT"),
        ("100", "110", "95", "101", False, False, "STOP_LOSS"),
        ("100", "102", "99", "101", False, False, "HOLD"),
        ("100", "102", "99", "101", True, False, "DRAWDOWN_EXIT"),
        ("100", "102", "99", "101", False, True, "END_OF_BACKTEST"),
        ("100", "110", "95", "101", True, True, "DRAWDOWN_EXIT"),
        ("100", "110", "95", "101", False, True, "STOP_LOSS"),
        ("100", "110", "99", "108", False, True, "TAKE_PROFIT"),
    ],
)
def test_complete_exit_priority(
    manager, overnight_position, opening, high, low, close, forced, final, expected
):

    decision = manager.evaluate_exit(
        overnight_position,
        make_bar(opening=opening, high=high, low=low, close=close),
        forced_exit=forced,
        final_session=final,
    )

    assert decision.reason == expected


# ============================================================
# EXECUTION VALIDATION
# ============================================================


def test_execute_invalid_decision(manager):

    with pytest.raises(ValueError):
        manager.execute_decision({})


def test_cannot_execute_hold(manager, overnight_position):

    with pytest.raises(ValueError):
        manager.execute_decision(manager.hold(overnight_position))


def test_execute_valid_decision(manager, capital, overnight_position):

    decision = manager.evaluate_intraday_exit(overnight_position, make_bar(high="110"))

    result = manager.execute_decision(decision)

    assert result.executed
    assert result.trade.exit_reason == "TAKE_PROFIT"
    assert capital.state.open_positions == 0

    assert_accounting(capital)


def test_exit_slippage_applied_once(manager, capital, overnight_position):

    decision = manager.evaluate_intraday_exit(overnight_position, make_bar(high="110"))

    result = manager.execute_decision(decision)

    expected = decision.reference_price * (D(1) - capital.slippage)

    assert result.trade.exit_price == expected


def test_exit_fee_applied_once(manager, capital, overnight_position):

    decision = manager.evaluate_intraday_exit(overnight_position, make_bar(high="110"))

    result = manager.execute_decision(decision)

    expected = normalize_money(result.trade.exit_notional * capital.fee_rate)

    assert result.trade.exit_fee == expected


# ============================================================
# MARKET DATA VALIDATION
# ============================================================


def test_missing_market_bar(manager, capital, overnight_position):

    before = capital.state

    with pytest.raises(ValueError):
        manager.process_opening_exits({})

    assert capital.state == before


def test_wrong_market_session(manager, capital, overnight_position):

    before = capital.state

    with pytest.raises(ValueError):
        manager.process_opening_exits({"AAPL": make_bar(session=FINAL_SESSION)})

    assert capital.state == before


@pytest.mark.parametrize("invalid", [None, [], 123, True, "invalid"])
def test_invalid_bar_mapping(manager, overnight_position, invalid):

    with pytest.raises(ValueError):
        manager.process_opening_exits(invalid)


def test_invalid_bar_value(manager, overnight_position):

    with pytest.raises(ValueError):
        manager.process_opening_exits({"AAPL": {}})


def test_wrong_symbol_in_mapping(manager, overnight_position):

    with pytest.raises(ValueError):
        manager.process_opening_exits({"AAPL": make_bar(symbol="NVDA")})


# ============================================================
# EXECUTION SEQUENCING
# ============================================================


def test_opening_requires_session(manager):

    with pytest.raises(RuntimeError):
        manager.process_opening_exits({})


def test_empty_portfolio_opening(manager, capital):

    capital.start_session(ENTRY_DATE)

    assert manager.process_opening_exits({}) == ()


def test_duplicate_opening_rejected(manager, capital):

    capital.start_session(ENTRY_DATE)

    manager.process_opening_exits({})

    with pytest.raises(RuntimeError):
        manager.process_opening_exits({})


def test_opening_must_precede_entries(manager, capital):

    capital.start_session(ENTRY_DATE)

    open_position(capital)

    with pytest.raises(RuntimeError):
        manager.process_opening_exits({"AAPL": make_bar(session=ENTRY_DATE)})


def test_intraday_requires_opening(manager, capital):

    capital.start_session(ENTRY_DATE)

    with pytest.raises(RuntimeError):
        manager.process_intraday_exits({})


def test_duplicate_intraday_rejected(manager, capital):

    capital.start_session(ENTRY_DATE)

    manager.process_opening_exits({})
    manager.process_intraday_exits({})

    with pytest.raises(RuntimeError):
        manager.process_intraday_exits({})


def test_closing_requires_intraday(manager, capital):

    capital.start_session(ENTRY_DATE)

    manager.process_opening_exits({})

    with pytest.raises(RuntimeError):
        manager.process_closing_exits({})


def test_duplicate_closing_rejected(manager, capital):

    capital.start_session(ENTRY_DATE)

    manager.process_opening_exits({})
    manager.process_intraday_exits({})
    manager.process_closing_exits({})

    with pytest.raises(RuntimeError):
        manager.process_closing_exits({})


def test_closing_mark_requires_closing_phase(manager, capital):

    capital.start_session(ENTRY_DATE)

    with pytest.raises(RuntimeError):
        manager.mark_closing_positions({})


def test_new_session_resets_phase_eligibility(manager, capital):

    for session in [ENTRY_DATE, NEXT_SESSION]:
        capital.start_session(session)

        assert manager.process_opening_exits({}) == ()
        assert manager.process_intraday_exits({}) == ()
        assert manager.process_closing_exits({}) == ()

        manager.mark_closing_positions({})


# ============================================================
# MARKET VALUATION
# ============================================================


def test_opening_market_valuation(manager, capital, overnight_position):

    bars = {"AAPL": make_bar(opening="101", high="103", low="99", close="102")}

    manager.process_opening_exits(bars)

    manager.mark_open_positions(bars)

    assert capital.state.positions[0].current_price == D(101)

    assert_accounting(capital)


def test_closing_market_valuation(manager, capital, overnight_position):

    bars = {"AAPL": make_bar(close="102")}

    manager.process_opening_exits(bars)
    manager.mark_open_positions(bars)
    manager.process_intraday_exits(bars)
    manager.process_closing_exits(bars)
    manager.mark_closing_positions(bars)

    assert capital.state.positions[0].current_price == D(102)

    assert_accounting(capital)


# ============================================================
# PORTFOLIO LIFECYCLE
# ============================================================


def test_multiple_position_lifecycle(manager, capital):

    capital.start_session(ENTRY_DATE)

    manager.process_opening_exits({})

    open_position(capital, "AAPL")
    open_position(capital, "NVDA")

    entry_bars = {
        symbol: make_bar(session=ENTRY_DATE, symbol=symbol)
        for symbol in ["AAPL", "NVDA"]
    }

    manager.process_intraday_exits(entry_bars)
    manager.process_closing_exits(entry_bars)
    manager.mark_closing_positions(entry_bars)

    capital.start_session(NEXT_SESSION)

    bars = {
        "AAPL": make_bar(opening="90", high="95", low="85", close="92"),
        "NVDA": make_bar(symbol="NVDA", high="110", close="108"),
    }

    opening = manager.process_opening_exits(bars)

    assert len(opening) == 1
    assert opening[0].trade.symbol == "AAPL"
    assert opening[0].trade.exit_reason == "STOP_GAP"

    assert capital.state.open_positions == 1

    assert_accounting(capital)

    manager.mark_open_positions(bars)

    intraday = manager.process_intraday_exits(bars)

    assert len(intraday) == 1
    assert intraday[0].trade.symbol == "NVDA"
    assert intraday[0].trade.exit_reason == "TAKE_PROFIT"

    manager.process_closing_exits(bars)
    manager.mark_closing_positions(bars)

    assert capital.state.open_positions == 0
    assert capital.state.aggregate_open_risk == ZERO
    assert capital.state.total_exposure == ZERO
    assert len(capital.trades) == 2

    assert capital.trades[0].pnl < ZERO
    assert capital.trades[1].pnl > ZERO

    assert_accounting(capital)


def test_missing_second_bar_preserves_all_positions(manager, capital):

    capital.start_session(ENTRY_DATE)

    manager.process_opening_exits({})

    open_position(capital, "AAPL")
    open_position(capital, "NVDA")

    capital.start_session(NEXT_SESSION)

    before = capital.state

    with pytest.raises(ValueError):
        manager.process_opening_exits(
            {"AAPL": make_bar(opening="90", high="95", low="85", close="92")}
        )

    assert capital.state == before
    assert capital.trades == ()


def test_final_session_closes_all_positions(manager, capital, overnight_position):

    bars = {"AAPL": make_bar()}

    manager.process_opening_exits(bars)
    manager.mark_open_positions(bars)
    manager.process_intraday_exits(bars)

    results = manager.process_closing_exits(bars, final_session=True)

    manager.mark_closing_positions(bars)

    assert len(results) == 1

    assert results[0].trade.exit_reason == ("END_OF_BACKTEST")

    assert capital.state.open_positions == 0
    assert capital.state.cash == capital.state.equity

    assert_accounting(capital)


# ============================================================
# RISK PROTECTION
# ============================================================


def test_forced_drawdown_liquidation(manager, capital):

    capital.start_session(ENTRY_DATE)

    manager.process_opening_exits({})

    open_position(capital)

    capital.mark_to_market({"AAPL": D(50)})

    assert capital.state.trading_halted

    capital.start_session(NEXT_SESSION)

    results = manager.process_opening_exits({"AAPL": make_bar()})

    assert len(results) == 1

    assert results[0].trade.exit_reason == "DRAWDOWN_EXIT"

    assert capital.state.open_positions == 0
    assert capital.state.trading_halted

    assert_accounting(capital)


def test_gap_loss_records_risk_breach(manager, capital, overnight_position):

    results = manager.process_opening_exits(
        {"AAPL": make_bar(opening="90", high="95", low="85", close="92")}
    )

    assert len(results) == 1

    assert results[0].trade.risk_breached

    assert any(
        event.event_type == "PLANNED_RISK_BREACH" for event in capital.risk_events
    )


def test_normal_stop_matches_planned_risk(manager, capital, overnight_position):

    bars = {"AAPL": make_bar(low="95")}

    manager.process_opening_exits(bars)
    manager.mark_open_positions(bars)

    results = manager.process_intraday_exits(bars)

    trade = results[0].trade

    assert abs(trade.actual_loss - overnight_position.planned_risk) <= D(
        "0.000000000001"
    )

    assert_accounting(capital)


def test_automatic_exit_prevents_same_session_reentry(
    manager, capital, overnight_position
):

    manager.process_opening_exits(
        {"AAPL": make_bar(opening="90", high="95", low="85", close="92")}
    )

    result = capital.open_position(
        make_opportunity(capital, signal_date=ENTRY_DATE), D(100)
    )

    assert not result.executed


def test_intraday_exit_does_not_exist_at_opening(manager, capital, overnight_position):

    bars = {"AAPL": make_bar(high="110", close="108")}

    cash_before = capital.state.cash

    opening = manager.process_opening_exits(bars)

    assert opening == ()

    assert capital.state.cash == cash_before

    assert capital.state.open_positions == 1

    manager.mark_open_positions(bars)

    intraday = manager.process_intraday_exits(bars)

    assert len(intraday) == 1

    assert capital.state.cash > cash_before

    assert capital.state.open_positions == 0

    assert_accounting(capital)


# ============================================================
# SUMMARY
# ============================================================


def test_display_summary(manager, capsys):

    manager.display_summary()

    output = capsys.readouterr().out

    assert "POSITION MANAGER" in output
    assert "STOP_FIRST" in output
    assert "Historical simulation only" in output
    assert "Broker orders: DISABLED" in output
    assert "Live trading: DISABLED" in output
