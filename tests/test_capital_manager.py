from datetime import date
from decimal import Decimal as D

import pytest

from portfolio.capital_manager import CapitalAllocation, CapitalManager
from portfolio.models import PortfolioOpportunity

from portfolio.decimal_utils import normalize_money

ZERO = D(0)
HUNDRED = D(100)

SIGNAL_DATE = date(2026, 11, 20)
SESSION_DATE = date(2026, 11, 23)
NEXT_SESSION = date(2026, 11, 24)


# ============================================================
# TEST FIXTURES
# ============================================================


@pytest.fixture
def manager():

    return CapitalManager()


@pytest.fixture
def active_manager():

    manager = CapitalManager()

    manager.start_session(SESSION_DATE)

    return manager


def make_opportunity(manager, **overrides):

    values = {
        "symbol": "AAPL",
        "strategy": "MOMENTUM",
        "signal_date": SIGNAL_DATE,
        "score": D(90),
        "reference_price": D(100),
        "atr": D(2),
        "stop_atr_multiplier": (manager.stop_atr_multiplier),
        "risk_reward_ratio": (manager.risk_reward_ratio),
    }

    values.update(overrides)

    return PortfolioOpportunity(**values)


def open_test_position(manager, symbol="AAPL"):

    result = manager.open_position(make_opportunity(manager, symbol=symbol), D(100))

    assert result.executed

    return result.position


def assert_accounting(manager):

    state = manager.state

    assert state.cash >= ZERO

    assert state.equity == (state.cash + state.market_value)

    assert state.equity == (state.initial_capital + state.total_pnl)

    assert state.realized_pnl == sum((trade.pnl for trade in manager.trades), ZERO)


# ============================================================
# INITIALIZATION
# ============================================================


def test_initial_capital(manager):

    assert manager.initial_capital == D(100)


def test_initial_cash(manager):

    assert manager.state.cash == D(100)


def test_initial_equity(manager):

    assert manager.state.equity == D(100)


def test_initial_market_value(manager):

    assert manager.state.market_value == ZERO


def test_initial_positions(manager):

    assert manager.state.open_positions == 0


def test_initial_trades(manager):

    assert manager.trades == ()


def test_initial_risk_events(manager):

    assert manager.risk_events == ()


def test_initial_session(manager):

    assert manager.session is None


def test_initial_halt_flags(manager):

    assert not manager.state.trading_halted

    assert not manager.state.daily_entry_halted


def test_initial_position_limit(manager):

    assert manager.maximum_position_value == D(20)


def test_initial_exposure_limit(manager):

    assert manager.maximum_exposure == D(60)


def test_initial_trade_risk(manager):

    assert manager.maximum_trade_risk == D("0.50")


def test_initial_aggregate_risk(manager):

    assert manager.maximum_aggregate_risk == D(1)


def test_initial_remaining_exposure(manager):

    assert manager.remaining_exposure == D(60)


def test_initial_remaining_open_risk(manager):

    assert manager.remaining_open_risk == D(1)


def test_initial_daily_loss_limit(manager):

    assert manager.daily_loss_limit == D(1)


def test_initial_daily_risk(manager):

    assert manager.remaining_daily_risk == D(1)


def test_initial_drawdown_risk(manager):

    assert manager.remaining_drawdown_risk == D(5)


def test_invalid_configuration():

    with pytest.raises(ValueError):
        CapitalManager(config={})


def test_independent_manager_instances():

    first = CapitalManager()

    second = CapitalManager()

    first.start_session(SESSION_DATE)

    open_test_position(first)

    assert first.state.open_positions == 1

    assert second.state.open_positions == 0

    assert second.state.cash == D(100)


# ============================================================
# SESSION MANAGEMENT
# ============================================================


def test_start_session(manager):

    manager.start_session(SESSION_DATE)

    assert manager.session == SESSION_DATE


def test_session_starting_equity(active_manager):

    assert active_manager.state.day_start_equity == D(100)


def test_next_session(active_manager):

    active_manager.start_session(NEXT_SESSION)

    assert active_manager.session == NEXT_SESSION


def test_duplicate_session_rejected(active_manager):

    with pytest.raises(ValueError):
        active_manager.start_session(SESSION_DATE)


def test_backward_session_rejected(active_manager):

    with pytest.raises(ValueError):
        active_manager.start_session(SIGNAL_DATE)


@pytest.mark.parametrize("session", [None, "2026-11-23", 123, True])
def test_invalid_session(manager, session):

    with pytest.raises(ValueError):
        manager.start_session(session)


def test_allocation_requires_session(manager):

    with pytest.raises(RuntimeError):
        manager.calculate_allocation(make_opportunity(manager), D(100))


def test_entry_requires_session(manager):

    with pytest.raises(RuntimeError):
        manager.open_position(make_opportunity(manager), D(100))


def test_mark_requires_session(manager):

    with pytest.raises(RuntimeError):
        manager.mark_to_market({})


def test_exit_requires_session(manager):

    with pytest.raises(RuntimeError):
        manager.close_position("AAPL", D(100), "TEST")


def test_risk_monitor_requires_session(manager):

    with pytest.raises(RuntimeError):
        manager.evaluate_risk_thresholds()


def test_holding_sessions_increment(active_manager):

    position = open_test_position(active_manager)

    assert position.holding_sessions == 1

    active_manager.start_session(NEXT_SESSION)

    assert active_manager.state.positions[0].holding_sessions == 2


def test_next_session_updates_baseline(active_manager):

    open_test_position(active_manager)

    active_manager.mark_to_market({"AAPL": D(105)})

    previous_equity = active_manager.state.equity

    active_manager.start_session(NEXT_SESSION)

    assert active_manager.state.day_start_equity == previous_equity


# ============================================================
# CAPITAL ALLOCATION
# ============================================================


def test_valid_allocation(active_manager):

    allocation = active_manager.calculate_allocation(
        make_opportunity(active_manager), D(100)
    )

    assert isinstance(allocation, CapitalAllocation)

    assert allocation.approved

    assert allocation.quantity > ZERO


def test_allocation_does_not_change_account(active_manager):

    before = active_manager.state

    active_manager.calculate_allocation(make_opportunity(active_manager), D(100))

    assert active_manager.state == before

    assert active_manager.trades == ()


def test_allocation_entry_slippage(active_manager):

    allocation = active_manager.calculate_allocation(
        make_opportunity(active_manager), D(100)
    )

    expected = D(100) * (D(1) + active_manager.slippage)

    assert allocation.entry_price == expected


def test_allocation_stop_distance(active_manager):

    opportunity = make_opportunity(active_manager)

    allocation = active_manager.calculate_allocation(opportunity, D(100))

    expected = opportunity.atr * opportunity.stop_atr_multiplier

    assert allocation.entry_price - allocation.stop_price == expected


def test_allocation_target_distance(active_manager):

    allocation = active_manager.calculate_allocation(
        make_opportunity(active_manager), D(100)
    )

    risk_distance = allocation.entry_price - allocation.stop_price

    assert (
        allocation.target_price - allocation.entry_price
        == risk_distance * active_manager.risk_reward_ratio
    )


def test_allocation_entry_fee(active_manager):

    allocation = active_manager.calculate_allocation(
        make_opportunity(active_manager), D(100)
    )

    assert allocation.entry_fee == (allocation.position_value * active_manager.fee_rate)


def test_allocation_entry_cost(active_manager):

    allocation = active_manager.calculate_allocation(
        make_opportunity(active_manager), D(100)
    )

    assert allocation.entry_cost == (allocation.position_value + allocation.entry_fee)


def test_allocation_position_value(active_manager):

    allocation = active_manager.calculate_allocation(
        make_opportunity(active_manager), D(100)
    )

    assert allocation.position_value == (allocation.quantity * allocation.entry_price)


def test_allocation_quantity_precision(active_manager):

    allocation = active_manager.calculate_allocation(
        make_opportunity(active_manager), D(100)
    )

    assert allocation.quantity == (
        allocation.quantity.quantize(active_manager.quantity_step)
    )


def test_allocation_risk_includes_costs(active_manager):
    from decimal import ROUND_HALF_EVEN

    allocation = active_manager.calculate_allocation(
        make_opportunity(active_manager), D(100)
    )

    assert allocation.approved is True

    price_step = D("0.00000001")
    money_step = D("0.000000000001")

    stop_fill = (
        allocation.stop_price
        * (D(1) - active_manager.slippage)
    ).quantize(
        price_step,
        rounding=ROUND_HALF_EVEN,
    )

    stop_notional = (
        allocation.quantity * stop_fill
    ).quantize(
        money_step,
        rounding=ROUND_HALF_EVEN,
    )

    stop_fee = (
        stop_notional * active_manager.fee_rate
    ).quantize(
        money_step,
        rounding=ROUND_HALF_EVEN,
    )

    stop_proceeds = stop_notional - stop_fee

    expected_risk = allocation.entry_cost - stop_proceeds

    assert allocation.planned_risk == expected_risk

    assert allocation.planned_risk > D(0)

    assert allocation.planned_risk.as_tuple().exponent == -12


def test_allocation_respects_trade_risk(active_manager):

    allocation = active_manager.calculate_allocation(
        make_opportunity(active_manager), D(100)
    )

    assert allocation.planned_risk <= active_manager.maximum_trade_risk


def test_allocation_respects_position_limit(active_manager):

    allocation = active_manager.calculate_allocation(
        make_opportunity(active_manager, atr=D("0.1")), D(100)
    )

    assert allocation.approved

    assert allocation.position_value <= active_manager.maximum_position_value


def test_allocation_respects_cash(active_manager):

    allocation = active_manager.calculate_allocation(
        make_opportunity(active_manager), D(100)
    )

    assert allocation.entry_cost <= active_manager.state.cash


def test_rejection_has_zero_allocation(active_manager):

    allocation = active_manager.calculate_allocation(
        make_opportunity(active_manager, score=ZERO), D(100)
    )

    assert not allocation.approved

    assert allocation.quantity == ZERO

    assert allocation.entry_cost == ZERO

    assert allocation.planned_risk == ZERO


# ============================================================
# INVALID OPPORTUNITIES
# ============================================================


@pytest.mark.parametrize(
    "market_price", [ZERO, D(-1), D("NaN"), D("Infinity"), True, None]
)
def test_invalid_entry_price(active_manager, market_price):

    with pytest.raises(ValueError):
        active_manager.calculate_allocation(
            make_opportunity(active_manager), market_price
        )


def test_invalid_opportunity_type(active_manager):

    with pytest.raises(ValueError):
        active_manager.calculate_allocation({}, D(100))


def test_same_session_signal_rejected(active_manager):

    allocation = active_manager.calculate_allocation(
        make_opportunity(active_manager, signal_date=SESSION_DATE), D(100)
    )

    assert not allocation.approved

    assert "precede" in allocation.reason.lower()


def test_future_signal_rejected(active_manager):

    allocation = active_manager.calculate_allocation(
        make_opportunity(active_manager, signal_date=NEXT_SESSION), D(100)
    )

    assert not allocation.approved


def test_unknown_symbol_rejected(active_manager):

    allocation = active_manager.calculate_allocation(
        make_opportunity(active_manager, symbol="UNKNOWN"), D(100)
    )

    assert not allocation.approved

    assert "universe" in allocation.reason.lower()


def test_disabled_strategy_rejected(active_manager):

    allocation = active_manager.calculate_allocation(
        make_opportunity(active_manager, strategy="UNKNOWN"), D(100)
    )

    assert not allocation.approved


def test_low_score_rejected(active_manager):

    allocation = active_manager.calculate_allocation(
        make_opportunity(active_manager, score=ZERO), D(100)
    )

    assert not allocation.approved

    assert "score" in allocation.reason.lower()


def test_score_boundary(active_manager):

    allocation = active_manager.calculate_allocation(
        make_opportunity(active_manager, score=active_manager.minimum_score), D(100)
    )

    assert allocation.approved


def test_wrong_stop_multiplier(active_manager):

    allocation = active_manager.calculate_allocation(
        make_opportunity(
            active_manager,
            stop_atr_multiplier=(active_manager.stop_atr_multiplier + D(1)),
        ),
        D(100),
    )

    assert not allocation.approved

    assert "mismatch" in allocation.reason.lower()


def test_wrong_risk_reward(active_manager):

    allocation = active_manager.calculate_allocation(
        make_opportunity(
            active_manager,
            risk_reward_ratio=(active_manager.risk_reward_ratio + D(1)),
        ),
        D(100),
    )

    assert not allocation.approved


def test_invalid_calculated_stop(active_manager):

    allocation = active_manager.calculate_allocation(
        make_opportunity(active_manager, atr=D(1000)), D(100)
    )

    assert not allocation.approved


def test_duplicate_position_rejected(active_manager):

    open_test_position(active_manager)

    result = active_manager.open_position(make_opportunity(active_manager), D(100))

    assert not result.executed

    assert "already exists" in result.reason.lower()


def test_rejected_entry_preserves_state(active_manager):

    before = active_manager.state

    result = active_manager.open_position(
        make_opportunity(active_manager, score=ZERO), D(100)
    )

    assert not result.executed

    assert active_manager.state == before

    assert active_manager.trades == ()


# ============================================================
# POSITION ENTRY ACCOUNTING
# ============================================================


def test_successful_entry(active_manager):

    position = open_test_position(active_manager)

    assert position.symbol == "AAPL"

    assert active_manager.state.open_positions == 1


def test_entry_cash_deduction(active_manager):

    position = open_test_position(active_manager)

    assert active_manager.state.cash == (D(100) - position.entry_cost)


def test_entry_equity_fee_accounting(active_manager):

    position = open_test_position(active_manager)

    slippage_cost = position.quantity * (position.entry_price - D(100))

    assert active_manager.state.equity == (
        D(100) - position.entry_fee - slippage_cost
    )


def test_entry_market_value(active_manager):

    position = open_test_position(active_manager)

    assert position.current_price == D(100)

    assert active_manager.state.market_value == (position.quantity * D(100))


def test_entry_unrealized_pnl(active_manager):

    position = open_test_position(active_manager)

    slippage_cost = position.quantity * (position.entry_price - D(100))

    assert active_manager.state.unrealized_pnl == (-position.entry_fee - slippage_cost)


def test_entry_does_not_create_closed_trade(active_manager):

    open_test_position(active_manager)

    assert active_manager.trades == ()


def test_entry_aggregate_risk(active_manager):

    position = open_test_position(active_manager)

    assert active_manager.state.aggregate_open_risk == (position.planned_risk)


def test_entry_account_reconciliation(active_manager):

    open_test_position(active_manager)

    assert_accounting(active_manager)


def test_post_entry_position_limit(active_manager):

    position = open_test_position(active_manager)

    assert position.market_value <= active_manager.maximum_position_value


def test_post_entry_risk_limit(active_manager):

    position = open_test_position(active_manager)

    assert position.planned_risk <= active_manager.maximum_trade_risk


def test_capital_base_decreases_after_fee(active_manager):

    open_test_position(active_manager)

    assert active_manager.capital_base < D(100)


# ============================================================
# SHARED CAPITAL COMPETITION
# ============================================================


def test_multiple_opportunities_share_cash(active_manager):

    results = []

    for symbol in ["AAPL", "NVDA", "MSFT", "AMD", "AMZN"]:
        result = active_manager.open_position(
            make_opportunity(active_manager, symbol=symbol), D(100)
        )

        results.append(result)

        assert_accounting(active_manager)

    approved = [result for result in results if result.executed]

    rejected = [result for result in results if not result.executed]

    assert len(approved) >= 1

    assert len(rejected) >= 1

    assert active_manager.state.open_positions == len(approved)

    assert active_manager.state.cash == (
        D(100) - sum((result.position.entry_cost for result in approved), ZERO)
    )


def test_aggregate_risk_never_exceeded_on_entry(active_manager):

    for symbol in ["AAPL", "NVDA", "MSFT", "AMD"]:
        active_manager.open_position(
            make_opportunity(active_manager, symbol=symbol), D(100)
        )

        assert (
            active_manager.state.aggregate_open_risk
            <= active_manager.maximum_aggregate_risk
        )


def test_exposure_never_exceeded_on_entry(active_manager):

    for symbol in ["AAPL", "NVDA", "MSFT", "AMD"]:
        active_manager.open_position(
            make_opportunity(active_manager, symbol=symbol, atr=D("0.1")), D(100)
        )

        assert active_manager.state.total_exposure <= active_manager.maximum_exposure


def test_position_count_limit(active_manager):

    for symbol in ["AAPL", "NVDA", "MSFT", "AMD"]:
        active_manager.open_position(
            make_opportunity(active_manager, symbol=symbol, atr=D("0.1")), D(100)
        )

        assert active_manager.state.open_positions <= active_manager.maximum_positions


def test_remaining_risk_decreases(active_manager):

    before = active_manager.remaining_open_risk

    open_test_position(active_manager)

    assert active_manager.remaining_open_risk < before


def test_remaining_exposure_decreases(active_manager):

    before = active_manager.remaining_exposure

    open_test_position(active_manager)

    assert active_manager.remaining_exposure < before


# ============================================================
# MARK TO MARKET
# ============================================================


def test_profitable_market_mark(active_manager):

    open_test_position(active_manager)

    active_manager.mark_to_market({"AAPL": D(105)})

    assert active_manager.state.equity > D(100)

    assert active_manager.state.unrealized_pnl > ZERO

    assert_accounting(active_manager)


def test_losing_market_mark(active_manager):

    open_test_position(active_manager)

    active_manager.mark_to_market({"AAPL": D(99)})

    assert active_manager.state.equity < D(100)

    assert active_manager.state.unrealized_pnl < ZERO

    assert_accounting(active_manager)


def test_market_mark_preserves_cash(active_manager):

    open_test_position(active_manager)

    before = active_manager.state.cash

    active_manager.mark_to_market({"AAPL": D(105)})

    assert active_manager.state.cash == before


def test_market_mark_preserves_realized_pnl(active_manager):

    open_test_position(active_manager)

    active_manager.mark_to_market({"AAPL": D(105)})

    assert active_manager.state.realized_pnl == ZERO


def test_market_mark_updates_peak(active_manager):

    open_test_position(active_manager)

    active_manager.mark_to_market({"AAPL": D(110)})

    assert active_manager.state.peak_equity == active_manager.state.equity


def test_market_mark_does_not_reduce_peak(active_manager):

    open_test_position(active_manager)

    active_manager.mark_to_market({"AAPL": D(110)})

    peak = active_manager.state.peak_equity

    active_manager.mark_to_market({"AAPL": D(100)})

    assert active_manager.state.peak_equity == peak


def test_capital_ceiling_after_profit(active_manager):

    open_test_position(active_manager)

    active_manager.mark_to_market({"AAPL": D(110)})

    assert active_manager.state.equity > D(100)

    assert active_manager.capital_base == D(100)


def test_missing_market_price(active_manager):

    open_test_position(active_manager)

    before = active_manager.state

    with pytest.raises(ValueError):
        active_manager.mark_to_market({})

    assert active_manager.state == before


@pytest.mark.parametrize("prices", [None, [], "invalid", 100])
def test_invalid_market_mapping(active_manager, prices):

    with pytest.raises(ValueError):
        active_manager.mark_to_market(prices)


@pytest.mark.parametrize("price", [ZERO, D(-1), D("NaN"), D("Infinity"), True])
def test_invalid_market_mark(active_manager, price):

    open_test_position(active_manager)

    before = active_manager.state

    with pytest.raises(ValueError):
        active_manager.mark_to_market({"AAPL": price})

    assert active_manager.state == before


def test_atomic_multiple_position_mark(active_manager):

    open_test_position(active_manager, "AAPL")

    open_test_position(active_manager, "NVDA")

    before = active_manager.state

    with pytest.raises(ValueError):
        active_manager.mark_to_market({"AAPL": D(105), "NVDA": D("NaN")})

    assert active_manager.state == before


def test_cash_only_market_mark(active_manager):

    before = active_manager.state

    active_manager.mark_to_market({})

    assert active_manager.state == before


# ============================================================
# POSITION EXIT
# ============================================================


def test_successful_exit(active_manager):

    open_test_position(active_manager)

    result = active_manager.close_position("AAPL", D(105), "TEST_EXIT")

    assert result.executed

    assert result.trade is not None

    assert active_manager.state.open_positions == 0


def test_exit_slippage(active_manager):

    open_test_position(active_manager)

    result = active_manager.close_position("AAPL", D(105), "TEST_EXIT")

    expected = D(105) * (D(1) - active_manager.slippage)

    assert result.trade.exit_price == expected


def test_exit_fee(active_manager):

    open_test_position(active_manager)

    result = active_manager.close_position("AAPL", D(105), "TEST_EXIT")

    assert result.trade.exit_fee == (
        normalize_money(result.trade.exit_notional * active_manager.fee_rate)
    )


def test_exit_releases_risk(active_manager):

    open_test_position(active_manager)

    active_manager.close_position("AAPL", D(105), "TEST_EXIT")

    assert active_manager.state.aggregate_open_risk == ZERO


def test_exit_releases_exposure(active_manager):

    open_test_position(active_manager)

    active_manager.close_position("AAPL", D(105), "TEST_EXIT")

    assert active_manager.state.total_exposure == ZERO


def test_profitable_exit_accounting(active_manager):

    open_test_position(active_manager)

    result = active_manager.close_position("AAPL", D(105), "TEST_EXIT")

    assert result.trade.pnl > ZERO

    assert active_manager.state.realized_pnl == (result.trade.pnl)

    assert active_manager.state.equity == (D(100) + result.trade.pnl)

    assert_accounting(active_manager)


def test_losing_exit_accounting(active_manager):

    open_test_position(active_manager)

    result = active_manager.close_position("AAPL", D(99), "TEST_EXIT")

    assert result.trade.pnl < ZERO

    assert active_manager.state.realized_pnl == (result.trade.pnl)

    assert_accounting(active_manager)


def test_exit_without_position(active_manager):

    before = active_manager.state

    result = active_manager.close_position("AAPL", D(100), "TEST_EXIT")

    assert not result.executed

    assert active_manager.state == before


def test_duplicate_exit_rejected(active_manager):

    open_test_position(active_manager)

    active_manager.close_position("AAPL", D(105), "TEST_EXIT")

    result = active_manager.close_position("AAPL", D(105), "TEST_EXIT")

    assert not result.executed

    assert len(active_manager.trades) == 1


def test_same_session_reentry_rejected(active_manager):

    open_test_position(active_manager)

    active_manager.close_position("AAPL", D(105), "TEST_EXIT")

    result = active_manager.open_position(make_opportunity(active_manager), D(100))

    assert not result.executed

    assert "reentry" in result.reason.lower()


def test_next_session_reentry_permitted(active_manager):

    open_test_position(active_manager)

    active_manager.close_position("AAPL", D(105), "TEST_EXIT")

    active_manager.start_session(NEXT_SESSION)

    result = active_manager.open_position(
        make_opportunity(active_manager, signal_date=SESSION_DATE), D(100)
    )

    assert result.executed


@pytest.mark.parametrize("price", [ZERO, D(-1), D("NaN"), D("Infinity"), True])
def test_invalid_exit_price(active_manager, price):

    open_test_position(active_manager)

    before = active_manager.state

    with pytest.raises(ValueError):
        active_manager.close_position("AAPL", price, "TEST_EXIT")

    assert active_manager.state == before

    assert active_manager.trades == ()


def test_invalid_exit_reason(active_manager):

    open_test_position(active_manager)

    before = active_manager.state

    with pytest.raises(ValueError):
        active_manager.close_position("AAPL", D(105), "")

    assert active_manager.state == before


def test_trade_holding_sessions(active_manager):

    open_test_position(active_manager)

    active_manager.start_session(NEXT_SESSION)

    result = active_manager.close_position("AAPL", D(105), "TEST_EXIT")

    assert result.trade.holding_sessions == 2


# ============================================================
# RISK PROTECTION
# ============================================================


def test_normal_stop_matches_planned_risk(active_manager):

    position = open_test_position(active_manager)

    result = active_manager.close_position("AAPL", position.stop_price, "STOP_LOSS")

    assert result.executed

    assert abs(result.trade.actual_loss - position.planned_risk) <= D("0.000000000001")


def test_gap_loss_exceeds_planned_risk(active_manager):

    position = open_test_position(active_manager)

    result = active_manager.close_position(
        "AAPL", position.stop_price - D(5), "STOP_GAP"
    )

    assert result.trade.risk_breached

    assert result.trade.actual_loss > position.planned_risk


def test_gap_loss_records_event(active_manager):

    position = open_test_position(active_manager)

    active_manager.close_position("AAPL", position.stop_price - D(5), "STOP_GAP")

    assert any(
        event.event_type == "PLANNED_RISK_BREACH"
        for event in active_manager.risk_events
    )


def test_daily_loss_halt(active_manager):

    open_test_position(active_manager)

    active_manager.mark_to_market({"AAPL": D(80)})

    assert active_manager.state.daily_entry_halted


def test_daily_halt_rejects_entries(active_manager):

    open_test_position(active_manager)

    active_manager.mark_to_market({"AAPL": D(80)})

    result = active_manager.open_position(
        make_opportunity(active_manager, symbol="NVDA"), D(100)
    )

    assert not result.executed

    assert "halt" in result.reason.lower()


def test_daily_halt_persists_after_recovery(active_manager):

    open_test_position(active_manager)

    active_manager.mark_to_market({"AAPL": D(80)})

    active_manager.mark_to_market({"AAPL": D(110)})

    assert active_manager.state.daily_entry_halted


def test_daily_halt_resets_next_session(active_manager):

    open_test_position(active_manager)

    active_manager.mark_to_market({"AAPL": D(80)})

    assert active_manager.state.daily_entry_halted

    assert not active_manager.state.trading_halted

    active_manager.start_session(NEXT_SESSION)

    assert not active_manager.state.daily_entry_halted


def test_daily_loss_event_recorded_once(active_manager):

    open_test_position(active_manager)

    active_manager.mark_to_market({"AAPL": D(80)})

    active_manager.mark_to_market({"AAPL": D(79)})

    events = [
        event
        for event in active_manager.risk_events
        if event.event_type == "DAILY_LOSS"
    ]

    assert len(events) == 1


def test_drawdown_halt(active_manager):

    open_test_position(active_manager)

    active_manager.mark_to_market({"AAPL": D(1)})

    assert active_manager.state.trading_halted


def test_drawdown_halt_persists_next_session(active_manager):

    open_test_position(active_manager)

    active_manager.mark_to_market({"AAPL": D(1)})

    active_manager.start_session(NEXT_SESSION)

    assert active_manager.state.trading_halted


def test_drawdown_halt_persists_after_recovery(active_manager):

    open_test_position(active_manager)

    active_manager.mark_to_market({"AAPL": D(1)})

    active_manager.mark_to_market({"AAPL": D(110)})

    assert active_manager.state.trading_halted


def test_drawdown_halt_rejects_entry(active_manager):

    open_test_position(active_manager)

    active_manager.mark_to_market({"AAPL": D(1)})

    result = active_manager.open_position(
        make_opportunity(active_manager, symbol="NVDA"), D(100)
    )

    assert not result.executed

    assert "halt" in result.reason.lower()


def test_drawdown_event_recorded_once(active_manager):

    open_test_position(active_manager)

    active_manager.mark_to_market({"AAPL": D(2)})

    active_manager.mark_to_market({"AAPL": D(1)})

    events = [
        event
        for event in active_manager.risk_events
        if event.event_type == "MAXIMUM_DRAWDOWN"
    ]

    assert len(events) == 1


def test_exit_permitted_during_halt(active_manager):

    open_test_position(active_manager)

    active_manager.mark_to_market({"AAPL": D(1)})

    assert active_manager.state.trading_halted

    result = active_manager.close_position("AAPL", D(1), "DRAWDOWN_EXIT")

    assert result.executed

    assert active_manager.state.open_positions == 0

    assert active_manager.state.trading_halted

    assert_accounting(active_manager)


def test_risk_events_are_read_only(active_manager):

    open_test_position(active_manager)

    active_manager.mark_to_market({"AAPL": D(1)})

    assert isinstance(active_manager.risk_events, tuple)


def test_trades_are_read_only(active_manager):

    open_test_position(active_manager)

    active_manager.close_position("AAPL", D(105), "TEST_EXIT")

    assert isinstance(active_manager.trades, tuple)


# ============================================================
# COMPLETE PORTFOLIO ACCOUNTING
# ============================================================


def test_complete_multiple_position_lifecycle(active_manager):

    open_test_position(active_manager, "AAPL")

    open_test_position(active_manager, "NVDA")

    assert active_manager.state.open_positions == 2

    assert_accounting(active_manager)

    active_manager.mark_to_market({"AAPL": D(105), "NVDA": D(99)})

    assert_accounting(active_manager)

    active_manager.start_session(NEXT_SESSION)

    first = active_manager.close_position("AAPL", D(105), "PROFIT_EXIT")

    second = active_manager.close_position("NVDA", D(99), "LOSS_EXIT")

    assert first.executed

    assert second.executed

    assert first.trade.pnl > ZERO

    assert second.trade.pnl < ZERO

    assert active_manager.state.open_positions == 0

    assert active_manager.state.market_value == ZERO

    assert active_manager.state.aggregate_open_risk == ZERO

    assert active_manager.state.cash == (active_manager.state.equity)

    assert active_manager.state.realized_pnl == (first.trade.pnl + second.trade.pnl)

    assert len(active_manager.trades) == 2

    assert_accounting(active_manager)


def test_accounting_after_extreme_loss(active_manager):

    open_test_position(active_manager)

    active_manager.mark_to_market({"AAPL": D(1)})

    active_manager.close_position("AAPL", D(1), "EXTREME_LOSS")

    assert active_manager.state.cash > ZERO

    assert active_manager.state.trading_halted

    assert_accounting(active_manager)


def test_accounting_after_rejected_opportunities(active_manager):

    open_test_position(active_manager)

    before = active_manager.state

    for _ in range(10):
        result = active_manager.open_position(
            make_opportunity(active_manager), D(100)
        )

        assert not result.executed

    assert active_manager.state == before

    assert_accounting(active_manager)
    # ============================================================


# V0.8.6.7.2B - ENTRY VALUATION HARDENING
# ============================================================


def test_entry_price_and_market_price_are_distinct(active_manager):

    position = open_test_position(active_manager)

    assert position.entry_price == (D(100) * (D(1) + active_manager.slippage))

    assert position.current_price == D(100)

    assert position.entry_price > position.current_price


def test_entry_slippage_reduces_equity(active_manager):

    position = open_test_position(active_manager)

    slippage_cost = position.quantity * (position.entry_price - position.current_price)

    assert slippage_cost > D(0)

    assert active_manager.state.equity == (
        D(100) - position.entry_fee - slippage_cost
    )


def test_entry_cash_uses_execution_cost(active_manager):

    position = open_test_position(active_manager)

    expected_cost = position.quantity * position.entry_price + position.entry_fee

    assert position.entry_cost == expected_cost

    assert active_manager.state.cash == (D(100) - expected_cost)


def test_entry_equity_matches_market_valuation(active_manager):

    position = open_test_position(active_manager)

    expected_market_value = position.quantity * D(100)

    assert active_manager.state.equity == (
        active_manager.state.cash + expected_market_value
    )


def test_unchanged_market_price_preserves_equity(active_manager):

    open_test_position(active_manager)

    previous_equity = active_manager.state.equity

    active_manager.mark_to_market({"AAPL": D(100)})

    assert active_manager.state.equity == previous_equity


def test_entry_does_not_create_artificial_peak(active_manager):

    open_test_position(active_manager)

    assert active_manager.state.peak_equity == D(100)

    assert active_manager.state.equity < D(100)


def test_entry_drawdown_includes_execution_costs(active_manager):

    position = open_test_position(active_manager)

    slippage_cost = position.quantity * (position.entry_price - position.current_price)

    expected_loss = position.entry_fee + slippage_cost

    expected_drawdown = expected_loss / D(100) * D(100)

    assert active_manager.state.drawdown_percent == expected_drawdown


def test_corrected_post_entry_risk_limits(active_manager):

    position = open_test_position(active_manager)

    assert position.planned_risk <= (active_manager.maximum_trade_risk)

    assert (
        active_manager.state.aggregate_open_risk
        <= active_manager.maximum_aggregate_risk
    )

    assert active_manager.state.total_exposure <= active_manager.maximum_exposure

    assert position.market_value <= active_manager.maximum_position_value


def test_corrected_entry_accounting_identity(active_manager):

    open_test_position(active_manager)

    state = active_manager.state

    assert state.equity == (state.cash + state.market_value)

    assert state.equity == (
        state.initial_capital + state.realized_pnl + state.unrealized_pnl
    )


def test_immediate_round_trip_costs(active_manager):

    position = open_test_position(active_manager)

    result = active_manager.close_position("AAPL", D(100), "SYNTHETIC_EXIT")

    assert result.executed

    trade = result.trade

    entry_slippage = position.quantity * (position.entry_price - D(100))

    exit_slippage = position.quantity * (D(100) - trade.exit_price)

    expected_loss = entry_slippage + exit_slippage + trade.entry_fee + trade.exit_fee

    assert abs(trade.actual_loss - expected_loss) <= D("0.000000000001")

    assert active_manager.state.open_positions == 0

    assert active_manager.state.equity == (D(100) + trade.pnl)
