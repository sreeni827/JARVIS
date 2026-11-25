from datetime import date, timedelta
from decimal import Decimal

import pytest

from portfolio.capital_manager import CapitalManager
from portfolio.models import PortfolioOpportunity


D = Decimal

ZERO = D("0")

PRICES = [
    "5.123456789",
    "10.987654321",
    "25.123456789123",
    "50.987654321987",
    "100",
    "100.12345678912345",
    "150.98765432198765",
    "250.12345678912345",
    "500.98765432198765",
    "999.12345678912345",
]

ATR_PERCENTAGES = [
    "0.5",
    "1",
    "2",
    "3",
    "5",
]


def make_manager():

    manager = CapitalManager()

    manager.start_session(date(2026, 11, 23))

    return manager


def make_opportunity(manager, price, atr_percent):

    price = D(str(price))

    atr_percent = D(str(atr_percent))

    return PortfolioOpportunity(
        symbol="AAPL",
        strategy="MOMENTUM",
        signal_date=manager.session - timedelta(days=1),
        score=D("85"),
        reference_price=price,
        atr=price * atr_percent / D("100"),
        stop_atr_multiplier=manager.stop_atr_multiplier,
        risk_reward_ratio=manager.risk_reward_ratio,
    )


def assert_accounting(manager):

    state = manager.state

    assert state.equity == (
        state.cash + state.market_value
    )

    assert state.equity == (
        state.initial_capital
        + state.realized_pnl
        + state.unrealized_pnl
    )

    assert state.realized_pnl == sum(
        (trade.pnl for trade in manager.trades),
        ZERO,
    )

    assert state.cash >= ZERO


@pytest.mark.parametrize("price", PRICES)
@pytest.mark.parametrize("atr_percent", ATR_PERCENTAGES)
def test_exact_normal_stop_reconciliation(
    price,
    atr_percent,
):

    manager = make_manager()

    opportunity = make_opportunity(
        manager,
        price,
        atr_percent,
    )

    entry = manager.open_position(
        opportunity,
        D(price),
    )

    assert entry.executed is True, entry.reason

    position = entry.position

    assert_accounting(manager)

    result = manager.close_position(
        position.symbol,
        position.stop_price,
        "STOP_LOSS",
    )

    assert result.executed is True

    trade = result.trade

    assert trade.actual_loss == position.planned_risk

    assert trade.actual_loss == trade.planned_risk

    assert trade.pnl == -position.planned_risk

    assert trade.risk_breached is False

    assert manager.state.open_positions == 0

    assert manager.state.aggregate_open_risk == ZERO

    assert not any(
        event.event_type == "PLANNED_RISK_BREACH"
        for event in manager.risk_events
    )

    assert_accounting(manager)


@pytest.mark.parametrize(
    "gap_percent",
    [
        "0.01",
        "0.1",
        "1",
        "5",
        "10",
    ],
)
def test_genuine_gap_breach_preserved(gap_percent):

    manager = make_manager()

    opportunity = make_opportunity(
        manager,
        "100",
        "2",
    )

    entry = manager.open_position(
        opportunity,
        D("100"),
    )

    assert entry.executed is True

    position = entry.position

    gap_price = position.stop_price * (
        D(1) - D(gap_percent) / D("100")
    )

    result = manager.close_position(
        position.symbol,
        gap_price,
        "STOP_GAP",
    )

    assert result.executed is True

    trade = result.trade

    assert trade.actual_loss > trade.planned_risk

    assert trade.risk_breached is True

    breach_events = [
        event
        for event in manager.risk_events
        if event.event_type == "PLANNED_RISK_BREACH"
    ]

    assert len(breach_events) == 1

    assert breach_events[0].observed_value == (
        trade.actual_loss
    )

    assert breach_events[0].threshold == (
        trade.planned_risk
    )

    assert_accounting(manager)


@pytest.mark.parametrize("price", PRICES)
def test_normalized_allocation_matches_position(price):

    manager = make_manager()

    opportunity = make_opportunity(
        manager,
        price,
        "2",
    )

    allocation = manager.calculate_allocation(
        opportunity,
        D(price),
    )

    assert allocation.approved is True

    result = manager.open_position(
        opportunity,
        D(price),
    )

    assert result.executed is True

    position = result.position

    assert allocation.quantity == position.quantity

    assert allocation.entry_price == position.entry_price

    assert allocation.stop_price == position.stop_price

    assert allocation.target_price == position.target_price

    assert allocation.entry_fee == position.entry_fee

    assert allocation.entry_cost == position.entry_cost

    assert allocation.position_value == position.entry_notional

    assert allocation.planned_risk == position.planned_risk

    assert position.entry_price.as_tuple().exponent == -8

    assert position.planned_risk.as_tuple().exponent == -12

    assert_accounting(manager)


@pytest.mark.parametrize("price", PRICES)
def test_profitable_exit_has_no_false_breach(price):

    manager = make_manager()

    opportunity = make_opportunity(
        manager,
        price,
        "2",
    )

    entry = manager.open_position(
        opportunity,
        D(price),
    )

    assert entry.executed is True

    position = entry.position

    result = manager.close_position(
        position.symbol,
        position.target_price,
        "TAKE_PROFIT",
    )

    assert result.executed is True

    trade = result.trade

    assert trade.pnl > ZERO

    assert trade.actual_loss == ZERO

    assert trade.risk_breached is False

    assert_accounting(manager)