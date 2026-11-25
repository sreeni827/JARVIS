from dataclasses import FrozenInstanceError, replace
from datetime import date, datetime
from decimal import Decimal as D

import pytest

from portfolio.models import (
    PortfolioAccountState,
    PortfolioExecutionResult,
    PortfolioOpportunity,
    PortfolioPosition,
    PortfolioRejection,
    PortfolioRiskEvent,
    PortfolioSnapshot,
    PortfolioTrade,
    to_decimal,
    validate_integer,
    validate_quantity,
    validate_score,
    validate_session,
    validate_symbol,
)

from portfolio.accounting_context import accounting_context

SESSION = date(2026, 11, 23)
ENTRY_SESSION = date(2026, 11, 20)


# ============================================================
# FIXTURES
# ============================================================


@pytest.fixture
def position():

    return PortfolioPosition(
        symbol="TEST",
        strategy="MOMENTUM",
        entry_date=SESSION,
        quantity=D("0.1"),
        entry_price=D(100),
        stop_price=D(95),
        target_price=D(110),
        current_price=D(100),
        planned_risk=D("0.50"),
    )


@pytest.fixture
def trade():

    return PortfolioTrade(
        symbol="TEST",
        strategy="MOMENTUM",
        entry_date=ENTRY_SESSION,
        exit_date=SESSION,
        quantity=D("0.1"),
        entry_price=D(100),
        exit_price=D(110),
        stop_price=D(95),
        target_price=D(110),
        planned_risk=D("0.52"),
        exit_reason="TAKE_PROFIT",
        entry_fee=D("0.01"),
        exit_fee=D("0.01"),
        holding_sessions=2,
    )


@pytest.fixture
def account(position):

    return PortfolioAccountState(
        initial_capital=D(100),
        cash=D(90),
        day_start_equity=D(100),
        peak_equity=D(100),
        positions=(position,),
    )


@pytest.fixture
def opportunity():

    return PortfolioOpportunity(
        symbol="TEST",
        strategy="MOMENTUM",
        signal_date=SESSION,
        score=D(75),
        reference_price=D(100),
        atr=D(2),
        stop_atr_multiplier=D("1.5"),
        risk_reward_ratio=D(2),
    )


# ============================================================
# DECIMAL VALIDATION
# ============================================================


@pytest.mark.parametrize(
    "value,expected",
    [
        (100, D(100)),
        (0.1, D("0.1")),
        ("0.123456", D("0.123456")),
        (D("10.50"), D("10.50")),
        (-5, D(-5)),
        (0, D(0)),
    ],
)
def test_decimal_conversion(value, expected):

    result = to_decimal(value)

    assert isinstance(result, D)
    assert result == expected


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        None,
        [],
        {},
        "invalid",
        float("nan"),
        float("inf"),
        float("-inf"),
        D("NaN"),
        D("Infinity"),
    ],
)
def test_invalid_decimal(value):

    with pytest.raises(ValueError):
        to_decimal(value)


@pytest.mark.parametrize("value", ["0.000001", "0.1", "1", "10.123456"])
def test_valid_quantity(value):

    assert validate_quantity(value) == D(value)


@pytest.mark.parametrize("value", ["0", "-1", "0.0000001", "0.1234567", True, "NaN"])
def test_invalid_quantity(value):

    with pytest.raises(ValueError):
        validate_quantity(value)


@pytest.mark.parametrize("value", [0, 55, 75, 100])
def test_valid_score(value):

    assert validate_score(value) == D(str(value))


@pytest.mark.parametrize("value", [-1, 101, True, "NaN", "invalid"])
def test_invalid_score(value):

    with pytest.raises(ValueError):
        validate_score(value)


@pytest.mark.parametrize("symbol", ["AAPL", "NVDA", "BRK.B", "TEST"])
def test_valid_symbol(symbol):

    assert validate_symbol(symbol) == symbol


@pytest.mark.parametrize("symbol", ["", "aapl", "AAPL ", "BAD SYMBOL", 123, None])
def test_invalid_symbol(symbol):

    with pytest.raises(ValueError):
        validate_symbol(symbol)


def test_valid_session():

    assert validate_session(SESSION, "session") == SESSION


@pytest.mark.parametrize("value", ["2026-11-23", datetime(2026, 11, 23), None, 123])
def test_invalid_session(value):

    with pytest.raises(ValueError):
        validate_session(value, "session")


@pytest.mark.parametrize("value", [-1, True, 1.5, "1", None])
def test_invalid_integer(value):

    with pytest.raises(ValueError):
        validate_integer(value, "count")


# ============================================================
# PORTFOLIO POSITION
# ============================================================


def test_position_entry_notional(position):

    assert position.entry_notional == D(10)


def test_position_entry_cost(position):

    updated = replace(position, entry_fee=D("0.01"))

    assert updated.entry_cost == D("10.01")


def test_position_market_value(position):

    assert position.market_value == D(10)


def test_position_profitable_mark(position):

    updated = replace(position, current_price=D(110))

    assert updated.market_value == D(11)
    assert updated.unrealized_pnl == D(1)
    assert updated.unrealized_return_percent == D(10)


def test_position_losing_mark(position):

    updated = replace(position, current_price=D(90))

    assert updated.unrealized_pnl == D(-1)
    assert updated.unrealized_return_percent == D(-10)


def test_position_fee_in_unrealized_pnl(position):

    updated = replace(position, entry_fee=D("0.01"))

    assert updated.unrealized_pnl == D("-0.01")


def test_position_risk_reward(position):

    assert position.risk_reward_ratio == D(2)


def test_position_is_immutable(position):

    with pytest.raises(FrozenInstanceError):
        position.current_price = D(200)


def test_position_replacement_preserves_original(position):

    updated = replace(position, current_price=D(105), holding_sessions=2)

    assert position.current_price == D(100)
    assert position.holding_sessions == 1

    assert updated.current_price == D(105)
    assert updated.holding_sessions == 2


def test_position_decimal_normalization(position):

    updated = replace(position, quantity=0.1, entry_price=100, current_price="105")

    assert isinstance(updated.quantity, D)
    assert isinstance(updated.entry_price, D)
    assert isinstance(updated.current_price, D)


@pytest.mark.parametrize(
    "field,value",
    [
        ("symbol", "test"),
        ("strategy", ""),
        ("entry_date", "2026-11-23"),
        ("quantity", 0),
        ("quantity", -1),
        ("quantity", "0.1234567"),
        ("entry_price", 0),
        ("stop_price", 0),
        ("stop_price", 100),
        ("stop_price", 105),
        ("target_price", 100),
        ("target_price", 90),
        ("current_price", 0),
        ("current_price", "NaN"),
        ("planned_risk", 0),
        ("planned_risk", -1),
        ("entry_fee", -1),
        ("holding_sessions", 0),
        ("holding_sessions", True),
        ("holding_sessions", 1.5),
    ],
)
def test_invalid_position(position, field, value):

    with pytest.raises(ValueError):
        replace(position, **{field: value})


# ============================================================
# COMPLETED PORTFOLIO TRADE
# ============================================================


def test_trade_entry_notional(trade):

    assert trade.entry_notional == D(10)


def test_trade_exit_notional(trade):

    assert trade.exit_notional == D(11)


def test_trade_entry_cost(trade):

    assert trade.entry_cost == D("10.01")


def test_trade_exit_proceeds(trade):

    assert trade.exit_proceeds == D("10.99")


def test_trade_total_fees(trade):

    assert trade.total_fees == D("0.02")


def test_trade_realized_profit(trade):

    assert trade.pnl == D("0.98")


def test_trade_return(trade):

    with accounting_context():
        expected = D("0.98") / D("10.01") * D(100)
    assert trade.return_percent == expected


def test_trade_risk_multiple(trade):

    with accounting_context():
        expected = D("0.98") / D("0.52")
    assert trade.risk_multiple == expected


def test_profitable_trade_has_no_loss(trade):

    assert trade.actual_loss == D(0)
    assert trade.risk_breached is False


def test_trade_normal_stop(trade):

    stopped = replace(trade, exit_price=D(95), exit_reason="STOP_LOSS")

    assert stopped.pnl == D("-0.52")
    assert stopped.actual_loss == D("0.52")
    assert stopped.risk_multiple == D(-1)
    assert stopped.risk_breached is False


def test_trade_gap_loss(trade):

    stopped = replace(trade, exit_price=D(90), exit_reason="STOP_GAP")

    assert stopped.pnl == D("-1.02")
    assert stopped.actual_loss == D("1.02")
    assert stopped.risk_breached is True


def test_trade_exact_breakeven(trade):

    closed = replace(trade, exit_price=D("100.20"))

    assert closed.pnl == D(0)
    assert closed.actual_loss == D(0)
    assert closed.risk_breached is False


def test_trade_same_session_exit(trade):

    closed = replace(trade, exit_date=trade.entry_date, holding_sessions=1)

    assert closed.exit_date == closed.entry_date


def test_trade_zero_execution_fees(trade):

    closed = replace(trade, entry_fee=D(0), exit_fee=D(0))

    assert closed.pnl == D(1)


def test_trade_is_immutable(trade):

    with pytest.raises(FrozenInstanceError):
        trade.exit_price = D(200)


@pytest.mark.parametrize(
    "field,value",
    [
        ("symbol", ""),
        ("strategy", ""),
        ("exit_reason", ""),
        ("entry_date", "2026-11-20"),
        ("exit_date", date(2026, 11, 19)),
        ("quantity", 0),
        ("quantity", -1),
        ("quantity", "0.1234567"),
        ("entry_price", 0),
        ("exit_price", 0),
        ("exit_price", "Infinity"),
        ("stop_price", 105),
        ("target_price", 90),
        ("planned_risk", 0),
        ("entry_fee", -1),
        ("exit_fee", -1),
        ("exit_fee", 12),
        ("holding_sessions", 0),
    ],
)
def test_invalid_trade(trade, field, value):

    with pytest.raises(ValueError):
        replace(trade, **{field: value})


# ============================================================
# SHARED ACCOUNT STATE
# ============================================================


def test_cash_only_account():

    account = PortfolioAccountState(
        initial_capital=D(100),
        cash=D(100),
        day_start_equity=D(100),
        peak_equity=D(100),
    )

    assert account.equity == D(100)
    assert account.market_value == D(0)
    assert account.open_positions == 0
    assert account.total_exposure == D(0)
    assert account.aggregate_open_risk == D(0)
    assert account.total_pnl == D(0)
    assert account.return_percent == D(0)


def test_account_equity(account):

    assert account.equity == D(100)


def test_account_cash(account):

    assert account.cash == D(90)


def test_account_market_value(account):

    assert account.market_value == D(10)


def test_account_position_count(account):

    assert account.open_positions == 1


def test_account_existing_symbols(account):

    assert account.existing_symbols == ("TEST",)


def test_account_exposure(account):

    assert account.total_exposure == D(10)
    assert account.exposure_percent == D(10)


def test_account_aggregate_risk(account):

    assert account.aggregate_open_risk == D("0.50")


def test_account_unrealized_pnl(account):

    assert account.unrealized_pnl == D(0)


def test_account_total_pnl(account):

    assert account.total_pnl == D(0)


def test_account_profitable_mark(account, position):

    updated = replace(position, current_price=D(110))

    marked = replace(account, positions=(updated,), peak_equity=D(101))

    assert marked.equity == D(101)
    assert marked.unrealized_pnl == D(1)
    assert marked.return_percent == D(1)
    assert marked.drawdown_percent == D(0)
    assert marked.daily_loss == D(0)


def test_account_losing_mark(account, position):

    updated = replace(position, current_price=D(90))

    marked = replace(account, positions=(updated,))

    assert marked.equity == D(99)
    assert marked.unrealized_pnl == D(-1)
    assert marked.total_pnl == D(-1)
    assert marked.daily_loss == D(1)
    assert marked.daily_loss_percent == D(1)
    assert marked.drawdown_percent == D(1)
    assert marked.return_percent == D(-1)


def test_account_entry_fee_accounting(account, position):

    updated = replace(position, entry_fee=D("0.01"))

    marked = replace(account, cash=D("89.99"), positions=(updated,))

    assert marked.equity == D("99.99")
    assert marked.unrealized_pnl == D("-0.01")
    assert marked.total_pnl == D("-0.01")


def test_account_realized_profit():

    closed = PortfolioAccountState(
        initial_capital=D(100),
        cash=D("100.98"),
        day_start_equity=D(100),
        peak_equity=D("100.98"),
        realized_pnl=D("0.98"),
    )

    assert closed.equity == D("100.98")
    assert closed.realized_pnl == D("0.98")
    assert closed.unrealized_pnl == D(0)
    assert closed.total_pnl == D("0.98")
    assert closed.return_percent == D("0.98")


def test_account_realized_loss():

    closed = PortfolioAccountState(
        initial_capital=D(100),
        cash=D("99.48"),
        day_start_equity=D(100),
        peak_equity=D(100),
        realized_pnl=D("-0.52"),
    )

    assert closed.equity == D("99.48")
    assert closed.daily_loss == D("0.52")
    assert closed.drawdown_percent == D("0.52")


def test_account_three_shared_positions(position):

    positions = tuple(
        replace(position, symbol=symbol, quantity=D("0.2"), planned_risk=D("0.30"))
        for symbol in ["AAA", "BBB", "CCC"]
    )

    account = PortfolioAccountState(
        initial_capital=D(100),
        cash=D(40),
        day_start_equity=D(100),
        peak_equity=D(100),
        positions=positions,
    )

    assert account.equity == D(100)
    assert account.market_value == D(60)
    assert account.open_positions == 3
    assert account.exposure_percent == D(60)
    assert account.aggregate_open_risk == D("0.90")


def test_account_position_list_is_copied(account, position):

    external = [position]

    updated = replace(account, positions=external)

    external.clear()

    assert isinstance(updated.positions, tuple)
    assert updated.open_positions == 1


def test_account_duplicate_positions(account, position):

    with pytest.raises(ValueError, match="Duplicate portfolio positions"):
        replace(account, positions=(position, position))


def test_account_invalid_position_type(account):

    with pytest.raises(ValueError):
        replace(account, positions=("TEST",))


def test_account_peak_below_equity(account):

    with pytest.raises(ValueError, match="Peak equity cannot be below current equity"):
        replace(account, peak_equity=D(99))


def test_account_peak_below_day_start(account):

    with pytest.raises(
        ValueError, match="Peak equity cannot be below day-start equity"
    ):
        replace(account, day_start_equity=D(101))


def test_account_exact_reconciliation_failure(account):

    # Lowering cash keeps equity below the peak.
    # This isolates the accounting reconciliation rule.

    with pytest.raises(ValueError, match="Portfolio accounting mismatch"):
        replace(account, cash=D(89))


def test_account_realized_pnl_mismatch(account):

    with pytest.raises(ValueError, match="Portfolio accounting mismatch"):
        replace(account, realized_pnl=D(1))


def test_account_halt_flags(account):

    halted = replace(account, trading_halted=True, daily_entry_halted=True)

    assert halted.trading_halted is True
    assert halted.daily_entry_halted is True


def test_account_is_immutable(account):

    with pytest.raises(FrozenInstanceError):
        account.cash = D(999)


@pytest.mark.parametrize(
    "field,value",
    [
        ("initial_capital", 0),
        ("initial_capital", -1),
        ("cash", -1),
        ("cash", "NaN"),
        ("day_start_equity", 0),
        ("peak_equity", 0),
        ("realized_pnl", "Infinity"),
        ("positions", None),
        ("trading_halted", "false"),
        ("trading_halted", 1),
        ("daily_entry_halted", "true"),
    ],
)
def test_invalid_account(account, field, value):

    with pytest.raises(ValueError):
        replace(account, **{field: value})


# ============================================================
# PORTFOLIO SNAPSHOT
# ============================================================


def test_snapshot_serialization(account):

    snapshot = PortfolioSnapshot(
        session=SESSION, account=account, entries=1, exits=0, rejected_opportunities=2
    )

    record = snapshot.to_record()

    assert record["session"] == "2026-11-23"
    assert D(record["cash"]) == D(90)
    assert D(record["equity"]) == D(100)
    assert D(record["market_value"]) == D(10)
    assert D(record["aggregate_open_risk"]) == D("0.50")

    assert record["open_positions"] == 1
    assert record["entries"] == 1
    assert record["exits"] == 0
    assert record["rejected_opportunities"] == 2
    assert record["trading_halted"] is False


def test_snapshot_record_is_independent(account):

    snapshot = PortfolioSnapshot(session=SESSION, account=account)

    record = snapshot.to_record()

    record["cash"] = "999"

    assert D(snapshot.to_record()["cash"]) == D(90)


@pytest.mark.parametrize(
    "field,value",
    [
        ("session", "2026-11-23"),
        ("account", None),
        ("entries", -1),
        ("exits", -1),
        ("rejected_opportunities", -1),
        ("entries", True),
        ("exits", 1.5),
    ],
)
def test_invalid_snapshot(account, field, value):

    snapshot = PortfolioSnapshot(session=SESSION, account=account)

    with pytest.raises(ValueError):
        replace(snapshot, **{field: value})


# ============================================================
# PORTFOLIO OPPORTUNITY
# ============================================================


def test_opportunity_stop_distance(opportunity):

    assert opportunity.stop_distance == D(3)


def test_opportunity_entry_prices(opportunity):

    stop, target = opportunity.prices_for_entry(D(100))

    assert stop == D(97)
    assert target == D(106)


def test_opportunity_next_open_gap(opportunity):

    stop, target = opportunity.prices_for_entry(D(110))

    assert stop == D(107)
    assert target == D(116)


def test_opportunity_invalid_calculated_stop(opportunity):

    with pytest.raises(ValueError, match="Calculated stop price is invalid"):
        opportunity.prices_for_entry(D(3))


@pytest.mark.parametrize("entry_price", [0, -1, "NaN", True])
def test_invalid_opportunity_entry(opportunity, entry_price):

    with pytest.raises(ValueError):
        opportunity.prices_for_entry(entry_price)


@pytest.mark.parametrize(
    "field,value",
    [
        ("symbol", ""),
        ("strategy", ""),
        ("signal_date", "2026-11-23"),
        ("score", -1),
        ("score", 101),
        ("reference_price", 0),
        ("atr", 0),
        ("atr", -1),
        ("stop_atr_multiplier", 0),
        ("risk_reward_ratio", 0.5),
        ("risk_reward_ratio", "Infinity"),
    ],
)
def test_invalid_opportunity(opportunity, field, value):

    with pytest.raises(ValueError):
        replace(opportunity, **{field: value})


# ============================================================
# PORTFOLIO RISK EVENT
# ============================================================


@pytest.fixture
def risk_event():

    return PortfolioRiskEvent(
        session=SESSION,
        event_type="DAILY_LOSS",
        reason="Daily loss threshold reached",
        observed_value=D("1.2"),
        threshold=D(1),
    )


def test_portfolio_risk_event(risk_event):

    assert risk_event.observed_value == D("1.2")
    assert risk_event.threshold == D(1)
    assert risk_event.symbol is None


def test_symbol_specific_risk_event(risk_event):

    event = replace(risk_event, symbol="TEST")

    assert event.symbol == "TEST"


@pytest.mark.parametrize(
    "field,value",
    [
        ("session", None),
        ("event_type", ""),
        ("reason", ""),
        ("observed_value", -1),
        ("observed_value", "NaN"),
        ("threshold", -1),
        ("symbol", "test"),
    ],
)
def test_invalid_risk_event(risk_event, field, value):

    with pytest.raises(ValueError):
        replace(risk_event, **{field: value})


# ============================================================
# PORTFOLIO REJECTION
# ============================================================


@pytest.fixture
def rejection():

    return PortfolioRejection(
        session=SESSION,
        symbol="TEST",
        strategy="MOMENTUM",
        score=D(75),
        reason="Maximum exposure reached",
    )


def test_portfolio_rejection(rejection):

    assert rejection.symbol == "TEST"
    assert rejection.score == D(75)
    assert rejection.reason == "Maximum exposure reached"


@pytest.mark.parametrize(
    "field,value",
    [
        ("session", None),
        ("symbol", ""),
        ("strategy", ""),
        ("score", -1),
        ("score", 101),
        ("reason", ""),
    ],
)
def test_invalid_rejection(rejection, field, value):

    with pytest.raises(ValueError):
        replace(rejection, **{field: value})


# ============================================================
# PORTFOLIO EXECUTION RESULT
# ============================================================


def test_successful_entry_result(position):

    result = PortfolioExecutionResult(
        executed=True, symbol="TEST", reason="Entry executed", position=position
    )

    assert result.executed is True
    assert result.position == position
    assert result.trade is None


def test_successful_exit_result(trade):

    result = PortfolioExecutionResult(
        executed=True, symbol="TEST", reason="Exit executed", trade=trade
    )

    assert result.executed is True
    assert result.trade == trade
    assert result.position is None


def test_rejected_execution_result():

    result = PortfolioExecutionResult(
        executed=False, symbol="TEST", reason="Risk limit reached"
    )

    assert result.executed is False
    assert result.position is None
    assert result.trade is None


def test_success_without_record():

    with pytest.raises(ValueError):
        PortfolioExecutionResult(executed=True, symbol="TEST", reason="Entry executed")


def test_success_with_two_records(position, trade):

    with pytest.raises(ValueError):
        PortfolioExecutionResult(
            executed=True,
            symbol="TEST",
            reason="Invalid execution",
            position=position,
            trade=trade,
        )


def test_rejection_with_position(position):

    with pytest.raises(ValueError):
        PortfolioExecutionResult(
            executed=False, symbol="TEST", reason="Rejected", position=position
        )


def test_rejection_with_trade(trade):

    with pytest.raises(ValueError):
        PortfolioExecutionResult(
            executed=False, symbol="TEST", reason="Rejected", trade=trade
        )


def test_execution_position_symbol_mismatch(position):

    with pytest.raises(ValueError, match="symbol mismatch"):
        PortfolioExecutionResult(
            executed=True, symbol="OTHER", reason="Entry", position=position
        )


def test_execution_trade_symbol_mismatch(trade):

    with pytest.raises(ValueError, match="symbol mismatch"):
        PortfolioExecutionResult(
            executed=True, symbol="OTHER", reason="Exit", trade=trade
        )


@pytest.mark.parametrize("value", [1, 0, "true", "false", None])
def test_invalid_execution_flag(value):

    with pytest.raises(ValueError):
        PortfolioExecutionResult(executed=value, symbol="TEST", reason="Test")


def test_invalid_execution_position_type():

    with pytest.raises(ValueError):
        PortfolioExecutionResult(
            executed=True, symbol="TEST", reason="Entry", position={}
        )


def test_invalid_execution_trade_type():

    with pytest.raises(ValueError):
        PortfolioExecutionResult(executed=True, symbol="TEST", reason="Exit", trade={})


# ============================================================
# COMPLETE ACCOUNTING LIFECYCLE
# ============================================================


def test_complete_profitable_accounting_lifecycle(position, trade):

    initial = PortfolioAccountState(
        initial_capital=D(100),
        cash=D(100),
        day_start_equity=D(100),
        peak_equity=D(100),
    )

    opened_position = replace(position, entry_fee=D("0.01"), planned_risk=D("0.52"))

    opened = replace(
        initial,
        cash=initial.cash - opened_position.entry_cost,
        positions=(opened_position,),
    )

    assert opened.cash == D("89.99")
    assert opened.equity == D("99.99")

    marked_position = replace(opened_position, current_price=D(110))

    marked = replace(opened, positions=(marked_position,), peak_equity=D("100.99"))

    assert marked.equity == D("100.99")
    assert marked.unrealized_pnl == D("0.99")

    completed_trade = replace(
        trade, entry_date=opened_position.entry_date, holding_sessions=1
    )

    closed = replace(
        marked,
        cash=marked.cash + completed_trade.exit_proceeds,
        positions=(),
        realized_pnl=completed_trade.pnl,
    )

    assert closed.cash == D("100.98")
    assert closed.equity == D("100.98")
    assert closed.realized_pnl == D("0.98")
    assert closed.unrealized_pnl == D(0)
    assert closed.open_positions == 0

    assert closed.equity == (closed.initial_capital + closed.total_pnl)


def test_complete_losing_accounting_lifecycle(position, trade):

    opened_position = replace(position, entry_fee=D("0.01"), planned_risk=D("0.52"))

    opened = PortfolioAccountState(
        initial_capital=D(100),
        cash=D(100) - opened_position.entry_cost,
        day_start_equity=D(100),
        peak_equity=D(100),
        positions=(opened_position,),
    )

    stopped_trade = replace(
        trade,
        entry_date=opened_position.entry_date,
        holding_sessions=1,
        exit_price=D(95),
        exit_reason="STOP_LOSS",
    )

    closed = replace(
        opened,
        cash=opened.cash + stopped_trade.exit_proceeds,
        positions=(),
        realized_pnl=stopped_trade.pnl,
    )

    assert closed.equity == D("99.48")
    assert closed.total_pnl == D("-0.52")
    assert closed.daily_loss == D("0.52")
    assert closed.drawdown_percent == D("0.52")
    assert stopped_trade.risk_breached is False
