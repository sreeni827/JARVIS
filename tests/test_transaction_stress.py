from datetime import date, timedelta
from decimal import (
    Decimal,
    Context,
    getcontext,
    localcontext,
    ROUND_HALF_EVEN,
    ROUND_DOWN,
    Inexact,
    Rounded,
)

import pytest

from portfolio.capital_manager import CapitalManager
from portfolio.models import PortfolioOpportunity


D = Decimal

ZERO = D("0")
ONE = D("1")

PRICE_STEP = D("0.00000001")
MONEY_STEP = D("0.000000000001")

START_SESSION = date(2020, 1, 1)

TRANSACTION_COUNTS = [
    10,
    100,
    500,
    1000,
]

SYMBOLS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMD",
    "AMZN",
    "META",
    "GOOGL",
    "TSLA",
    "WMT",
    "COST",
    "SPY",
    "QQQ",
]

MARKET_PRICES = [
    D("5.123456789"),
    D("10.987654321"),
    D("25.123456789123"),
    D("50.987654321987"),
    D("100.12345678912345"),
    D("150.98765432198765"),
    D("250.12345678912345"),
    D("500.98765432198765"),
    D("999.12345678912345"),
]


# =========================================================
# INDEPENDENT REFERENCE ACCOUNTING
# =========================================================


def reference_context():

    return localcontext(
        Context(
            prec=80,
            rounding=ROUND_HALF_EVEN,
            Emin=-999999,
            Emax=999999,
        )
    )


def reference_price(value):

    with reference_context():

        return value.quantize(
            PRICE_STEP,
            rounding=ROUND_HALF_EVEN,
        )


def reference_money(value):

    with reference_context():

        return value.quantize(
            MONEY_STEP,
            rounding=ROUND_HALF_EVEN,
        )


def reference_notional(quantity, price):

    with reference_context():

        return reference_money(
            quantity * price
        )


def reference_fee(notional, fee_rate):

    with reference_context():

        return reference_money(
            notional * fee_rate
        )


def reference_buy_price(price, slippage):

    with reference_context():

        return reference_price(
            reference_price(price)
            * (ONE + slippage)
        )


def reference_sell_price(price, slippage):

    with reference_context():

        return reference_price(
            reference_price(price)
            * (ONE - slippage)
        )


def context_signature():

    context = getcontext()

    return (
        context.prec,
        context.rounding,
        context.Emin,
        context.Emax,
        context.capitals,
        context.clamp,
        dict(context.flags),
        dict(context.traps),
    )


# =========================================================
# SYNTHETIC MARKET GENERATION
# =========================================================


def create_market_scenario(index):

    with reference_context():

        symbol = SYMBOLS[
            index % len(SYMBOLS)
        ]

        base_price = MARKET_PRICES[
            index % len(MARKET_PRICES)
        ]

        variation = (
            D(index % 17)
            / D("1000")
        )

        market_price = reference_price(
            base_price * (ONE + variation)
        )

        atr = market_price * D("0.02")

        mark_price = reference_price(
            market_price * D("1.005")
        )

        # Three profitable transactions followed
        # by one deliberately small losing transaction.

        if index % 4 == 3:

            exit_price = reference_price(
                market_price * D("0.997")
            )

            expected_profit = False

        else:

            exit_price = reference_price(
                market_price * D("1.02")
            )

            expected_profit = True

        return {
            "symbol": symbol,
            "market_price": market_price,
            "atr": atr,
            "mark_price": mark_price,
            "exit_price": exit_price,
            "expected_profit": expected_profit,
        }


# =========================================================
# ACCOUNTING RECONCILIATION
# =========================================================


def assert_account_identity(manager):

    state = manager.state

    with reference_context():

        assert state.equity == (
            state.cash
            + state.market_value
        )

        assert state.equity == (
            state.initial_capital
            + state.realized_pnl
            + state.unrealized_pnl
        )

        assert state.cash >= ZERO

        assert state.peak_equity >= state.equity

        assert state.open_positions == len(
            state.positions
        )


# =========================================================
# TRANSACTION STRESS ENGINE
# =========================================================


def run_transaction_sequence(transaction_count):

    manager = CapitalManager()

    expected_cash = manager.state.initial_capital

    expected_realized_pnl = ZERO

    expected_total_fees = ZERO

    winning_trades = 0

    losing_trades = 0

    total_entries = 0

    total_exits = 0

    slippage = manager.slippage

    fee_rate = manager.fee_rate

    for index in range(transaction_count):

        session = (
            START_SESSION
            + timedelta(days=index)
        )

        market = create_market_scenario(index)

        symbol = market["symbol"]

        manager.start_session(session)

        assert manager.state.day_start_equity == (
            expected_cash
        )

        opportunity = PortfolioOpportunity(
            symbol=symbol,
            strategy="MOMENTUM",
            signal_date=(
                session - timedelta(days=1)
            ),
            score=D("85"),
            reference_price=market["market_price"],
            atr=market["atr"],
            stop_atr_multiplier=(
                manager.stop_atr_multiplier
            ),
            risk_reward_ratio=(
                manager.risk_reward_ratio
            ),
        )

        allocation = manager.calculate_allocation(
            opportunity,
            market["market_price"],
        )

        assert allocation.approved, (
            f"Transaction {index}: "
            f"{allocation.reason}"
        )

        entry = manager.open_position(
            opportunity,
            market["market_price"],
        )

        assert entry.executed, (
            f"Transaction {index}: "
            f"{entry.reason}"
        )

        total_entries += 1

        position = entry.position

        quantity = position.quantity

        # -------------------------------------------------
        # Independently calculate entry accounting.
        # -------------------------------------------------

        with reference_context():

            expected_entry_price = (
                reference_buy_price(
                    market["market_price"],
                    slippage,
                )
            )

            expected_entry_notional = (
                reference_notional(
                    quantity,
                    expected_entry_price,
                )
            )

            expected_entry_fee = reference_fee(
                expected_entry_notional,
                fee_rate,
            )

            expected_entry_cost = reference_money(
                expected_entry_notional
                + expected_entry_fee
            )

            expected_cash = reference_money(
                expected_cash
                - expected_entry_cost
            )

            expected_market_value = (
                reference_notional(
                    quantity,
                    market["market_price"],
                )
            )

            expected_unrealized_pnl = (
                reference_money(
                    expected_market_value
                    - expected_entry_cost
                )
            )

            expected_equity = reference_money(
                expected_cash
                + expected_market_value
            )

            expected_stop_fill = (
                reference_sell_price(
                    position.stop_price,
                    slippage,
                )
            )

            expected_stop_notional = (
                reference_notional(
                    quantity,
                    expected_stop_fill,
                )
            )

            expected_stop_fee = reference_fee(
                expected_stop_notional,
                fee_rate,
            )

            expected_planned_risk = (
                reference_money(
                    expected_entry_cost
                    - expected_stop_notional
                    + expected_stop_fee
                )
            )

        assert position.quantity == (
            allocation.quantity
        )

        assert position.entry_price == (
            expected_entry_price
        )

        assert position.entry_fee == (
            expected_entry_fee
        )

        assert position.entry_cost == (
            expected_entry_cost
        )

        assert position.planned_risk == (
            expected_planned_risk
        )

        assert manager.state.cash == (
            expected_cash
        )

        assert manager.state.market_value == (
            expected_market_value
        )

        assert manager.state.unrealized_pnl == (
            expected_unrealized_pnl
        )

        assert manager.state.equity == (
            expected_equity
        )

        assert_account_identity(manager)

        # -------------------------------------------------
        # Independently calculate market valuation.
        # -------------------------------------------------

        manager.mark_to_market(
            {
                symbol: market["mark_price"],
            }
        )

        with reference_context():

            expected_market_value = (
                reference_notional(
                    quantity,
                    market["mark_price"],
                )
            )

            expected_unrealized_pnl = (
                reference_money(
                    expected_market_value
                    - expected_entry_cost
                )
            )

            expected_equity = reference_money(
                expected_cash
                + expected_market_value
            )

        assert manager.state.market_value == (
            expected_market_value
        )

        assert manager.state.unrealized_pnl == (
            expected_unrealized_pnl
        )

        assert manager.state.equity == (
            expected_equity
        )

        assert_account_identity(manager)

        # -------------------------------------------------
        # Close the position.
        # -------------------------------------------------

        exit_result = manager.close_position(
            symbol,
            market["exit_price"],
            "STRESS_TEST_EXIT",
        )

        assert exit_result.executed, (
            f"Transaction {index}: "
            f"{exit_result.reason}"
        )

        total_exits += 1

        trade = exit_result.trade

        # -------------------------------------------------
        # Independently calculate exit accounting.
        # -------------------------------------------------

        with reference_context():

            expected_exit_price = (
                reference_sell_price(
                    market["exit_price"],
                    slippage,
                )
            )

            expected_exit_notional = (
                reference_notional(
                    quantity,
                    expected_exit_price,
                )
            )

            expected_exit_fee = reference_fee(
                expected_exit_notional,
                fee_rate,
            )

            expected_exit_proceeds = (
                reference_money(
                    expected_exit_notional
                    - expected_exit_fee
                )
            )

            expected_trade_pnl = reference_money(
                expected_exit_proceeds
                - expected_entry_cost
            )

            expected_cash = reference_money(
                expected_cash
                + expected_exit_proceeds
            )

            expected_realized_pnl = (
                reference_money(
                    expected_realized_pnl
                    + expected_trade_pnl
                )
            )

            expected_total_fees = reference_money(
                expected_total_fees
                + expected_entry_fee
                + expected_exit_fee
            )

        assert trade.exit_price == (
            expected_exit_price
        )

        assert trade.exit_fee == (
            expected_exit_fee
        )

        assert trade.exit_proceeds == (
            expected_exit_proceeds
        )

        assert trade.pnl == expected_trade_pnl

        assert trade.risk_breached is False

        if market["expected_profit"]:

            assert trade.pnl > ZERO

            winning_trades += 1

        else:

            assert trade.pnl < ZERO

            losing_trades += 1

        assert manager.state.cash == (
            expected_cash
        )

        assert manager.state.equity == (
            expected_cash
        )

        assert manager.state.realized_pnl == (
            expected_realized_pnl
        )

        assert manager.state.unrealized_pnl == ZERO

        assert manager.state.market_value == ZERO

        assert manager.state.aggregate_open_risk == ZERO

        assert manager.state.open_positions == 0

        assert manager.state.trading_halted is False

        assert manager.state.daily_entry_halted is False

        assert_account_identity(manager)

    # -----------------------------------------------------
    # Complete sequence reconciliation.
    # -----------------------------------------------------

    assert total_entries == transaction_count

    assert total_exits == transaction_count

    assert len(manager.trades) == transaction_count

    assert winning_trades + losing_trades == (
        transaction_count
    )

    assert losing_trades == (
        transaction_count // 4
    )

    with reference_context():

        actual_total_fees = sum(
            (
                trade.total_fees
                for trade in manager.trades
            ),
            ZERO,
        )

        actual_total_pnl = sum(
            (
                trade.pnl
                for trade in manager.trades
            ),
            ZERO,
        )

        expected_final_equity = (
            manager.state.initial_capital
            + expected_realized_pnl
        )

    assert actual_total_fees == (
        expected_total_fees
    )

    assert actual_total_pnl == (
        expected_realized_pnl
    )

    assert manager.state.equity == (
        expected_final_equity
    )

    assert manager.state.cash == (
        expected_final_equity
    )

    assert len(manager.risk_events) == 0

    return {
        "transactions": transaction_count,
        "entries": total_entries,
        "exits": total_exits,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "initial_capital": (
            manager.state.initial_capital
        ),
        "final_cash": manager.state.cash,
        "final_equity": manager.state.equity,
        "realized_pnl": manager.state.realized_pnl,
        "total_fees": actual_total_fees,
        "peak_equity": manager.state.peak_equity,
        "drawdown_percent": (
            manager.state.drawdown_percent
        ),
        "risk_events": len(manager.risk_events),
        "open_positions": manager.state.open_positions,
    }


# =========================================================
# PERMANENT AUTOMATED STRESS TESTS
# =========================================================


@pytest.mark.parametrize(
    "transaction_count",
    TRANSACTION_COUNTS,
)
def test_repeated_transaction_stress(
    transaction_count,
):

    before = context_signature()

    try:

        result = run_transaction_sequence(
            transaction_count
        )

        assert result["transactions"] == (
            transaction_count
        )

        assert result["risk_events"] == 0

        assert result["open_positions"] == 0

    finally:

        assert context_signature() == before


@pytest.mark.parametrize(
    "transaction_count",
    TRANSACTION_COUNTS,
)
def test_repeated_transaction_hostile_context(
    transaction_count,
):

    with localcontext() as caller:

        caller.prec = 10

        caller.rounding = ROUND_DOWN

        caller.Emin = -2

        caller.Emax = 2

        caller.traps[Inexact] = True

        caller.traps[Rounded] = True

        before = context_signature()

        try:

            result = run_transaction_sequence(
                transaction_count
            )

            assert result["transactions"] == (
                transaction_count
            )

            assert result["risk_events"] == 0

            assert result["open_positions"] == 0

        finally:

            assert context_signature() == before


def test_transaction_sequence_context_equivalence():

    baseline = run_transaction_sequence(100)

    with localcontext() as caller:

        caller.prec = 10

        caller.rounding = ROUND_DOWN

        caller.Emin = -2

        caller.Emax = 2

        caller.traps[Inexact] = True

        caller.traps[Rounded] = True

        before = context_signature()

        try:

            result = run_transaction_sequence(100)

            assert result == baseline

        finally:

            assert context_signature() == before