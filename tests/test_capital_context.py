from datetime import date, timedelta
from decimal import (
    Decimal,
    getcontext,
    localcontext,
    ROUND_HALF_EVEN,
    ROUND_HALF_UP,
    ROUND_DOWN,
    ROUND_UP,
    ROUND_FLOOR,
    ROUND_CEILING,
    Inexact,
    Rounded,
)

import pytest

from portfolio.accounting_context import (
    accounting_context,
)

from portfolio.capital_manager import (
    CapitalManager,
)

from portfolio.models import (
    PortfolioOpportunity,
)

from tests.decimal_context_diagnostic import (
    run_scenario,
)


D = Decimal

SESSION = date(2026, 11, 23)

MARKET_PRICE = D("100.12345678912345")

PRECISIONS = [
    10,
    16,
    20,
    28,
    32,
    50,
    64,
]

ROUNDING_MODES = [
    ROUND_HALF_EVEN,
    ROUND_HALF_UP,
    ROUND_DOWN,
    ROUND_UP,
    ROUND_FLOOR,
    ROUND_CEILING,
]


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


def configure_hostile_context(
    context,
    precision=10,
    rounding=ROUND_DOWN,
):

    context.prec = precision

    context.rounding = rounding

    context.Emin = -2

    context.Emax = 2

    context.traps[Inexact] = True

    context.traps[Rounded] = True


def create_opportunity(
    manager,
    session=SESSION,
    symbol="AAPL",
    score=D("85"),
):

    return PortfolioOpportunity(
        symbol=symbol,
        strategy="MOMENTUM",
        signal_date=session - timedelta(days=1),
        score=score,
        reference_price=MARKET_PRICE,
        atr=D("2.002469135782469"),
        stop_atr_multiplier=(
            manager.stop_atr_multiplier
        ),
        risk_reward_ratio=(
            manager.risk_reward_ratio
        ),
    )


def assert_accounting_identity(manager):

    state = manager.state

    with accounting_context():

        assert state.equity == (
            state.cash
            + state.market_value
        )

        assert state.equity == (
            state.initial_capital
            + state.realized_pnl
            + state.unrealized_pnl
        )

        assert state.cash >= D("0")

        assert state.peak_equity >= state.equity


@pytest.fixture
def baseline():

    with accounting_context():

        return run_scenario()


# =========================================================
# CONTEXT ISOLATION MATRIX
# =========================================================


@pytest.mark.parametrize(
    "precision",
    PRECISIONS,
)
@pytest.mark.parametrize(
    "rounding",
    ROUNDING_MODES,
)
def test_capital_context_matrix(
    baseline,
    precision,
    rounding,
):

    with localcontext() as caller:

        configure_hostile_context(
            caller,
            precision,
            rounding,
        )

        before = context_signature()

        try:

            result = run_scenario()

            assert result == baseline

        finally:

            assert context_signature() == before


# =========================================================
# INDEPENDENT ACCOUNTING EXPECTATIONS
# =========================================================


def test_baseline_quantity(baseline):

    assert baseline["quantity"] == D("0.156233")


def test_baseline_entry_price(baseline):

    assert baseline["entry_price"] == D(
        "100.22358025"
    )


def test_baseline_stop_price(baseline):

    assert baseline["stop_price"] == D(
        "97.21987655"
    )


def test_baseline_entry_cost(baseline):

    assert baseline["entry_cost"] == D(
        "15.666059728505"
    )


def test_baseline_planned_risk(baseline):

    assert baseline["planned_risk"] == D(
        "0.499882590991"
    )


def test_baseline_final_cash(baseline):

    assert baseline["final_cash"] == D(
        "99.500117409009"
    )


def test_baseline_final_equity(baseline):

    assert baseline["final_equity"] == D(
        "99.500117409009"
    )


def test_baseline_normal_stop_reconciliation(
    baseline,
):

    assert (
        baseline["actual_loss"]
        == baseline["planned_risk"]
    )

    assert baseline["risk_breached"] is False


# =========================================================
# CALLER CONTEXT PRESERVATION
# =========================================================


def test_preserve_existing_caller_flags(
    baseline,
):

    with localcontext() as caller:

        configure_hostile_context(caller)

        caller.flags[Inexact] = True

        caller.flags[Rounded] = True

        before = context_signature()

        try:

            assert run_scenario() == baseline

        finally:

            assert context_signature() == before


def test_global_context_unchanged(
    baseline,
):

    before = context_signature()

    try:

        assert run_scenario() == baseline

    finally:

        assert context_signature() == before


# =========================================================
# EXCEPTION SAFETY
# =========================================================


def test_invalid_market_price_restores_context():

    with localcontext() as caller:

        configure_hostile_context(caller)

        before = context_signature()

        try:

            manager = CapitalManager()

            manager.start_session(SESSION)

            opportunity = create_opportunity(
                manager
            )

            initial_state = manager.state

            with pytest.raises(ValueError):

                manager.calculate_allocation(
                    opportunity,
                    D("-100"),
                )

            assert manager.state == initial_state

        finally:

            assert context_signature() == before


def test_missing_session_restores_context():

    with localcontext() as caller:

        configure_hostile_context(caller)

        before = context_signature()

        try:

            manager = CapitalManager()

            opportunity = create_opportunity(
                manager
            )

            initial_state = manager.state

            with pytest.raises(RuntimeError):

                manager.calculate_allocation(
                    opportunity,
                    MARKET_PRICE,
                )

            assert manager.state == initial_state

        finally:

            assert context_signature() == before


def test_invalid_valuation_restores_context():

    with localcontext() as caller:

        configure_hostile_context(caller)

        before = context_signature()

        try:

            manager = CapitalManager()

            manager.start_session(SESSION)

            opportunity = create_opportunity(
                manager
            )

            result = manager.open_position(
                opportunity,
                MARKET_PRICE,
            )

            assert result.executed

            initial_state = manager.state

            with pytest.raises(ValueError):

                manager.mark_to_market(
                    {"AAPL": D("-1")}
                )

            assert manager.state == initial_state

        finally:

            assert context_signature() == before


# =========================================================
# REJECTED OPPORTUNITIES
# =========================================================


def test_rejected_opportunity_preserves_account():

    with localcontext() as caller:

        configure_hostile_context(caller)

        before = context_signature()

        try:

            manager = CapitalManager()

            manager.start_session(SESSION)

            opportunity = create_opportunity(
                manager,
                score=D("0"),
            )

            initial_state = manager.state

            result = manager.open_position(
                opportunity,
                MARKET_PRICE,
            )

            assert result.executed is False

            assert manager.state == initial_state

            assert len(manager.trades) == 0

            assert_accounting_identity(manager)

        finally:

            assert context_signature() == before


def test_duplicate_position_preserves_account():

    with localcontext() as caller:

        configure_hostile_context(caller)

        before = context_signature()

        try:

            manager = CapitalManager()

            manager.start_session(SESSION)

            opportunity = create_opportunity(
                manager
            )

            first = manager.open_position(
                opportunity,
                MARKET_PRICE,
            )

            assert first.executed

            initial_state = manager.state

            second = manager.open_position(
                opportunity,
                MARKET_PRICE,
            )

            assert second.executed is False

            assert manager.state == initial_state

            assert manager.state.open_positions == 1

            assert_accounting_identity(manager)

        finally:

            assert context_signature() == before


# =========================================================
# GENUINE GAP RISK
# =========================================================


@pytest.mark.parametrize(
    "gap_price",
    [
        D("96"),
        D("95"),
        D("90"),
        D("85"),
        D("80"),
    ],
)
def test_genuine_gap_breach_under_hostile_context(
    gap_price,
):

    with localcontext() as caller:

        configure_hostile_context(caller)

        before = context_signature()

        try:

            manager = CapitalManager()

            manager.start_session(SESSION)

            opportunity = create_opportunity(
                manager
            )

            entry = manager.open_position(
                opportunity,
                MARKET_PRICE,
            )

            assert entry.executed

            exit_result = manager.close_position(
                "AAPL",
                gap_price,
                "STOP_GAP",
            )

            assert exit_result.executed

            trade = exit_result.trade

            assert trade.risk_breached is True

            assert trade.actual_loss > trade.planned_risk

            assert any(
                event.event_type == "PLANNED_RISK_BREACH"
                for event in manager.risk_events
            )

            assert_accounting_identity(manager)

        finally:

            assert context_signature() == before


# =========================================================
# REPEATED TRANSACTIONS
# =========================================================


@pytest.mark.parametrize(
    "transaction_count",
    [10, 25, 50],
)
def test_repeated_profitable_transactions(
    transaction_count,
):

    with localcontext() as caller:

        configure_hostile_context(caller)

        before = context_signature()

        try:

            manager = CapitalManager()

            for index in range(transaction_count):

                session = (
                    SESSION
                    + timedelta(days=index)
                )

                manager.start_session(session)

                opportunity = create_opportunity(
                    manager,
                    session=session,
                )

                entry = manager.open_position(
                    opportunity,
                    MARKET_PRICE,
                )

                assert entry.executed, entry.reason

                assert_accounting_identity(manager)

                exit_result = manager.close_position(
                    "AAPL",
                    D("102"),
                    "TEST_EXIT",
                )

                assert exit_result.executed

                assert exit_result.trade.pnl > D("0")

                assert (
                    exit_result.trade.risk_breached
                    is False
                )

                assert_accounting_identity(manager)

                assert manager.state.open_positions == 0

            assert len(manager.trades) == (
                transaction_count
            )

            with accounting_context():

                expected_realized_pnl = sum(
                    (
                        trade.pnl
                        for trade in manager.trades
                    ),
                    D("0"),
                )

                expected_equity = (
                    manager.state.initial_capital
                    + expected_realized_pnl
                )

            assert (
                manager.state.realized_pnl
                == expected_realized_pnl
            )

            assert (
                manager.state.equity
                == expected_equity
            )

            assert (
                manager.state.cash
                == expected_equity
            )

            assert manager.state.trading_halted is False

        finally:

            assert context_signature() == before