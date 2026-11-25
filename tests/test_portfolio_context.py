from datetime import date
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

from portfolio.models import (
    PortfolioPosition,
    PortfolioTrade,
    PortfolioAccountState,
    PortfolioOpportunity,
    PortfolioSnapshot,
)

from portfolio.decimal_utils import (
    calculate_buy_price,
    calculate_sell_price,
    calculate_notional,
    calculate_fee,
    calculate_planned_risk,
    calculate_entry_cost,
    calculate_exit_proceeds,
    normalize_money,
)


D = Decimal

SESSION = date(2026, 11, 23)

PRECISIONS = [
    10, 16, 20, 28, 32, 50, 64
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


@pytest.fixture
def accounting_inputs():

    with accounting_context():

        quantity = D("0.156233")

        market_price = D(
            "100.12345678912345"
        )

        slippage = D("0.001")

        fee_rate = D("0.0005")

        entry_price = calculate_buy_price(
            market_price,
            slippage,
        )

        stop_price = D("97.21987655")

        target_price = D("106.23098766")

        entry_notional = calculate_notional(
            quantity,
            entry_price,
        )

        entry_fee = calculate_fee(
            entry_notional,
            fee_rate,
        )

        entry_cost = calculate_entry_cost(
            quantity,
            entry_price,
            fee_rate,
        )

        planned_risk = calculate_planned_risk(
            quantity,
            entry_price,
            stop_price,
            slippage,
            fee_rate,
        )

        exit_price = calculate_sell_price(
            stop_price,
            slippage,
        )

        exit_notional = calculate_notional(
            quantity,
            exit_price,
        )

        exit_fee = calculate_fee(
            exit_notional,
            fee_rate,
        )

        exit_proceeds = calculate_exit_proceeds(
            quantity,
            exit_price,
            fee_rate,
        )

        cash = normalize_money(
            D("100") - entry_cost
        )

        final_cash = normalize_money(
            cash + exit_proceeds
        )

        realized_pnl = normalize_money(
            exit_proceeds - entry_cost
        )

        return {
            "quantity": quantity,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": target_price,
            "entry_fee": entry_fee,
            "entry_cost": entry_cost,
            "planned_risk": planned_risk,
            "exit_price": exit_price,
            "exit_fee": exit_fee,
            "cash": cash,
            "final_cash": final_cash,
            "realized_pnl": realized_pnl,
        }


def run_model_scenario(inputs):

    opportunity = PortfolioOpportunity(
        symbol="AAPL",
        strategy="MOMENTUM",
        signal_date=date(2026, 11, 20),
        score=D("85"),
        reference_price=D(
            "100.12345678912345"
        ),
        atr=D("2.002469135782469"),
        stop_atr_multiplier=D("1.5"),
        risk_reward_ratio=D("2"),
    )

    stop, target = opportunity.prices_for_entry(
        inputs["entry_price"]
    )

    position = PortfolioPosition(
        symbol="AAPL",
        strategy="MOMENTUM",
        entry_date=SESSION,
        quantity=inputs["quantity"],
        entry_price=inputs["entry_price"],
        stop_price=inputs["stop_price"],
        target_price=inputs["target_price"],
        current_price=D(
            "101.98765432198765"
        ),
        planned_risk=inputs["planned_risk"],
        entry_fee=inputs["entry_fee"],
    )

    account = PortfolioAccountState(
        initial_capital=D("100"),
        cash=inputs["cash"],
        day_start_equity=D("100"),
        peak_equity=D("101"),
        positions=(position,),
    )

    trade = PortfolioTrade(
        symbol=position.symbol,
        strategy=position.strategy,
        entry_date=SESSION,
        exit_date=SESSION,
        quantity=position.quantity,
        entry_price=position.entry_price,
        exit_price=inputs["exit_price"],
        stop_price=position.stop_price,
        target_price=position.target_price,
        planned_risk=position.planned_risk,
        exit_reason="STOP_LOSS",
        entry_fee=position.entry_fee,
        exit_fee=inputs["exit_fee"],
    )

    final_account = PortfolioAccountState(
        initial_capital=D("100"),
        cash=inputs["final_cash"],
        day_start_equity=D("100"),
        peak_equity=D("101"),
        realized_pnl=inputs["realized_pnl"],
    )

    snapshot = PortfolioSnapshot(
        session=SESSION,
        account=final_account,
        entries=1,
        exits=1,
    )

    return {
        "stop_distance": opportunity.stop_distance,
        "calculated_stop": stop,
        "calculated_target": target,
        "entry_notional": position.entry_notional,
        "entry_cost": position.entry_cost,
        "market_value": position.market_value,
        "unrealized_pnl": position.unrealized_pnl,
        "unrealized_return": (
            position.unrealized_return_percent
        ),
        "risk_reward": position.risk_reward_ratio,
        "account_equity": account.equity,
        "account_exposure": account.total_exposure,
        "exposure_percent": account.exposure_percent,
        "aggregate_risk": account.aggregate_open_risk,
        "account_drawdown": account.drawdown_percent,
        "account_return": account.return_percent,
        "trade_exit_notional": trade.exit_notional,
        "trade_exit_proceeds": trade.exit_proceeds,
        "trade_fees": trade.total_fees,
        "trade_pnl": trade.pnl,
        "trade_return": trade.return_percent,
        "actual_loss": trade.actual_loss,
        "risk_multiple": trade.risk_multiple,
        "risk_breached": trade.risk_breached,
        "final_equity": final_account.equity,
        "final_cash": final_account.cash,
        "final_realized_pnl": final_account.realized_pnl,
        "final_daily_loss": final_account.daily_loss,
        "final_daily_loss_percent": (
            final_account.daily_loss_percent
        ),
        "final_drawdown": final_account.drawdown_percent,
        "snapshot": snapshot.to_record(),
    }


@pytest.fixture
def baseline(accounting_inputs):

    with accounting_context():

        return run_model_scenario(
            accounting_inputs
        )


@pytest.mark.parametrize(
    "precision",
    PRECISIONS,
)
@pytest.mark.parametrize(
    "rounding",
    ROUNDING_MODES,
)
def test_model_context_matrix(
    accounting_inputs,
    baseline,
    precision,
    rounding,
):

    with localcontext() as caller:

        caller.prec = precision

        caller.rounding = rounding

        caller.Emin = -2

        caller.Emax = 2

        caller.traps[Inexact] = True

        caller.traps[Rounded] = True

        before = context_signature()

        result = run_model_scenario(
            accounting_inputs
        )

        after = context_signature()

        assert result == baseline

        assert before == after


def test_model_normal_stop_reconciliation(
    accounting_inputs,
    baseline,
):

    assert (
        baseline["actual_loss"]
        == accounting_inputs["planned_risk"]
    )

    assert baseline["risk_breached"] is False

    assert baseline["risk_multiple"] == D("-1")


def test_model_final_accounting(
    accounting_inputs,
    baseline,
):

    with accounting_context():

        expected_equity = (
            D("100")
            + accounting_inputs["realized_pnl"]
        )

    assert (
        baseline["final_equity"]
        == expected_equity
    )

    assert (
        baseline["final_equity"]
        == baseline["final_cash"]
    )


def test_model_context_preserves_existing_flags(
    accounting_inputs,
    baseline,
):

    with localcontext() as caller:

        caller.flags[Inexact] = True

        caller.flags[Rounded] = True

        before = context_signature()

        result = run_model_scenario(
            accounting_inputs
        )

        assert result == baseline

        assert context_signature() == before


def test_model_context_restored_after_exception():

    with localcontext() as caller:

        caller.prec = 10

        caller.rounding = ROUND_DOWN

        before = context_signature()

        with pytest.raises(ValueError):

            PortfolioOpportunity(
                symbol="AAPL",
                strategy="MOMENTUM",
                signal_date=date(2026, 11, 20),
                score=D("85"),
                reference_price=D("100"),
                atr=D("-1"),
                stop_atr_multiplier=D("1.5"),
                risk_reward_ratio=D("2"),
            )

        assert context_signature() == before


def test_model_repeated_context_isolation(
    accounting_inputs,
    baseline,
):

    before = context_signature()

    for _ in range(100):

        result = run_model_scenario(
            accounting_inputs
        )

        assert result == baseline

    assert context_signature() == before


def test_model_snapshot_consistency(
    baseline,
):

    snapshot = baseline["snapshot"]

    assert D(snapshot["equity"]) == (
        baseline["final_equity"]
    )

    assert D(snapshot["cash"]) == (
        baseline["final_cash"]
    )

    assert D(snapshot["realized_pnl"]) == (
        baseline["final_realized_pnl"]
    )