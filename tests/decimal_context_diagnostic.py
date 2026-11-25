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
)

from portfolio.capital_manager import CapitalManager
from portfolio.models import PortfolioOpportunity


D = Decimal

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


def run_scenario():

    manager = CapitalManager()

    manager.start_session(date(2026, 11, 23))

    market_price = D("100.12345678912345")

    opportunity = PortfolioOpportunity(
        symbol="AAPL",
        strategy="MOMENTUM",
        signal_date=manager.session - timedelta(days=1),
        score=D("85"),
        reference_price=market_price,
        atr=D("2.002469135782469"),
        stop_atr_multiplier=manager.stop_atr_multiplier,
        risk_reward_ratio=manager.risk_reward_ratio,
    )

    allocation = manager.calculate_allocation(
        opportunity,
        market_price,
    )

    if not allocation.approved:
        raise RuntimeError(allocation.reason)

    entry = manager.open_position(
        opportunity,
        market_price,
    )

    if not entry.executed:
        raise RuntimeError(entry.reason)

    position = entry.position

    entry_cash = manager.state.cash

    entry_equity = manager.state.equity

    manager.mark_to_market(
        {
            position.symbol: D("101.98765432198765"),
        }
    )

    marked_equity = manager.state.equity

    marked_market_value = manager.state.market_value

    marked_unrealized_pnl = manager.state.unrealized_pnl

    result = manager.close_position(
        position.symbol,
        position.stop_price,
        "STOP_LOSS",
    )

    if not result.executed:
        raise RuntimeError(result.reason)

    trade = result.trade

    return {
        "quantity": allocation.quantity,
        "entry_price": allocation.entry_price,
        "stop_price": allocation.stop_price,
        "target_price": allocation.target_price,
        "entry_fee": allocation.entry_fee,
        "entry_cost": allocation.entry_cost,
        "planned_risk": allocation.planned_risk,
        "entry_cash": entry_cash,
        "entry_equity": entry_equity,
        "marked_equity": marked_equity,
        "marked_market_value": marked_market_value,
        "marked_unrealized_pnl": marked_unrealized_pnl,
        "exit_price": trade.exit_price,
        "exit_fee": trade.exit_fee,
        "exit_proceeds": trade.exit_proceeds,
        "actual_loss": trade.actual_loss,
        "trade_pnl": trade.pnl,
        "risk_breached": trade.risk_breached,
        "final_cash": manager.state.cash,
        "final_equity": manager.state.equity,
        "realized_pnl": manager.state.realized_pnl,
        "peak_equity": manager.state.peak_equity,
        "drawdown_percent": manager.state.drawdown_percent,
        "daily_loss_percent": manager.state.daily_loss_percent,
        "trading_halted": manager.state.trading_halted,
        "daily_entry_halted": manager.state.daily_entry_halted,
    }


def context_signature():

    context = getcontext()

    return (
        context.prec,
        context.rounding,
        context.Emin,
        context.Emax,
        context.capitals,
        context.clamp,
        dict(context.traps),
        dict(context.flags),
    )


def main():

    print("=" * 75)
    print("JARVIS DECIMAL CONTEXT ISOLATION DIAGNOSTIC")
    print("VERSION 0.8.6.7.2C.4D")
    print("=" * 75)

    original_context = context_signature()

    with localcontext() as context:

        context.prec = 64

        context.rounding = ROUND_HALF_EVEN

        baseline = run_scenario()

    print()
    print("BASELINE ACCOUNTING")
    print("-" * 75)

    for name, value in baseline.items():

        print(f"{name:28}: {value}")

    attempted = 0

    exact_matches = 0

    mismatches = 0

    errors = 0

    context_changes = 0

    field_differences = {}

    print()
    print("CONTEXT TEST MATRIX")
    print("-" * 75)

    for precision in PRECISIONS:

        for rounding in ROUNDING_MODES:

            attempted += 1

            try:

                with localcontext() as context:

                    context.prec = precision

                    context.rounding = rounding

                    before = context_signature()

                    result = run_scenario()

                    after = context_signature()

                    if before != after:

                        context_changes += 1

                differences = [
                    name
                    for name in baseline
                    if result[name] != baseline[name]
                ]

                if differences:

                    mismatches += 1

                    print(
                        f"PREC={precision:2} "
                        f"{rounding:18} "
                        f"MISMATCH ({len(differences)})"
                    )

                    for name in differences:

                        field_differences[name] = (
                            field_differences.get(name, 0) + 1
                        )

                        print(
                            f"    {name}: "
                            f"{result[name]} "
                            f"| Expected: {baseline[name]}"
                        )

                else:

                    exact_matches += 1

                    print(
                        f"PREC={precision:2} "
                        f"{rounding:18} PASS"
                    )

            except Exception as error:

                errors += 1

                print(
                    f"PREC={precision:2} "
                    f"{rounding:18} ERROR: "
                    f"{type(error).__name__}: {error}"
                )

    global_context_unchanged = (
        original_context == context_signature()
    )

    print()
    print("=" * 75)
    print("CONTEXT ISOLATION RESULTS")
    print("=" * 75)

    print("Attempted scenarios:", attempted)
    print("Exact matches:", exact_matches)
    print("Mismatched scenarios:", mismatches)
    print("Execution errors:", errors)
    print("Caller context changes:", context_changes)
    print("Global context unchanged:", global_context_unchanged)

    print()
    print("AFFECTED ACCOUNTING FIELDS")
    print("-" * 75)

    if field_differences:

        for name, count in sorted(
            field_differences.items(),
            key=lambda item: (-item[1], item[0]),
        ):

            print(f"{name:28}: {count}")

    else:

        print("No mismatched fields detected.")

    print()
    print("=" * 75)
    print("FINAL DIAGNOSTIC STATUS")
    print("=" * 75)

    passed = (
        exact_matches == attempted
        and mismatches == 0
        and errors == 0
        and context_changes == 0
        and global_context_unchanged
    )

    if passed:

        print("CONTEXT ISOLATION: PASS")

    else:

        print("CONTEXT ISOLATION: IMPROVEMENT REQUIRED")

    print("Historical simulation only")
    print("Production code modified: NO")
    print("Broker orders submitted: 0")
    print("Live trading: DISABLED")


if __name__ == "__main__":

    main()