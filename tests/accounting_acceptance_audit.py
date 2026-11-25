import ast

from datetime import date, timedelta
from decimal import (
    Decimal,
    getcontext,
    localcontext,
    ROUND_DOWN,
    Inexact,
    Rounded,
)
from pathlib import Path
from time import perf_counter

from portfolio.accounting_context import (
    create_accounting_context,
)

from portfolio.decimal_utils import (
    normalize_price,
    normalize_quantity,
    normalize_money,
    calculate_notional,
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

from tests.test_transaction_stress import (
    run_transaction_sequence,
)


D = Decimal

ROOT = Path(__file__).resolve().parents[1]

SESSION = date(2026, 11, 23)


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


def count_protected_functions(path):

    tree = ast.parse(
        path.read_text(encoding="utf-8")
    )

    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(decorator, ast.Name)
            and decorator.id == "isolated_accounting"
            for decorator in node.decorator_list
        )
    )


def audit_context_configuration():

    context = create_accounting_context()

    assert context.prec == 64

    assert context.rounding == "ROUND_HALF_EVEN"

    assert context.Emin == -999999

    assert context.Emax == 999999

    assert not context.traps[Inexact]

    assert not context.traps[Rounded]

    return "64-digit isolated accounting context"


def audit_protected_functions():

    expected = {
        "portfolio/decimal_utils.py": 14,
        "portfolio/models.py": 52,
        "portfolio/capital_manager.py": 25,
    }

    total = 0

    for filename, expected_count in expected.items():

        actual_count = count_protected_functions(
            ROOT / filename
        )

        assert actual_count == expected_count, (
            f"{filename}: "
            f"expected {expected_count}, "
            f"found {actual_count}"
        )

        total += actual_count

    assert total == 91

    return f"{total} protected functions"


def audit_price_precision():

    assert normalize_price(
        D("100.123456789")
    ) == D("100.12345679")

    assert normalize_price(
        D("1.000000005")
    ) == D("1.00000000")

    assert normalize_price(
        D("1.000000015")
    ) == D("1.00000002")

    return "8-decimal price precision"


def audit_quantity_precision():

    assert normalize_quantity(
        D("0.123456789")
    ) == D("0.123456")

    assert normalize_quantity(
        D("0.0000009")
    ) == D("0.000000")

    assert normalize_quantity(
        D("1.9999999")
    ) == D("1.999999")

    return "6-decimal quantity precision"


def audit_money_precision():

    assert normalize_money(
        D("1.1234567890129")
    ) == D("1.123456789013")

    assert normalize_money(
        D("-1.1234567890129")
    ) == D("-1.123456789013")

    assert normalize_money(
        D("0.0000000000005")
    ) == D("0.000000000000")

    assert normalize_money(
        D("0.0000000000015")
    ) == D("0.000000000002")

    return "12-decimal monetary precision"


def audit_invalid_numeric_inputs():

    invalid_values = [
        D("NaN"),
        D("Infinity"),
        D("-Infinity"),
        True,
        None,
        "invalid",
    ]

    rejected = 0

    for value in invalid_values:

        try:

            normalize_money(value)

        except (ValueError, TypeError):

            rejected += 1

        else:

            raise AssertionError(
                f"Invalid value accepted: {value!r}"
            )

    assert rejected == len(invalid_values)

    return f"{rejected} invalid inputs rejected"


def audit_small_price_boundary():

    assert normalize_price(
        D("0.00000001")
    ) == D("0.00000001")

    try:

        normalize_price(
            D("0.000000001")
        )

    except ValueError:

        pass

    else:

        raise AssertionError(
            "Price rounded to zero was accepted."
        )

    return "Zero-price rounding rejected"


def audit_notional_precision():

    result = calculate_notional(
        D("0.123456"),
        D("100.12345678"),
    )

    assert result == D("12.360841480232"), (
        f"Expected notional: 12.360841480232, "
        f"Actual notional: {result}"
    )

    return f"Exact reference notional: {result}"


def audit_normal_stop():

    result = run_scenario()

    assert result["planned_risk"] == D(
        "0.499882590991"
    )

    assert result["actual_loss"] == (
        result["planned_risk"]
    )

    assert result["risk_breached"] is False

    assert result["final_cash"] == D(
        "99.500117409009"
    )

    assert result["final_equity"] == (
        result["final_cash"]
    )

    return "Normal stop reconciles exactly"


def audit_hostile_context():

    baseline = run_scenario()

    with localcontext() as caller:

        caller.prec = 10

        caller.rounding = ROUND_DOWN

        caller.Emin = -2

        caller.Emax = 2

        caller.traps[Inexact] = True

        caller.traps[Rounded] = True

        before = context_signature()

        try:

            result = run_scenario()

            assert result == baseline

        finally:

            assert context_signature() == before

    return "Hostile context produces identical results"


def audit_repeated_transactions():

    result = run_transaction_sequence(1000)

    assert result["transactions"] == 1000

    assert result["entries"] == 1000

    assert result["exits"] == 1000

    assert result["winning_trades"] == 750

    assert result["losing_trades"] == 250

    assert result["final_equity"] == D(
        "275.684785502563"
    )

    assert result["total_fees"] == D(
        "15.757514863899"
    )

    assert result["risk_events"] == 0

    assert result["open_positions"] == 0

    return "1,000 transactions reconciled exactly"


def audit_genuine_gap_breach():

    manager = CapitalManager()

    manager.start_session(SESSION)

    opportunity = PortfolioOpportunity(
        symbol="AAPL",
        strategy="MOMENTUM",
        signal_date=SESSION - timedelta(days=1),
        score=D("85"),
        reference_price=D("100"),
        atr=D("2"),
        stop_atr_multiplier=(
            manager.stop_atr_multiplier
        ),
        risk_reward_ratio=(
            manager.risk_reward_ratio
        ),
    )

    entry = manager.open_position(
        opportunity,
        D("100"),
    )

    assert entry.executed, entry.reason

    result = manager.close_position(
        "AAPL",
        D("90"),
        "STOP_GAP",
    )

    assert result.executed

    assert result.trade.risk_breached is True

    assert (
        result.trade.actual_loss
        > result.trade.planned_risk
    )

    assert any(
        event.event_type == "PLANNED_RISK_BREACH"
        for event in manager.risk_events
    )

    assert manager.state.open_positions == 0

    return "Genuine gap-loss breach detected"


def audit_safety_configuration():

    manager = CapitalManager()

    config = manager._config

    config.validate()

    safety = config.get_section("safety")

    account = config.get_section("account")

    strategy = config.get_section("strategy")

    assert safety["historical_simulation_only"]

    assert not safety["live_trading_enabled"]

    assert not safety["broker_order_submission_enabled"]

    assert not safety["automatic_strategy_activation"]

    assert safety["require_forward_validation"]

    assert safety["fail_on_accounting_mismatch"]

    assert not account["margin_enabled"]

    assert not account["short_selling_enabled"]

    assert strategy["research_only"]

    return "Historical research safety restrictions active"


def audit_initial_account():

    manager = CapitalManager()

    state = manager.state

    assert state.initial_capital == D("100")

    assert state.cash == D("100")

    assert state.equity == D("100")

    assert state.realized_pnl == D("0")

    assert state.unrealized_pnl == D("0")

    assert state.open_positions == 0

    assert state.aggregate_open_risk == D("0")

    assert len(manager.trades) == 0

    assert len(manager.risk_events) == 0

    return "Initial account reconciles exactly"


def main():

    checks = [
        ("Context configuration", audit_context_configuration),
        ("Protected functions", audit_protected_functions),
        ("Price precision", audit_price_precision),
        ("Quantity precision", audit_quantity_precision),
        ("Money precision", audit_money_precision),
        ("Invalid numeric inputs", audit_invalid_numeric_inputs),
        ("Small price boundary", audit_small_price_boundary),
        ("Notional precision", audit_notional_precision),
        ("Normal stop", audit_normal_stop),
        ("Hostile context", audit_hostile_context),
        ("Repeated transactions", audit_repeated_transactions),
        ("Genuine gap breach", audit_genuine_gap_breach),
        ("Safety configuration", audit_safety_configuration),
        ("Initial account", audit_initial_account),
    ]

    started = perf_counter()

    initial_context = context_signature()

    passed = 0

    failed = 0

    print("=" * 75)
    print("JARVIS FINAL ACCOUNTING ACCEPTANCE AUDIT")
    print("VERSION 0.8.6.7.2C.6")
    print("=" * 75)

    for name, function in checks:

        before = context_signature()

        try:

            try:

                result = function()

            finally:

                assert context_signature() == before, (
                    "Caller Decimal context modified."
                )

            passed += 1

            print(f"[PASS] {name}")
            print(f"       {result}")

        except Exception as error:

            failed += 1

            print(f"[FAIL] {name}")
            print(
                f"       {type(error).__name__}: {error}"
            )

    unchanged = (
        context_signature() == initial_context
    )

    elapsed = perf_counter() - started

    print()
    print("=" * 75)
    print("FINAL ACCOUNTING ACCEPTANCE RESULTS")
    print("=" * 75)

    print("Total checks:", len(checks))
    print("Passed:", passed)
    print("Failed:", failed)
    print("Global context unchanged:", unchanged)
    print(f"Duration: {elapsed:.4f} seconds")

    success = (
        failed == 0
        and passed == len(checks)
        and unchanged
    )

    print()

    if success:

        print("ACCOUNTING ACCEPTANCE: PASS")

    else:

        print("ACCOUNTING ACCEPTANCE: FAIL")

    print("Historical simulation only")
    print("Broker orders submitted: 0")
    print("Live trading: DISABLED")

    return 0 if success else 1


if __name__ == "__main__":

    raise SystemExit(main())