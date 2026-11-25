from datetime import date, timedelta
from decimal import (
    Decimal,
    Context,
    localcontext,
    ROUND_HALF_EVEN,
)

from portfolio.capital_manager import CapitalManager
from portfolio.models import PortfolioOpportunity


D = Decimal

ZERO = D("0")
ONE = D("1")
HUNDRED = D("100")

PRICE_STEP = D("0.00000001")
MONEY_STEP = D("0.000000000001")

SESSION = date(2026, 11, 23)


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


# =========================================================
# DYNAMIC STOP RISK REFERENCE
# =========================================================


def calculate_reference_stop_risk(
    position,
    slippage_rate,
    fee_rate,
):

    with reference_context():

        # If the current market price is already
        # below the stop, do not assume recovery
        # to the original stop price.

        exit_reference = min(
            position.current_price,
            position.stop_price,
        )

        expected_exit_price = reference_price(
            exit_reference
            * (ONE - slippage_rate)
        )

        exit_notional = reference_money(
            position.quantity
            * expected_exit_price
        )

        exit_fee = reference_money(
            exit_notional
            * fee_rate
        )

        expected_exit_proceeds = reference_money(
            exit_notional - exit_fee
        )

        current_market_value = reference_money(
            position.quantity
            * position.current_price
        )

        dynamic_stop_risk = max(
            ZERO,
            reference_money(
                current_market_value
                - expected_exit_proceeds
            ),
        )

        return {
            "exit_reference": exit_reference,
            "expected_exit_price": (
                expected_exit_price
            ),
            "exit_notional": exit_notional,
            "exit_fee": exit_fee,
            "expected_exit_proceeds": (
                expected_exit_proceeds
            ),
            "current_market_value": (
                current_market_value
            ),
            "dynamic_stop_risk": (
                dynamic_stop_risk
            ),
        }


def create_opportunity(
    manager,
    symbol,
    market_price,
    atr,
):

    return PortfolioOpportunity(
        symbol=symbol,
        strategy="MOMENTUM",
        signal_date=(
            SESSION - timedelta(days=1)
        ),
        score=D("85"),
        reference_price=market_price,
        atr=atr,
        stop_atr_multiplier=(
            manager.stop_atr_multiplier
        ),
        risk_reward_ratio=(
            manager.risk_reward_ratio
        ),
    )


def create_single_position_manager():

    manager = CapitalManager()

    manager.start_session(SESSION)

    opportunity = create_opportunity(
        manager,
        "AAPL",
        D("100"),
        D("2"),
    )

    result = manager.open_position(
        opportunity,
        D("100"),
    )

    if not result.executed:

        raise RuntimeError(
            result.reason
        )

    return manager


def calculate_portfolio_reference_risk(
    manager,
):

    with reference_context():

        dynamic_risk = sum(
            (
                calculate_reference_stop_risk(
                    position,
                    manager.slippage,
                    manager.fee_rate,
                )["dynamic_stop_risk"]

                for position in manager.state.positions
            ),
            ZERO,
        )

        original_risk = sum(
            (
                position.planned_risk

                for position in manager.state.positions
            ),
            ZERO,
        )

        return (
            reference_money(original_risk),
            reference_money(dynamic_risk),
        )


def classify_risk_difference(
    original_risk,
    dynamic_risk,
):

    if dynamic_risk > original_risk:

        return "UNDER_RESERVED"

    if dynamic_risk < original_risk:

        return "OVER_RESERVED"

    return "EXACT"


# =========================================================
# SINGLE POSITION DIAGNOSTICS
# =========================================================


def run_single_position_scenario(
    scenario,
    market_price,
):

    manager = create_single_position_manager()

    initial_position = manager.state.positions[0]

    original_planned_risk = (
        initial_position.planned_risk
    )

    manager.mark_to_market(
        {
            "AAPL": market_price,
        }
    )

    position = manager.state.positions[0]

    reference = calculate_reference_stop_risk(
        position,
        manager.slippage,
        manager.fee_rate,
    )

    dynamic_risk = reference[
        "dynamic_stop_risk"
    ]

    with reference_context():

        difference = reference_money(
            dynamic_risk
            - original_planned_risk
        )

        dynamic_risk_percent = (
            dynamic_risk
            / manager.state.equity
            * HUNDRED
        )

        reference_remaining_capacity = max(
            ZERO,
            manager.maximum_aggregate_risk
            - dynamic_risk,
        )

    assert position.planned_risk == (
        original_planned_risk
    )

    assert manager.state.aggregate_open_risk == (
        original_planned_risk
    )

    return {
        "scenario": scenario,
        "symbol": position.symbol,
        "quantity": position.quantity,
        "entry_price": position.entry_price,
        "current_price": position.current_price,
        "stop_price": position.stop_price,
        "expected_exit_price": (
            reference["expected_exit_price"]
        ),
        "current_market_value": (
            position.market_value
        ),
        "expected_exit_proceeds": (
            reference["expected_exit_proceeds"]
        ),
        "original_planned_risk": (
            original_planned_risk
        ),
        "current_reserved_risk": (
            manager.state.aggregate_open_risk
        ),
        "reference_dynamic_risk": dynamic_risk,
        "risk_difference": difference,
        "dynamic_risk_percent": (
            dynamic_risk_percent
        ),
        "maximum_aggregate_risk": (
            manager.maximum_aggregate_risk
        ),
        "current_remaining_capacity": (
            manager.remaining_open_risk
        ),
        "reference_remaining_capacity": (
            reference_remaining_capacity
        ),
        "portfolio_equity": (
            manager.state.equity
        ),
        "daily_entry_halted": (
            manager.state.daily_entry_halted
        ),
        "trading_halted": (
            manager.state.trading_halted
        ),
        "classification": (
            classify_risk_difference(
                original_planned_risk,
                dynamic_risk,
            )
        ),
    }


# =========================================================
# MULTIPLE POSITION DIAGNOSTICS
# =========================================================


def run_multiple_position_scenario():

    manager = CapitalManager()

    manager.start_session(SESSION)

    opportunities = [
        ("AAPL", D("100"), D("2")),
        ("MSFT", D("250"), D("5")),
        ("NVDA", D("150"), D("3")),
    ]

    entry_results = []

    for symbol, price, atr in opportunities:

        opportunity = create_opportunity(
            manager,
            symbol,
            price,
            atr,
        )

        result = manager.open_position(
            opportunity,
            price,
        )

        entry_results.append(
            {
                "symbol": symbol,
                "executed": result.executed,
                "reason": result.reason,
            }
        )

    if manager.state.open_positions < 2:

        raise RuntimeError(
            "At least two open positions required."
        )

    updated_prices = {}

    with reference_context():

        for position in manager.state.positions:

            updated_prices[position.symbol] = (
                reference_price(
                    position.current_price
                    * D("1.05")
                )
            )

    manager.mark_to_market(
        updated_prices
    )

    original_risk, dynamic_risk = (
        calculate_portfolio_reference_risk(
            manager
        )
    )

    with reference_context():

        difference = reference_money(
            dynamic_risk - original_risk
        )

        reference_remaining_capacity = max(
            ZERO,
            manager.maximum_aggregate_risk
            - dynamic_risk,
        )

    candidate = create_opportunity(
        manager,
        "AMD",
        D("100"),
        D("2"),
    )

    candidate_allocation = (
        manager.calculate_allocation(
            candidate,
            D("100"),
        )
    )

    return {
        "entry_results": entry_results,
        "open_positions": (
            manager.state.open_positions
        ),
        "symbols": manager.state.existing_symbols,
        "cash": manager.state.cash,
        "equity": manager.state.equity,
        "market_value": manager.state.market_value,
        "original_planned_risk": original_risk,
        "current_reserved_risk": (
            manager.state.aggregate_open_risk
        ),
        "reference_dynamic_risk": dynamic_risk,
        "risk_difference": difference,
        "maximum_aggregate_risk": (
            manager.maximum_aggregate_risk
        ),
        "current_remaining_capacity": (
            manager.remaining_open_risk
        ),
        "reference_remaining_capacity": (
            reference_remaining_capacity
        ),
        "new_candidate_approved": (
            candidate_allocation.approved
        ),
        "new_candidate_reason": (
            candidate_allocation.reason
        ),
        "classification": (
            classify_risk_difference(
                original_risk,
                dynamic_risk,
            )
        ),
    }


# =========================================================
# REPORTING
# =========================================================


def print_result(result):

    for name, value in result.items():

        if name == "entry_results":

            print("Entry results:")

            for item in value:

                print(
                    f"  {item['symbol']}: "
                    f"{item['executed']} | "
                    f"{item['reason']}"
                )

        else:

            print(
                f"{name:35}: {value}"
            )


def main():

    scenarios = [
        ("ENTRY_REFERENCE", D("100")),
        ("SMALL_GAIN", D("101")),
        ("MODERATE_GAIN", D("103")),
        ("LARGE_GAIN", D("105")),
        ("STRONG_GAIN", D("110")),
        ("SMALL_LOSS", D("99")),
        ("NEAR_STOP", D("97.5")),
        ("AT_STOP", D("97.1")),
        ("BELOW_STOP", D("95")),
        ("SEVERE_GAP", D("90")),
    ]

    counts = {
        "UNDER_RESERVED": 0,
        "OVER_RESERVED": 0,
        "EXACT": 0,
    }

    print("=" * 75)
    print("JARVIS DYNAMIC OPEN-POSITION RISK DIAGNOSTIC")
    print("VERSION 0.8.6.7.2D.1")
    print("=" * 75)

    for name, price in scenarios:

        result = run_single_position_scenario(
            name,
            price,
        )

        counts[
            result["classification"]
        ] += 1

        print()
        print("-" * 75)
        print(name)
        print("-" * 75)

        print_result(result)

    print()
    print("=" * 75)
    print("MULTIPLE POSITION RISK DIAGNOSTIC")
    print("=" * 75)

    multiple = run_multiple_position_scenario()

    print_result(multiple)

    print()
    print("=" * 75)
    print("FINAL DYNAMIC RISK DIAGNOSTIC")
    print("=" * 75)

    print(
        "Single-position scenarios:",
        len(scenarios),
    )

    print(
        "Under-reserved scenarios:",
        counts["UNDER_RESERVED"],
    )

    print(
        "Over-reserved scenarios:",
        counts["OVER_RESERVED"],
    )

    print(
        "Exact scenarios:",
        counts["EXACT"],
    )

    print(
        "Multiple-position classification:",
        multiple["classification"],
    )

    print(
        "Original planned risk preserved: YES"
    )

    print(
        "Production risk logic modified: NO"
    )

    print(
        "DYNAMIC RISK DIAGNOSTIC: COMPLETE"
    )

    print("Historical simulation only")
    print("Broker orders submitted: 0")
    print("Live trading: DISABLED")


if __name__ == "__main__":

    main()