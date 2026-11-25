from datetime import date
from decimal import Decimal as D
from decimal import getcontext
from itertools import product

from portfolio.capital_manager import CapitalManager
from portfolio.models import PortfolioOpportunity


SESSION = date(2026, 11, 23)
SIGNAL_DATE = date(2026, 11, 20)

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


def make_opportunity(price, atr):

    return PortfolioOpportunity(
        symbol="AAPL",
        strategy="MOMENTUM",
        signal_date=SIGNAL_DATE,
        score=90,
        reference_price=price,
        atr=atr,
        stop_atr_multiplier="1.5",
        risk_reward_ratio="2",
    )


def verify_accounting(capital):

    state = capital.state

    assert state.equity == (
        state.cash + state.market_value
    ), "Cash and market value mismatch."

    assert state.equity == (
        state.initial_capital
        + state.realized_pnl
        + state.unrealized_pnl
    ), "Portfolio accounting mismatch."

    assert state.cash >= D("0")


def run_precision_matrix():

    results = {
        "attempted": 0,
        "executed": 0,
        "rejected": 0,
        "exact_matches": 0,
        "positive_differences": 0,
        "negative_differences": 0,
        "false_breaches": 0,
        "errors": 0,
    }

    largest_difference = D("0")

    print("=" * 70)
    print("PRECISION TEST MATRIX")
    print("=" * 70)

    for price_text, atr_percent in product(
        PRICES,
        ATR_PERCENTAGES,
    ):

        results["attempted"] += 1

        price = D(price_text)

        atr = (
            price * D(atr_percent) / D("100")
        )

        try:

            capital = CapitalManager()

            capital.start_session(SESSION)

            opportunity = make_opportunity(
                price,
                atr,
            )

            entry = capital.open_position(
                opportunity,
                price,
            )

            if not entry.executed:

                results["rejected"] += 1

                print(
                    "REJECTED:",
                    price,
                    "| ATR %:",
                    atr_percent,
                    "|",
                    entry.reason,
                )

                continue

            results["executed"] += 1

            verify_accounting(capital)

            position = entry.position

            exit_result = capital.close_position(
                "AAPL",
                position.stop_price,
                "STOP_LOSS",
            )

            assert exit_result.executed

            trade = exit_result.trade

            verify_accounting(capital)

            difference = (
                trade.actual_loss
                - trade.planned_risk
            )

            largest_difference = max(
                largest_difference,
                abs(difference),
            )

            if difference == D("0"):

                results["exact_matches"] += 1

            elif difference > D("0"):

                results[
                    "positive_differences"
                ] += 1

            else:

                results[
                    "negative_differences"
                ] += 1

            if trade.risk_breached:

                results["false_breaches"] += 1

            if difference != D("0"):

                print(
                    "PRECISION DIFFERENCE:",
                    "| Price:", price,
                    "| ATR %:", atr_percent,
                    "| Difference:", difference,
                    "| Risk breach:",
                    trade.risk_breached,
                )

        except Exception as error:

            results["errors"] += 1

            print(
                "ERROR:",
                "| Price:", price,
                "| ATR %:", atr_percent,
                "| Type:", type(error).__name__,
                "| Message:", str(error),
            )

    print()
    print("=" * 70)
    print("PRECISION MATRIX RESULTS")
    print("=" * 70)

    for name, value in results.items():

        print(
            name.replace("_", " ").title(),
            ":",
            value,
        )

    print(
        "Largest absolute difference:",
        largest_difference,
    )

    return results


def run_gap_loss_test():

    print()
    print("=" * 70)
    print("GENUINE RISK BREACH VERIFICATION")
    print("=" * 70)

    capital = CapitalManager()

    capital.start_session(SESSION)

    entry = capital.open_position(
        make_opportunity(
            D("100"),
            D("2"),
        ),
        D("100"),
    )

    assert entry.executed

    position = entry.position

    gap_price = (
        position.stop_price * D("0.90")
    )

    result = capital.close_position(
        "AAPL",
        gap_price,
        "STOP_GAP",
    )

    assert result.executed

    trade = result.trade

    print("Planned risk:", trade.planned_risk)
    print("Actual loss:", trade.actual_loss)

    print(
        "Excess loss:",
        trade.actual_loss - trade.planned_risk,
    )

    print("Risk breach:", trade.risk_breached)

    assert trade.actual_loss > trade.planned_risk
    assert trade.risk_breached

    verify_accounting(capital)

    print("Genuine risk breach detection: PASS")


def main():

    print("=" * 70)
    print("JARVIS DECIMAL PRECISION DIAGNOSTICS")
    print("VERSION 0.8.6.7.2C")
    print("=" * 70)

    print(
        "Decimal context precision:",
        getcontext().prec,
    )

    print(
        "Decimal rounding mode:",
        getcontext().rounding,
    )

    results = run_precision_matrix()

    run_gap_loss_test()

    print()
    print("=" * 70)
    print("FINAL DIAGNOSTIC STATUS")
    print("=" * 70)

    if (
        results["errors"] > 0
        or results["false_breaches"] > 0
        or results["positive_differences"] > 0
        or results["negative_differences"] > 0
        or results["executed"] == 0
    ):

        print("PRECISION HARDENING REQUIRED")

    else:

        print("PRECISION MATRIX PASSED")

    print("Historical simulation only")
    print("Broker orders submitted: 0")
    print("Live trading: DISABLED")


if __name__ == "__main__":

    main()