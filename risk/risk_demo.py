from dataclasses import replace
from decimal import Decimal

from risk.risk_guardian import AccountState, RiskGuardian, TradeRequest

D = Decimal


def display_decision(title, decision):

    print(f"\n{title}")
    print("-" * 60)

    print(f"Symbol         : {decision.symbol}")

    print(f"Decision       : {'APPROVED' if decision.approved else 'REJECTED'}")

    print(f"Reason         : {decision.reason}")

    if decision.approved:
        print(f"Quantity       : {decision.quantity:.6f}")

        print(f"Position Value : ${decision.position_value:.2f}")

        print(f"Estimated Risk : ${decision.estimated_risk:.2f}")


if __name__ == "__main__":
    print("=" * 60)
    print("              JARVIS RISK GUARDIAN")
    print("                   VERSION 0.6")
    print("=" * 60)

    print("Environment: SIMULATION")
    print("Order Execution: DISABLED")

    guardian = RiskGuardian()

    account = AccountState(
        equity=D("100"),
        cash=D("100"),
        day_start_equity=D("100"),
        peak_equity=D("100"),
        open_positions=0,
        total_exposure=D("0"),
    )

    trade = TradeRequest(symbol="TEST", entry_price=D("100"), stop_price=D("95"))

    display_decision("SCENARIO 1: VALID TRADE", guardian.evaluate(account, trade))

    display_decision(
        "SCENARIO 2: DAILY LOSS LIMIT",
        guardian.evaluate(replace(account, equity=D("99")), trade),
    )

    display_decision(
        "SCENARIO 3: MAXIMUM POSITIONS",
        guardian.evaluate(replace(account, open_positions=3), trade),
    )

    display_decision(
        "SCENARIO 4: INVALID STOP",
        guardian.evaluate(account, replace(trade, stop_price=D("105"))),
    )

    display_decision(
        "SCENARIO 5: MAXIMUM EXPOSURE",
        guardian.evaluate(replace(account, total_exposure=D("60")), trade),
    )

    print("\n" + "=" * 60)
    print("RISK GUARDIAN DEMONSTRATION COMPLETE")
    print("NO ORDERS EXECUTED")
    print("=" * 60)
