from pathlib import Path

import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"


def run():

    files = sorted(

        DATA_DIR.glob(
            "backtest_trades_*.csv"
        )

    )

    if not files:

        raise FileNotFoundError(
            "No backtest trade files found."
        )

    latest = files[-1]

    print(f"Loading: {latest.name}")

    trades = pd.read_csv(latest)

    required = {

        "symbol",

        "strategy",

        "pnl",

        "planned_risk",

        "exit_reason"

    }

    if not required.issubset(
        trades.columns
    ):

        raise ValueError(

            "Run the V0.8.5B backtest first."

        )

    if trades.empty:

        print("No trades available.")

        return

    if not np.isfinite(

        trades[
            ["pnl", "planned_risk"]
        ].to_numpy()

    ).all():

        raise ValueError(
            "Invalid risk or PnL values."
        )

    if (trades["planned_risk"] <= 0).any():

        raise ValueError(
            "Invalid planned trade risk."
        )

    trades["actual_loss"] = (

        -trades["pnl"]

    ).clip(lower=0)

    trades["risk_multiple"] = (

        trades["actual_loss"]

        /

        trades["planned_risk"]

    )

    trades["risk_breach"] = (

        trades["actual_loss"]

        >

        trades["planned_risk"] + 0.000001

    )

    breaches = trades[
        trades["risk_breach"]
    ].copy()

    print("\nRISK BREACH SUMMARY")

    print("-" * 75)

    print(
        f"Total trades        : {len(trades)}"
    )

    print(
        f"Risk breaches       : {len(breaches)}"
    )

    print(

        f"Breach rate         : "

        f"{len(breaches) / len(trades) * 100:.2f}%"

    )

    print(

        f"Largest actual loss : "

        f"${trades['actual_loss'].max():.4f}"

    )

    print(

        f"Largest risk multiple: "

        f"{trades['risk_multiple'].max():.2f}x"

    )

    if not breaches.empty:

        print("\nBREACHES BY EXIT REASON")

        print("-" * 75)

        grouped = breaches.groupby(

            "exit_reason"

        ).agg(

            breaches=("symbol", "count"),

            total_loss=("actual_loss", "sum"),

            average_risk_multiple=(
                "risk_multiple",
                "mean"
            )

        )

        print(
            grouped.round(4).to_string()
        )

        print("\nLARGEST RISK BREACHES")

        print("-" * 75)

        columns = [

            "symbol",

            "strategy",

            "planned_risk",

            "actual_loss",

            "risk_multiple",

            "exit_reason"

        ]

        print(

            breaches.sort_values(

                "risk_multiple",

                ascending=False

            )[columns]

            .head(10)

            .round(4)

            .to_string(index=False)

        )

    run_id = latest.stem.replace(
        "backtest_trades_", ""
    )

    output = (

        DATA_DIR

        /

        f"risk_breaches_{run_id}.csv"

    )

    breaches.to_csv(

        output,

        index=False

    )

    print(f"\nSaved: {output.name}")


if __name__ == "__main__":

    print("=" * 75)

    print("          JARVIS RISK BREACH INTELLIGENCE")

    print("                    VERSION 0.8.5B")

    print("=" * 75)

    run()

    print("\nRISK BREACH ANALYSIS COMPLETE")