from pathlib import Path

import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"


class BacktestAudit:

    def __init__(self):

        self.passed = 0
        self.failed = 0

    def check(self, name, condition):

        if condition:

            self.passed += 1
            status = "PASS"

        else:

            self.failed += 1
            status = "FAIL"

        print(f"[{status}] {name}")

    def load_reports(self):

        files = sorted(
            DATA_DIR.glob("backtest_summary_*.csv")
        )

        if not files:
            raise FileNotFoundError(
                "No backtest summary found."
            )

        summary_file = files[-1]

        timestamp = summary_file.stem.replace(
            "backtest_summary_", ""
        )

        trades_file = (
            DATA_DIR /
            f"backtest_trades_{timestamp}.csv"
        )

        equity_file = (
            DATA_DIR /
            f"backtest_equity_{timestamp}.csv"
        )

        print(f"Backtest ID: {timestamp}")

        summary = pd.read_csv(summary_file)
        trades = pd.read_csv(trades_file)
        equity = pd.read_csv(equity_file)

        return summary, trades, equity

    def run(self):

        summary, trades, equity = self.load_reports()

        print("\nGENERAL DATA VALIDATION")
        print("-" * 70)

        self.check(
            "Summary contains results",
            not summary.empty
        )

        self.check(
            "Trade records exist",
            not trades.empty
        )

        self.check(
            "Equity records exist",
            not equity.empty
        )

        self.check(
            "Trade PnL values are finite",
            np.isfinite(trades["pnl"]).all()
        )

        self.check(
            "All equity values are positive",
            (equity["equity"] > 0).all()
        )

        self.check(
            "All trade quantities are positive",
            (trades["quantity"] > 0).all()
        )

        self.check(
            "All entry prices are positive",
            (trades["entry_price"] > 0).all()
        )

        self.check(
            "All exit prices are positive",
            (trades["exit_price"] > 0).all()
        )

        entries = pd.to_datetime(
            trades["entry_date"], utc=True
        )

        exits = pd.to_datetime(
            trades["exit_date"], utc=True
        )

        self.check(
            "Exit dates are not before entry dates",
            (exits >= entries).all()
        )

        self.check(
            "Trade count matches summary",
            len(trades) == summary["total_trades"].sum()
        )

        self.check(
            "Summary symbols are unique",
            summary["symbol"].is_unique
        )

        self.check(
            "All trades belong to summarized symbols",
            set(trades["symbol"]).issubset(
                set(summary["symbol"])
            )
        )

        print("\nSYMBOL-LEVEL RECONCILIATION")
        print("-" * 70)

        for _, row in summary.iterrows():

            symbol = row["symbol"]

            symbol_trades = trades[
                trades["symbol"] == symbol
            ]

            symbol_equity = equity[
                equity["symbol"] == symbol
            ].copy()

            symbol_equity["timestamp"] = pd.to_datetime(
                symbol_equity["timestamp"],
                utc=True
            )

            symbol_equity = symbol_equity.sort_values(
                "timestamp"
            )

            print(f"\n{symbol}")

            if symbol_equity.empty:

                self.check(
                    f"{symbol}: Equity history exists",
                    False
                )

                continue

            pnl = symbol_trades["pnl"].sum()

            initial = row["initial_capital"]
            final = row["final_capital"]

            self.check(
                f"{symbol}: Trade PnL reconciles",
                np.isclose(
                    initial + pnl,
                    final,
                    atol=0.000001,
                    rtol=0
                )
            )

            self.check(
                f"{symbol}: Trade count reconciles",
                len(symbol_trades)
                == row["total_trades"]
            )

            self.check(
                f"{symbol}: Final equity reconciles",
                np.isclose(
                    symbol_equity["equity"].iloc[-1],
                    final,
                    atol=0.000001,
                    rtol=0
                )
            )

            expected_return = (
                (final - initial) / initial * 100
            )

            self.check(
                f"{symbol}: Return calculation reconciles",
                np.isclose(
                    expected_return,
                    row["total_return"],
                    atol=0.000001,
                    rtol=0
                )
            )

            peak = symbol_equity["equity"].cummax()

            drawdown = (
                (peak - symbol_equity["equity"])
                / peak * 100
            ).max()

            self.check(
                f"{symbol}: Drawdown reconciles",
                np.isclose(
                    drawdown,
                    row["max_drawdown"],
                    atol=0.000001,
                    rtol=0
                )
            )

        print("\nRISK OBSERVATIONS")
        print("-" * 70)

        losses = trades[trades["pnl"] < 0]

        print(
            "Largest simulated loss: "
            f"${abs(losses['pnl'].min()):.4f}"
            if not losses.empty
            else "No simulated losses"
        )

        print(
            f"Losses exceeding $0.50: "
            f"{(trades['pnl'] < -0.500001).sum()}"
        )

        print(
            f"Losses exceeding $1.00: "
            f"{(trades['pnl'] < -1).sum()}"
        )

        print("\n" + "=" * 70)
        print("BACKTEST AUDIT SUMMARY")
        print("=" * 70)

        print(f"Passed checks : {self.passed}")
        print(f"Failed checks : {self.failed}")
        print(f"Total checks  : {self.passed + self.failed}")

        if self.failed == 0:

            print("AUDIT RESULT: RECONCILIATION PASSED")

        else:

            print("AUDIT RESULT: INVESTIGATION REQUIRED")

            raise SystemExit(1)


if __name__ == "__main__":

    print("=" * 70)
    print("               JARVIS BACKTEST AUDIT")
    print("                    VERSION 0.8.2")
    print("=" * 70)

    audit = BacktestAudit()
    audit.run()