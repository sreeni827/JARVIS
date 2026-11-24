from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

from backtesting.strategy_analyzer import StrategyAnalyzer


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"


def align_trades_with_regimes(trades, regimes):

    trades = trades.copy()

    regimes = regimes.copy()

    trades["entry_session"] = (

        pd.to_datetime(
            trades["entry_date"],
            utc=True
        )

        .dt.tz_convert("America/New_York")

        .dt.tz_localize(None)

        .dt.normalize()

    )

    regimes["regime_session"] = (

        pd.to_datetime(
            regimes["session_date"]
        ).dt.normalize()

    )

    if regimes["regime_session"].duplicated().any():

        raise ValueError(
            "Duplicate historical regime sessions."
        )

    trades = trades.sort_values(
        "entry_session"
    )

    regimes = regimes.sort_values(
        "regime_session"
    )

    aligned = pd.merge_asof(

        trades,

        regimes,

        left_on="entry_session",

        right_on="regime_session",

        direction="backward",

        allow_exact_matches=False,

        tolerance=pd.Timedelta(days=7)

    )

    aligned["regime"] = (
        aligned["regime"].fillna("UNKNOWN")
    )

    return aligned


class RegimeStrategyAnalyzer:

    def __init__(self):

        self.analyzer = StrategyAnalyzer()

    def load_regimes(self):

        files = sorted(
            DATA_DIR.glob(
                "historical_regimes_*.csv"
            )
        )

        if not files:

            raise FileNotFoundError(
                "No historical regime reports found."
            )

        latest = files[-1]

        print(f"Loading: {latest.name}")

        return pd.read_csv(latest)

    def analyze_groups(self, trades, columns):

        results = []

        for keys, group in trades.groupby(columns):

            metrics = self.analyzer.calculate_metrics(
                group
            )

            if not isinstance(keys, tuple):

                keys = (keys,)

            for column, value in zip(columns, keys):

                metrics[column] = value

            metrics["assessment"] = (
                self.analyzer.classify(metrics)
            )

            results.append(metrics)

        return pd.DataFrame(results)

    def run(self):

        trades = self.analyzer.load_latest_trades()

        regimes = self.load_regimes()

        aligned = align_trades_with_regimes(
            trades,
            regimes
        )

        unknown = (
            aligned["regime"] == "UNKNOWN"
        ).sum()

        print("\nHISTORICAL ALIGNMENT")
        print("-" * 75)

        print(f"Total Trades   : {len(aligned)}")

        print(
            f"Matched Trades : "
            f"{len(aligned) - unknown}"
        )

        print(f"Unknown Regime : {unknown}")

        valid = aligned[
            aligned["regime"] != "UNKNOWN"
        ]

        if valid.empty:

            raise ValueError(
                "No trades matched valid market regimes."
            )

        regime_results = self.analyze_groups(

            valid,

            ["regime"]

        )

        strategy_results = self.analyze_groups(

            valid,

            ["strategy", "regime"]

        )

        metrics = [

            "total_trades",

            "win_rate",

            "net_profit",

            "profit_factor",

            "expectancy",

            "assessment"

        ]

        print("\nPERFORMANCE BY MARKET REGIME")
        print("-" * 90)

        print(

            regime_results[
                ["regime"] + metrics
            ]

            .round(4)

            .to_string(index=False)

        )

        print("\nSTRATEGY PERFORMANCE BY MARKET REGIME")
        print("-" * 100)

        print(

            strategy_results[
                ["strategy", "regime"] + metrics
            ]

            .round(4)

            .to_string(index=False)

        )

        timestamp = datetime.now(
            timezone.utc
        ).strftime("%Y%m%d_%H%M%S")

        reports = {

            "regime_trade_history": aligned,

            "regime_performance": regime_results,

            "regime_strategy_performance": (
                strategy_results
            )

        }

        for name, report in reports.items():

            filename = (
                DATA_DIR /
                f"{name}_{timestamp}.csv"
            )

            report.to_csv(
                filename,
                index=False
            )

            print(f"\nSaved: {filename.name}")


if __name__ == "__main__":

    print("=" * 90)

    print("        JARVIS MARKET-AWARE STRATEGY INTELLIGENCE")

    print("                      VERSION 0.8.4")

    print("=" * 90)

    print("Mode: HISTORICAL RESEARCH")

    print("Order Execution: DISABLED\n")

    analyzer = RegimeStrategyAnalyzer()

    analyzer.run()

    print("\n" + "=" * 90)

    print("MARKET-AWARE STRATEGY ANALYSIS COMPLETE")

    print("=" * 90)