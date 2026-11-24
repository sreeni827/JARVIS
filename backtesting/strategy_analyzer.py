from pathlib import Path
from datetime import datetime, timezone

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"


class StrategyAnalyzer:

    def load_latest_trades(self):

        files = sorted(
            DATA_DIR.glob("backtest_trades_*.csv")
        )

        if not files:
            raise FileNotFoundError(
                "No backtest trade files found."
            )

        latest = files[-1]

        print(f"\nLoading: {latest.name}")

        trades = pd.read_csv(latest)

        required = {
            "symbol",
            "strategy",
            "pnl",
            "exit_reason",
            "entry_date",
            "exit_date"
        }

        if not required.issubset(trades.columns):

            raise ValueError(
                "Required trade columns are missing."
            )

        return trades

    def calculate_metrics(self, trades):

        pnl = trades["pnl"]

        winners = pnl[pnl > 0]

        losers = pnl[pnl < 0]

        gross_profit = winners.sum()

        gross_loss = abs(losers.sum())

        total = len(trades)

        if gross_loss > 0:

            profit_factor = gross_profit / gross_loss

        elif gross_profit > 0:

            profit_factor = float("inf")

        else:

            profit_factor = 0

        return {

            "total_trades": total,

            "winning_trades": len(winners),

            "losing_trades": len(losers),

            "breakeven_trades": int(
                (pnl == 0).sum()
            ),

            "win_rate": (
                len(winners) / total * 100
                if total else 0
            ),

            "net_profit": pnl.sum(),

            "gross_profit": gross_profit,

            "gross_loss": gross_loss,

            "profit_factor": profit_factor,

            "expectancy": (
                pnl.mean() if total else 0
            ),

            "average_win": (
                winners.mean()
                if len(winners) else 0
            ),

            "average_loss": (
                losers.mean()
                if len(losers) else 0
            ),

            "largest_win": (
                pnl.max() if total else 0
            ),

            "largest_loss": (
                pnl.min() if total else 0
            )

        }

    def classify(self, metrics):

        if metrics["total_trades"] < 20:

            return "INSUFFICIENT DATA"

        if (
            metrics["profit_factor"] >= 1.5
            and metrics["expectancy"] > 0
        ):

            return "PROMISING"

        if (
            metrics["profit_factor"] >= 1
            and metrics["expectancy"] > 0
        ):

            return "NEEDS VALIDATION"

        return "NEEDS IMPROVEMENT"

    def analyze_strategies(self, trades):

        results = []

        for strategy, group in trades.groupby(
            "strategy"
        ):

            metrics = self.calculate_metrics(group)

            metrics["strategy"] = strategy

            metrics["assessment"] = self.classify(
                metrics
            )

            results.append(metrics)

        return pd.DataFrame(results)

    def analyze_symbols(self, trades):

        results = []

        for (symbol, strategy), group in trades.groupby(
            ["symbol", "strategy"]
        ):

            metrics = self.calculate_metrics(group)

            metrics["symbol"] = symbol

            metrics["strategy"] = strategy

            results.append(metrics)

        return pd.DataFrame(results)

    def analyze_exit_reasons(self, trades):

        return trades.groupby(
            "exit_reason"
        ).agg(

            total_trades=("pnl", "count"),

            net_profit=("pnl", "sum"),

            average_pnl=("pnl", "mean")

        ).reset_index()

    def analyze_holding_period(self, trades):

        entries = pd.to_datetime(
            trades["entry_date"],
            utc=True
        )

        exits = pd.to_datetime(
            trades["exit_date"],
            utc=True
        )

        trades = trades.copy()

        trades["holding_calendar_days"] = (
            exits - entries
        ).dt.total_seconds() / 86400

        return trades.groupby(
            "strategy"
        ).agg(

            average_holding_days=(
                "holding_calendar_days",
                "mean"
            ),

            maximum_holding_days=(
                "holding_calendar_days",
                "max"
            )

        ).reset_index()

    def run(self):

        trades = self.load_latest_trades()

        if trades.empty:

            print("No trades available for analysis.")

            return

        strategy_results = self.analyze_strategies(
            trades
        )

        symbol_results = self.analyze_symbols(
            trades
        )

        exit_results = self.analyze_exit_reasons(
            trades
        )

        holding_results = self.analyze_holding_period(
            trades
        )

        print("\nOVERALL TRADE STATISTICS")
        print("-" * 90)

        overall = self.calculate_metrics(trades)

        for key, value in overall.items():

            print(f"{key:<25}: {value:.4f}")

        print("\nPERFORMANCE BY STRATEGY")
        print("-" * 90)

        columns = [

            "strategy",

            "total_trades",

            "win_rate",

            "net_profit",

            "profit_factor",

            "expectancy",

            "assessment"

        ]

        print(
            strategy_results[columns]
            .round(4)
            .to_string(index=False)
        )

        print("\nEXIT REASON ANALYSIS")
        print("-" * 90)

        print(
            exit_results
            .round(4)
            .to_string(index=False)
        )

        print("\nHOLDING PERIOD ANALYSIS")
        print("-" * 90)

        print(
            holding_results
            .round(2)
            .to_string(index=False)
        )

        timestamp = datetime.now(
            timezone.utc
        ).strftime("%Y%m%d_%H%M%S")

        reports = {

            "strategy_performance": strategy_results,

            "symbol_strategy_performance": symbol_results,

            "exit_reason_analysis": exit_results,

            "holding_period_analysis": holding_results

        }

        for name, report in reports.items():

            filename = (
                DATA_DIR / f"{name}_{timestamp}.csv"
            )

            report.to_csv(
                filename,
                index=False
            )

            print(f"\nSaved: {filename.name}")


if __name__ == "__main__":

    print("=" * 90)

    print("              JARVIS STRATEGY PERFORMANCE INTELLIGENCE")

    print("                         VERSION 0.8.1")

    print("=" * 90)

    print("Mode: HISTORICAL RESEARCH")

    print("Order Execution: DISABLED")

    analyzer = StrategyAnalyzer()

    analyzer.run()

    print("\n" + "=" * 90)

    print("STRATEGY PERFORMANCE ANALYSIS COMPLETE")

    print("=" * 90)