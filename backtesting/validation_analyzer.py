from pathlib import Path
from itertools import product
import json

import numpy as np
import pandas as pd
import yaml

from backtesting.strategy_analyzer import StrategyAnalyzer


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
CONFIG_FILE = BASE_DIR / "config" / "validation.yaml"

KEYS = ["symbol", "period", "scenario"]


class ValidationAnalyzer:

    def __init__(self):

        with open(CONFIG_FILE, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)["validation"]

        self.summary = None
        self.trades = None
        self.run_id = None

        self.periods = list(self.config["periods"])
        self.scenarios = list(
            self.config["execution_scenarios"]
        )

        self.symbols = self.config["symbols"]

    def load_reports(self):

        files = sorted(
            DATA_DIR.glob("validation_summary_*.csv")
        )

        if not files:
            raise FileNotFoundError(
                "Run the validation engine first."
            )

        summary_file = files[-1]

        self.run_id = summary_file.stem.replace(
            "validation_summary_", ""
        )

        trades_file = (
            DATA_DIR
            / f"validation_trades_{self.run_id}.csv"
        )

        errors_file = (
            DATA_DIR
            / f"validation_errors_{self.run_id}.json"
        )

        equity_file = (
            DATA_DIR
            / f"validation_equity_{self.run_id}.csv"
        )

        for file in [
            trades_file,
            errors_file,
            equity_file
        ]:
            if not file.exists():
                raise FileNotFoundError(file)

        with open(
            errors_file, "r", encoding="utf-8"
        ) as file:

            errors = json.load(file)

        if errors:
            raise RuntimeError(
                "Validation run contains failed simulations."
            )

        self.summary = pd.read_csv(summary_file)
        self.trades = pd.read_csv(trades_file)

        self.validate_reports()

        print(f"Validation Run ID: {self.run_id}")
        print(f"Simulations: {len(self.summary)}")
        print(f"Trade records: {len(self.trades)}")

    def validate_reports(self):

        required_summary = [
            "symbol",
            "period",
            "scenario",
            "total_trades",
            "total_return",
            "net_profit",
            "profit_factor",
            "expectancy",
            "max_drawdown",
            "risk_breaches",
            "qualification"
        ]

        required_trades = [
            "symbol",
            "strategy",
            "period",
            "scenario",
            "pnl",
            "planned_risk",
            "entry_date",
            "exit_date"
        ]

        for column in required_summary:
            if column not in self.summary.columns:
                raise ValueError(
                    f"Missing summary column: {column}"
                )

        for column in required_trades:
            if column not in self.trades.columns:
                raise ValueError(
                    f"Missing trade column: {column}"
                )

        if self.summary.duplicated(KEYS).any():
            raise ValueError(
                "Duplicate simulation records."
            )

        expected = set(product(
            self.symbols,
            self.periods,
            self.scenarios
        ))

        actual = set(
            self.summary[KEYS].itertuples(
                index=False,
                name=None
            )
        )

        if actual != expected:
            raise ValueError(
                "Simulation coverage does not match configuration."
            )

        numeric_columns = [
            "total_trades",
            "total_return",
            "net_profit",
            "expectancy",
            "max_drawdown",
            "risk_breaches"
        ]

        if not np.isfinite(
            self.summary[numeric_columns]
            .to_numpy(dtype=float)
        ).all():
            raise ValueError(
                "Invalid numeric summary values."
            )

        if not np.isfinite(
            self.trades[["pnl", "planned_risk"]]
            .to_numpy(dtype=float)
        ).all():
            raise ValueError(
                "Invalid trade values."
            )

        if (self.trades["planned_risk"] <= 0).any():
            raise ValueError(
                "Invalid planned risk."
            )

        trade_keys = set(
            self.trades[KEYS].itertuples(
                index=False,
                name=None
            )
        )

        if not trade_keys.issubset(expected):
            raise ValueError(
                "Unexpected trade groups."
            )

        grouped = self.trades.groupby(KEYS).agg(
            trade_count=("pnl", "size"),
            trade_pnl=("pnl", "sum")
        )

        audit = self.summary.merge(
            grouped.reset_index(),
            on=KEYS,
            how="left"
        )

        audit["trade_count"] = (
            audit["trade_count"].fillna(0)
        )

        audit["trade_pnl"] = (
            audit["trade_pnl"].fillna(0)
        )

        if not (
            audit["total_trades"]
            == audit["trade_count"]
        ).all():
            raise ValueError(
                "Trade counts do not reconcile."
            )

        if not np.allclose(
            audit["net_profit"],
            audit["trade_pnl"],
            atol=0.000001,
            rtol=0
        ):
            raise ValueError(
                "Trade PnL does not reconcile."
            )

        print("Report integrity: PASSED")

    def analyze_periods(self):

        return (
            self.summary
            .groupby(["period", "scenario"])
            .agg(
                simulations=("symbol", "size"),
                average_return=("total_return", "mean"),
                median_return=("total_return", "median"),
                total_trades=("total_trades", "sum"),
                profitable_simulations=(
                    "total_return",
                    lambda values: int((values > 0).sum())
                ),
                average_drawdown=("max_drawdown", "mean"),
                maximum_drawdown=("max_drawdown", "max"),
                risk_breaches=("risk_breaches", "sum")
            )
            .reset_index()
        )

    def analyze_symbols(self):

        records = []

        for symbol, group in self.summary.groupby("symbol"):

            baseline = group[
                group["scenario"] == "baseline"
            ]

            profitable = int(
                (group["total_return"] > 0).sum()
            )

            simulations = len(group)

            consistency = (
                profitable / simulations * 100
            )

            if profitable == simulations:
                status = "CONSISTENT_POSITIVE"

            elif profitable == 0:
                status = "NO_POSITIVE_SIMULATIONS"

            else:
                status = "MIXED_PERFORMANCE"

            records.append({
                "symbol": symbol,
                "simulations": simulations,
                "profitable_simulations": profitable,
                "profitability_consistency": consistency,
                "average_return": group[
                    "total_return"
                ].mean(),
                "worst_return": group[
                    "total_return"
                ].min(),
                "best_return": group[
                    "total_return"
                ].max(),
                "baseline_trades": int(
                    baseline["total_trades"].sum()
                ),
                "maximum_drawdown": group[
                    "max_drawdown"
                ].max(),
                "status": status,
                "trading_authorized": False
            })

        return pd.DataFrame(records).sort_values(
            [
                "profitability_consistency",
                "average_return"
            ],
            ascending=False
        )

    def analyze_cost_sensitivity(self):

        pivot = self.summary.pivot(
            index=["symbol", "period"],
            columns="scenario",
            values="total_return"
        ).reset_index()

        pivot["moderate_cost_impact"] = (
            pivot["baseline"] - pivot["moderate"]
        )

        pivot["stressed_cost_impact"] = (
            pivot["baseline"] - pivot["stressed"]
        )

        pivot["profitability_lost_under_stress"] = (
            (pivot["baseline"] > 0)
            & (pivot["stressed"] <= 0)
        )

        return pivot

    def analyze_strategies(self):

        records = []

        for keys, group in self.trades.groupby(
            ["strategy", "period", "scenario"]
        ):

            strategy, period, scenario = keys

            metrics = StrategyAnalyzer().calculate_metrics(
                group
            )

            metrics["strategy"] = strategy
            metrics["period"] = period
            metrics["scenario"] = scenario

            actual_loss = (
                -group["pnl"]
            ).clip(lower=0)

            breaches = (
                actual_loss
                > group["planned_risk"] + 0.000001
            )

            metrics["risk_breaches"] = int(
                breaches.sum()
            )

            metrics["risk_breach_percent"] = (
                breaches.mean() * 100
            )

            if metrics["total_trades"] < 30:

                status = "INSUFFICIENT_DATA"

            elif (
                metrics["profit_factor"] >= 1.5
                and metrics["expectancy"] > 0
            ):

                status = "RESEARCH_CANDIDATE"

            else:

                status = "NEEDS_IMPROVEMENT"

            metrics["research_status"] = status

            metrics["trading_authorized"] = False

            records.append(metrics)

        return pd.DataFrame(records)

    def analyze_deterioration(self):

        baseline = self.summary[
            self.summary["scenario"] == "baseline"
        ]

        pivot = baseline.pivot(
            index="symbol",
            columns="period",
            values="total_return"
        ).reset_index()

        pivot["validation_change"] = (
            pivot["validation"]
            - pivot["development"]
        )

        pivot["historical_test_change"] = (
            pivot["historical_test"]
            - pivot["development"]
        )

        return pivot.sort_values(
            "historical_test_change"
        )

    def save_report(self, dataframe, name):

        output = (
            DATA_DIR
            / f"validation_{name}_{self.run_id}.csv"
        )

        dataframe.to_csv(output, index=False)

        print(f"Saved: {output.name}")

    def run(self):

        print("=" * 80)
        print("JARVIS VALIDATION INTELLIGENCE")
        print("VERSION 0.8.5E")
        print("=" * 80)

        self.load_reports()

        periods = self.analyze_periods()
        symbols = self.analyze_symbols()
        costs = self.analyze_cost_sensitivity()
        strategies = self.analyze_strategies()
        deterioration = self.analyze_deterioration()

        print("\nPERIOD PERFORMANCE")
        print("-" * 80)
        print(periods.round(4).to_string(index=False))

        print("\nSYMBOL CONSISTENCY")
        print("-" * 80)

        print(
            symbols[[
                "symbol",
                "profitable_simulations",
                "profitability_consistency",
                "average_return",
                "baseline_trades",
                "status"
            ]].round(4).to_string(index=False)
        )

        print("\nEXECUTION COST SENSITIVITY")
        print("-" * 80)

        print(
            "Average moderate cost impact:",
            round(
                costs["moderate_cost_impact"].mean(),
                4
            )
        )

        print(
            "Average stressed cost impact:",
            round(
                costs["stressed_cost_impact"].mean(),
                4
            )
        )

        print(
            "Profitable simulations lost under stress:",
            int(
                costs[
                    "profitability_lost_under_stress"
                ].sum()
            )
        )

        print("\nSTRATEGY PERFORMANCE")
        print("-" * 80)

        if not strategies.empty:

            baseline = strategies[
                strategies["scenario"] == "baseline"
            ]

            print(
                baseline[[
                    "strategy",
                    "period",
                    "total_trades",
                    "win_rate",
                    "net_profit",
                    "profit_factor",
                    "expectancy",
                    "research_status"
                ]].round(4).to_string(index=False)
            )

        print("\nHISTORICAL PERFORMANCE DETERIORATION")
        print("-" * 80)

        print(
            deterioration.round(4)
            .to_string(index=False)
        )

        print("\nGENERATING REPORTS")
        print("-" * 80)

        self.save_report(periods, "period_analysis")
        self.save_report(symbols, "symbol_consistency")
        self.save_report(costs, "cost_sensitivity")
        self.save_report(strategies, "strategy_analysis")
        self.save_report(
            deterioration,
            "performance_deterioration"
        )

        print("\n" + "=" * 80)
        print("VALIDATION INTELLIGENCE COMPLETE")
        print("FORWARD VALIDATION REQUIRED")
        print("NO STRATEGIES AUTHORIZED FOR TRADING")
        print("NO BROKER ORDERS EXECUTED")
        print("=" * 80)


if __name__ == "__main__":

    analyzer = ValidationAnalyzer()
    analyzer.run()