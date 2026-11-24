from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
import yaml

from backtesting.backtest_engine import BacktestEngine
from backtesting.execution_model import ExecutionModel


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"

CONFIG_FILE = BASE_DIR / "config" / "validation.yaml"


class ValidationEngine:

    def __init__(self):

        with open(
            CONFIG_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            self.config = yaml.safe_load(file)["validation"]

        self.periods = self.config["periods"]

        self.symbols = self.config["symbols"]

        self.scenarios = self.config["execution_scenarios"]

        self.qualification = self.config["qualification"]

        self.warmup = int(
            self.config["warmup_candles"]
        )

        self.validate_configuration()

        self.engine = BacktestEngine()

        self.engine.config["historical_days"] = (
            self.config["historical_days"]
        )

        self.engine.config["initial_capital"] = (
            self.config["initial_capital"]
        )

        self.engine.config["warmup_candles"] = (
            self.warmup
        )

        self.original_prepare_data = (
            self.engine.prepare_data
        )

        self.history_cache = {}

    def validate_configuration(self):

        safety = self.config["safety"]

        if safety["live_trading_enabled"]:

            raise RuntimeError(
                "Live trading is prohibited."
            )

        if not safety["historical_simulation_only"]:

            raise RuntimeError(
                "Historical simulation mode required."
            )

        if safety["automatic_strategy_activation"]:

            raise RuntimeError(
                "Automatic strategy activation prohibited."
            )

        if not safety["require_forward_validation"]:

            raise RuntimeError(
                "Forward validation must remain enabled."
            )

        if self.warmup < 200:

            raise ValueError(
                "At least 200 warmup candles required."
            )

        if not self.symbols:

            raise ValueError(
                "No validation symbols configured."
            )

        if len(set(self.symbols)) != len(self.symbols):

            raise ValueError(
                "Duplicate validation symbols."
            )

        boundaries = []

        for name, period in self.periods.items():

            start = pd.Timestamp(period["start_date"])

            end = pd.Timestamp(period["end_date"])

            if start > end:

                raise ValueError(
                    f"Invalid period: {name}"
                )

            boundaries.append(
                (start, end, name)
            )

        boundaries.sort()

        for i in range(1, len(boundaries)):

            previous = boundaries[i - 1]

            current = boundaries[i]

            if current[0] <= previous[1]:

                raise ValueError(
                    "Validation periods overlap."
                )

    def load_history(self, symbol):

        if symbol not in self.history_cache:

            print(
                f"Downloading historical data: {symbol}"
            )

            data = self.original_prepare_data(symbol)

            if not isinstance(
                data.index,
                pd.DatetimeIndex
            ):

                raise ValueError(
                    "Datetime index required."
                )

            if data.index.tz is None:

                raise ValueError(
                    "Timezone-aware historical data required."
                )

            data = data.sort_index()

            if data.index.has_duplicates:

                raise ValueError(
                    "Duplicate historical timestamps."
                )

            self.history_cache[symbol] = data

        return self.history_cache[symbol].copy()

    def prepare_period(
        self,
        symbol,
        start_date,
        end_date
    ):

        history = self.load_history(symbol)

        session_dates = (

            history.index

            .tz_convert("America/New_York")

            .tz_localize(None)

            .normalize()

        )

        start = pd.Timestamp(start_date)

        end = pd.Timestamp(end_date)

        mask = (

            (session_dates >= start)

            &

            (session_dates <= end)

        )

        locations = np.flatnonzero(mask)

        if len(locations) < 2:

            raise ValueError(
                f"Insufficient sessions for {symbol}."
            )

        first = int(locations[0])

        last = int(locations[-1])

        warmup_start = first - self.warmup

        if warmup_start < 0:

            raise ValueError(
                f"Insufficient warmup data for {symbol}."
            )

        period_data = history.iloc[

            warmup_start:last + 1

        ].copy()

        return period_data

    def calculate_risk_breaches(self, trades):

        if not trades:

            return 0, 0.0

        breaches = 0

        for trade in trades:

            planned = float(
                trade["planned_risk"]
            )

            actual_loss = max(

                0.0,

                -float(trade["pnl"])

            )

            if planned <= 0:

                raise ValueError(
                    "Invalid planned risk."
                )

            if actual_loss > planned + 0.000001:

                breaches += 1

        breach_rate = (

            breaches / len(trades)

        ) * 100

        return breaches, breach_rate

    def qualify(self, metrics):

        rules = self.qualification

        failures = []

        if (
            metrics["total_trades"]
            < rules["minimum_trades"]
        ):

            failures.append(
                "Insufficient trades"
            )

        if (
            metrics["profit_factor"]
            < rules["minimum_profit_factor"]
        ):

            failures.append(
                "Profit factor below threshold"
            )

        if (
            metrics["expectancy"]
            <= rules["minimum_expectancy"]
        ):

            failures.append(
                "Expectancy below threshold"
            )

        if (
            metrics["total_return"]
            <= rules["minimum_return_percent"]
        ):

            failures.append(
                "Return below threshold"
            )

        if (
            metrics["max_drawdown"]
            > rules["maximum_drawdown_percent"]
        ):

            failures.append(
                "Drawdown exceeds threshold"
            )

        if (
            metrics["risk_breach_percent"]
            > rules["maximum_risk_breach_percent"]
        ):

            failures.append(
                "Risk breach rate too high"
            )

        if failures:

            return (

                "NOT_QUALIFIED",

                "; ".join(failures)

            )

        return (

            "HISTORICAL_PASS",

            "Forward validation still required"

        )

    def run_simulation(
        self,
        symbol,
        period_name,
        scenario_name
    ):

        period = self.periods[period_name]

        scenario = self.scenarios[scenario_name]

        data = self.prepare_period(

            symbol,

            period["start_date"],

            period["end_date"]

        )

        self.engine.execution = ExecutionModel(

            slippage_percent=(
                scenario["slippage_percent"]
            ),

            fee_percent=(
                scenario["transaction_fee_percent"]
            )

        )

        self.engine.prepare_data = (
            lambda requested_symbol: data.copy()
        )

        try:

            metrics, trades, curve = self.engine.run(
                symbol
            )

        finally:

            self.engine.prepare_data = (
                self.original_prepare_data
            )

        breaches, breach_rate = (
            self.calculate_risk_breaches(trades)
        )

        metrics["period"] = period_name

        metrics["scenario"] = scenario_name

        metrics["risk_breaches"] = breaches

        metrics["risk_breach_percent"] = breach_rate

        metrics["slippage_percent"] = (
            scenario["slippage_percent"]
        )

        metrics["transaction_fee_percent"] = (
            scenario["transaction_fee_percent"]
        )

        status, reason = self.qualify(metrics)

        metrics["qualification"] = status

        metrics["qualification_reason"] = reason

        metrics["forward_validation_required"] = True

        for trade in trades:

            trade["period"] = period_name

            trade["scenario"] = scenario_name

        curve["period"] = period_name

        curve["scenario"] = scenario_name

        return metrics, trades, curve

    def run(self):

        summaries = []

        all_trades = []

        all_curves = []

        errors = []

        total = (

            len(self.symbols)

            * len(self.periods)

            * len(self.scenarios)

        )

        completed = 0

        print("=" * 80)

        print("JARVIS STRATEGY VALIDATION ENGINE")

        print("VERSION 0.8.5D")

        print("=" * 80)

        print("Mode: HISTORICAL SIMULATION")

        print("Live Trading: DISABLED")

        print(f"Planned simulations: {total}")

        for symbol in self.symbols:

            for period_name in self.periods:

                for scenario_name in self.scenarios:

                    completed += 1

                    print(

                        f"\n[{completed}/{total}] "

                        f"{symbol} | "

                        f"{period_name} | "

                        f"{scenario_name}"

                    )

                    try:

                        metrics, trades, curve = (
                            self.run_simulation(

                                symbol,

                                period_name,

                                scenario_name

                            )
                        )

                        summaries.append(metrics)

                        all_trades.extend(trades)

                        all_curves.append(curve)

                        print(

                            f"Return: "

                            f"{metrics['total_return']:.2f}% | "

                            f"Trades: "

                            f"{metrics['total_trades']} | "

                            f"Status: "

                            f"{metrics['qualification']}"

                        )

                    except Exception as error:

                        record = {

                            "symbol": symbol,

                            "period": period_name,

                            "scenario": scenario_name,

                            "error": str(error)

                        }

                        errors.append(record)

                        print(f"FAILED: {error}")

        if not summaries:

            raise RuntimeError(
                "No validation simulations completed."
            )

        results = pd.DataFrame(summaries)

        timestamp = datetime.now(

            timezone.utc

        ).strftime("%Y%m%d_%H%M%S")

        DATA_DIR.mkdir(

            parents=True,

            exist_ok=True

        )

        summary_file = (

            DATA_DIR

            / f"validation_summary_{timestamp}.csv"

        )

        trades_file = (

            DATA_DIR

            / f"validation_trades_{timestamp}.csv"

        )

        equity_file = (

            DATA_DIR

            / f"validation_equity_{timestamp}.csv"

        )

        errors_file = (

            DATA_DIR

            / f"validation_errors_{timestamp}.json"

        )

        results.to_csv(

            summary_file,

            index=False

        )

        trade_columns = [

            "symbol",

            "strategy",

            "entry_date",

            "exit_date",

            "entry_price",

            "exit_price",

            "quantity",

            "stop_price",

            "target_price",

            "planned_risk",

            "pnl",

            "exit_reason",

            "period",

            "scenario"

        ]

        pd.DataFrame(

            all_trades,

            columns=trade_columns

        ).to_csv(

            trades_file,

            index=False

        )

        pd.concat(

            all_curves,

            ignore_index=True

        ).to_csv(

            equity_file,

            index=False

        )

        with open(

            errors_file,

            "w",

            encoding="utf-8"

        ) as file:

            json.dump(

                errors,

                file,

                indent=4

            )

        print("\n" + "=" * 80)

        print("VALIDATION PERFORMANCE SUMMARY")

        print("=" * 80)

        grouped = results.groupby(

            ["period", "scenario"]

        ).agg(

            simulations=("symbol", "count"),

            average_return=("total_return", "mean"),

            total_trades=("total_trades", "sum"),

            average_drawdown=("max_drawdown", "mean"),

            maximum_drawdown=("max_drawdown", "max"),

            risk_breaches=("risk_breaches", "sum")

        )

        print(

            grouped.round(4).to_string()

        )

        print("\nQUALIFICATION SUMMARY")

        print(

            results["qualification"]

            .value_counts()

            .to_string()

        )

        print(f"\nSuccessful simulations: {len(results)}/{total}")

        print(f"Failed simulations: {len(errors)}")

        print(f"\nSaved: {summary_file.name}")

        print(f"Saved: {trades_file.name}")

        print(f"Saved: {equity_file.name}")

        print(f"Saved: {errors_file.name}")

        print("\nFORWARD VALIDATION REQUIRED")

        print("NO STRATEGIES AUTOMATICALLY ACTIVATED")

        print("NO BROKER ORDERS EXECUTED")

        return results


if __name__ == "__main__":

    engine = ValidationEngine()

    engine.run()