import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from backtesting.backtest_engine import BacktestEngine
from backtesting.execution_model import ExecutionModel
from backtesting.parameter_selector import ParameterSelector

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"

CONFIG_FILE = BASE_DIR / "config" / "walk_forward.yaml"


class WalkForwardEngine:
    def __init__(self):

        with open(CONFIG_FILE, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)["walk_forward"]

        self.selector = ParameterSelector(self.config)

        self.symbols = self.config["symbols"]

        self.warmup = int(self.config["warmup_candles"])

        self.initial_capital = float(self.config["initial_capital"])

        self.validate_configuration()

        self.engine = BacktestEngine()

        self.engine.config["historical_days"] = self.config["historical_days"]

        self.engine.config["initial_capital"] = self.initial_capital

        self.engine.config["warmup_candles"] = self.warmup

        self.original_prepare_data = self.engine.prepare_data

        self.original_backtest_config = deepcopy(self.engine.config)

        self.original_strategy_config = deepcopy(self.engine.strategy.config)

        self.original_execution = self.engine.execution

        self.history_cache = {}

    def validate_configuration(self):

        if self.warmup < 200:
            raise ValueError("At least 200 warmup candles required.")

        if not np.isfinite(self.initial_capital) or self.initial_capital <= 0:
            raise ValueError("Invalid initial capital.")

        windows = self.config["windows"]

        for name in ["training_months", "testing_months", "step_months"]:
            value = windows[name]

            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"Invalid window setting: {name}")

        if windows["step_months"] < windows["testing_months"]:
            raise ValueError("Overlapping testing windows prohibited.")

        start = pd.Timestamp(self.config["research_start_date"])

        end = pd.Timestamp(self.config["research_end_date"])

        if start >= end:
            raise ValueError("Invalid research date range.")

    def generate_windows(self):

        settings = self.config["windows"]

        training_months = settings["training_months"]

        testing_months = settings["testing_months"]

        step_months = settings["step_months"]

        research_start = pd.Timestamp(self.config["research_start_date"])

        research_end = pd.Timestamp(self.config["research_end_date"])

        windows = []

        index = 0

        while True:
            training_start = research_start + pd.DateOffset(months=index * step_months)

            testing_start = training_start + pd.DateOffset(months=training_months)

            training_end = testing_start - pd.Timedelta(days=1)

            testing_end = (
                testing_start
                + pd.DateOffset(months=testing_months)
                - pd.Timedelta(days=1)
            )

            if testing_end > research_end:
                break

            if training_end >= testing_start:
                raise RuntimeError("Training and testing dates overlap.")

            windows.append(
                {
                    "window_id": f"W{index + 1:03d}",
                    "training_start": (training_start.strftime("%Y-%m-%d")),
                    "training_end": (training_end.strftime("%Y-%m-%d")),
                    "testing_start": (testing_start.strftime("%Y-%m-%d")),
                    "testing_end": (testing_end.strftime("%Y-%m-%d")),
                }
            )

            index += 1

        if not windows:
            raise ValueError("No complete walk-forward windows.")

        return windows

    def load_history(self, symbol):

        if symbol not in self.history_cache:
            print(f"Downloading historical data: {symbol}")

            data = self.original_prepare_data(symbol)

            if not isinstance(data.index, pd.DatetimeIndex):
                raise ValueError("Datetime index required.")

            if data.index.tz is None:
                raise ValueError("Timezone-aware data required.")

            data = data.sort_index()

            if data.index.has_duplicates:
                raise ValueError("Duplicate historical timestamps.")

            self.history_cache[symbol] = data

        return self.history_cache[symbol].copy()

    def prepare_period(self, symbol, start_date, end_date):

        history = self.load_history(symbol)

        dates = (
            history.index.tz_convert("America/New_York").tz_localize(None).normalize()
        )

        start = pd.Timestamp(start_date)

        end = pd.Timestamp(end_date)

        locations = np.flatnonzero((dates >= start) & (dates <= end))

        if len(locations) < 2:
            raise ValueError(f"Insufficient period data: {symbol}")

        first = int(locations[0])

        last = int(locations[-1])

        warmup_start = first - self.warmup

        if warmup_start < 0:
            raise ValueError(f"Insufficient warmup data: {symbol}")

        result = history.iloc[warmup_start : last + 1].copy()

        trading_dates = (
            result.index[self.warmup :]
            .tz_convert("America/New_York")
            .tz_localize(None)
            .normalize()
        )

        if trading_dates.min() < start or trading_dates.max() > end:
            raise RuntimeError("Trading period boundary violation.")

        return result

    def apply_candidate(self, candidate):

        self.engine.config = deepcopy(self.original_backtest_config)

        self.engine.strategy.config = deepcopy(self.original_strategy_config)

        minimum_score = float(candidate["minimum_score"])

        stop_multiplier = float(candidate["stop_atr_multiplier"])

        risk_reward = float(candidate["risk_reward_ratio"])

        self.engine.config["stop_atr_multiplier"] = stop_multiplier

        self.engine.config["risk_reward_ratio"] = risk_reward

        settings = self.engine.strategy.config["settings"]

        settings["minimum_score"] = minimum_score

        settings["stop_atr_multiplier"] = stop_multiplier

        settings["minimum_risk_reward"] = risk_reward

        execution = self.config["execution"]

        self.engine.execution = ExecutionModel(
            slippage_percent=(execution["slippage_percent"]),
            fee_percent=(execution["transaction_fee_percent"]),
        )

    def restore_engine(self):

        self.engine.config = deepcopy(self.original_backtest_config)

        self.engine.strategy.config = deepcopy(self.original_strategy_config)

        self.engine.prepare_data = self.original_prepare_data

        self.engine.execution = self.original_execution

    def verify_trade_dates(self, trades, start_date, end_date):

        start = pd.Timestamp(start_date)

        end = pd.Timestamp(end_date)

        for trade in trades:
            entry = (
                pd.Timestamp(trade["entry_date"])
                .tz_convert("America/New_York")
                .tz_localize(None)
                .normalize()
            )

            exit_date = (
                pd.Timestamp(trade["exit_date"])
                .tz_convert("America/New_York")
                .tz_localize(None)
                .normalize()
            )

            if not (start <= entry <= exit_date <= end):
                raise RuntimeError("Trade outside permitted research period.")

    def simulate_candidate(self, candidate, window, data_role):

        if data_role == "TRAINING":
            start_date = window["training_start"]

            end_date = window["training_end"]

        elif data_role == "TESTING":
            start_date = window["testing_start"]

            end_date = window["testing_end"]

        else:
            raise ValueError("Invalid research data role.")

        summaries = []

        all_trades = []

        all_curves = []

        try:
            self.apply_candidate(candidate)

            for symbol in self.symbols:
                data = self.prepare_period(symbol, start_date, end_date)

                self.engine.prepare_data = lambda requested_symbol: data.copy()

                metrics, trades, curve = self.engine.run(symbol)

                self.verify_trade_dates(trades, start_date, end_date)

                metadata = {
                    "window_id": window["window_id"],
                    "candidate_id": candidate["candidate_id"],
                    "data_role": data_role,
                    "period_start": start_date,
                    "period_end": end_date,
                    "trading_authorized": False,
                }

                metrics.update(metadata)

                summaries.append(metrics)

                for trade in trades:
                    trade.update(metadata)

                all_trades.extend(trades)

                curve = curve.copy()

                for key, value in metadata.items():
                    curve[key] = value

                all_curves.append(curve)

        finally:
            self.restore_engine()

        return summaries, all_trades, all_curves

    def create_cash_results(self, window):

        summaries = []

        curves = []

        for symbol in self.symbols:
            data = self.prepare_period(
                symbol, window["testing_start"], window["testing_end"]
            )

            dates = data.index[self.warmup :]

            summaries.append(
                {
                    "symbol": symbol,
                    "window_id": window["window_id"],
                    "candidate_id": "NONE",
                    "data_role": "TESTING",
                    "period_start": window["testing_start"],
                    "period_end": window["testing_end"],
                    "initial_capital": self.initial_capital,
                    "final_capital": self.initial_capital,
                    "net_profit": 0.0,
                    "total_return": 0.0,
                    "total_trades": 0,
                    "win_rate": 0.0,
                    "profit_factor": 0.0,
                    "expectancy": 0.0,
                    "max_drawdown": 0.0,
                    "research_status": "CASH_ONLY",
                    "trading_authorized": False,
                }
            )

            curves.append(
                pd.DataFrame(
                    {
                        "timestamp": dates,
                        "equity": self.initial_capital,
                        "symbol": symbol,
                        "window_id": window["window_id"],
                        "candidate_id": "NONE",
                        "data_role": "TESTING",
                        "period_start": window["testing_start"],
                        "period_end": window["testing_end"],
                        "trading_authorized": False,
                    }
                )
            )

        return summaries, [], curves

    def run(self):

        windows = self.generate_windows()

        candidates = self.selector.generate_candidates()

        training_evaluations = []

        selections = []

        testing_summaries = []

        testing_trades = []

        testing_curves = []

        errors = []

        print("=" * 80)

        print("JARVIS WALK-FORWARD RESEARCH ENGINE")

        print("VERSION 0.8.5F")

        print("=" * 80)

        print(f"Research windows: {len(windows)}")

        print(f"Parameter combinations: {len(candidates)}")

        print(f"Symbols: {len(self.symbols)}")

        print("Live trading: DISABLED")

        for window in windows:
            window_id = window["window_id"]

            print("\n" + "=" * 80)

            print(f"WINDOW: {window_id}")

            print(f"Training: {window['training_start']} to {window['training_end']}")

            print(f"Testing: {window['testing_start']} to {window['testing_end']}")

            print("=" * 80)

            evaluations = []

            window_failed = False

            for candidate in candidates:
                candidate_id = candidate["candidate_id"]

                print(f"\nTraining candidate: {candidate_id}")

                try:
                    summaries, trades, _ = self.simulate_candidate(
                        candidate, window, "TRAINING"
                    )

                    evaluation = self.selector.evaluate_candidate(
                        candidate, summaries, trades, data_role="TRAINING"
                    )

                    evaluation.update(window)

                    evaluations.append(evaluation)

                    training_evaluations.append(evaluation)

                    print(
                        f"{candidate_id} | "
                        f"Trades: {evaluation['total_trades']} | "
                        f"Return: {evaluation['average_return']:.4f}% | "
                        f"PF: {evaluation['profit_factor']:.4f} | "
                        f"Qualified: {evaluation['qualified']}"
                    )

                except Exception as error:
                    errors.append(
                        {
                            "window_id": window_id,
                            "candidate_id": candidate_id,
                            "data_role": "TRAINING",
                            "error": str(error),
                        }
                    )

                    window_failed = True

                    print(f"FAILED: {error}")

            if window_failed:
                selections.append(
                    {
                        **window,
                        "status": "WINDOW_FAILED",
                        "candidate_id": "NONE",
                        "reason": ("Incomplete training evaluation"),
                        "trading_authorized": False,
                    }
                )

                print("Testing skipped: incomplete training.")

                continue

            selection = self.selector.select(evaluations)

            selected = selection["selected_candidate"]

            selection_record = {
                **window,
                "status": selection["status"],
                "reason": selection["reason"],
                "candidate_id": (
                    selected["candidate_id"] if selected is not None else "NONE"
                ),
                "qualified_candidates": (selection.get("qualified_candidates", 0)),
                "trading_authorized": False,
            }

            if selected is not None:
                selection_record.update(
                    {
                        "minimum_score": (selected["minimum_score"]),
                        "stop_atr_multiplier": (selected["stop_atr_multiplier"]),
                        "risk_reward_ratio": (selected["risk_reward_ratio"]),
                        "training_average_return": (selected["average_return"]),
                        "training_profit_factor": (selected["profit_factor"]),
                        "training_trades": (selected["total_trades"]),
                    }
                )

            selections.append(selection_record)

            print(f"\nSelection: {selection['status']}")

            try:
                if selected is None:
                    print("No qualified candidate. Cash-only period.")

                    summaries, trades, curves = self.create_cash_results(window)

                else:
                    frozen_candidate = deepcopy(selected)

                    print(
                        f"Testing frozen candidate: {frozen_candidate['candidate_id']}"
                    )

                    summaries, trades, curves = self.simulate_candidate(
                        frozen_candidate, window, "TESTING"
                    )

                    for summary in summaries:
                        summary["research_status"] = "HISTORICAL_TEST"

                testing_summaries.extend(summaries)

                testing_trades.extend(trades)

                testing_curves.extend(curves)

            except Exception as error:
                selection_record["status"] = "TESTING_FAILED"

                errors.append(
                    {
                        "window_id": window_id,
                        "candidate_id": (selection_record["candidate_id"]),
                        "data_role": "TESTING",
                        "error": str(error),
                    }
                )

                print(f"TESTING FAILED: {error}")

        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

        DATA_DIR.mkdir(parents=True, exist_ok=True)

        reports = {
            "windows": pd.DataFrame(windows),
            "training": pd.DataFrame(training_evaluations),
            "selections": pd.DataFrame(selections),
            "testing_summary": pd.DataFrame(testing_summaries),
            "testing_trades": pd.DataFrame(
                testing_trades,
                columns=[
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
                    "window_id",
                    "candidate_id",
                    "data_role",
                    "period_start",
                    "period_end",
                    "trading_authorized",
                ],
            ),
            "testing_equity": (
                pd.concat(testing_curves, ignore_index=True)
                if testing_curves
                else pd.DataFrame(
                    columns=[
                        "timestamp",
                        "equity",
                        "symbol",
                        "window_id",
                        "candidate_id",
                        "data_role",
                        "period_start",
                        "period_end",
                        "trading_authorized",
                    ]
                )
            ),
        }

        print("\n" + "=" * 80)

        print("WALK-FORWARD RESEARCH SUMMARY")

        print("=" * 80)

        selection_df = reports["selections"]

        print(
            selection_df[["window_id", "candidate_id", "status"]].to_string(index=False)
        )

        testing_df = reports["testing_summary"]

        if not testing_df.empty:
            grouped = testing_df.groupby("window_id").agg(
                symbols=("symbol", "count"),
                average_return=("total_return", "mean"),
                total_trades=("total_trades", "sum"),
                maximum_drawdown=("max_drawdown", "max"),
            )

            print("\nTESTING PERFORMANCE")

            print(grouped.round(4).to_string())

        print("\nTraining evaluations:", len(training_evaluations))

        print("Testing symbol records:", len(testing_summaries))

        print("Testing trades:", len(testing_trades))

        print("Errors:", len(errors))

        print("\nGENERATING REPORTS")

        for name, dataframe in reports.items():
            output = DATA_DIR / f"walk_forward_{name}_{timestamp}.csv"

            dataframe.to_csv(output, index=False)

            print(f"Saved: {output.name}")

        error_file = DATA_DIR / f"walk_forward_errors_{timestamp}.json"

        with open(error_file, "w", encoding="utf-8") as file:
            json.dump(errors, file, indent=4)

        print(f"Saved: {error_file.name}")

        print("\nFORWARD VALIDATION REQUIRED")

        print("NO STRATEGIES AUTHORIZED FOR TRADING")

        print("NO BROKER ORDERS EXECUTED")

        if errors:
            raise RuntimeError("Research incomplete. Review error report.")

        print("WALK-FORWARD RESEARCH COMPLETE")

        return reports


if __name__ == "__main__":
    engine = WalkForwardEngine()

    engine.run()
