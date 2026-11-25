import math
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

BASE_DIR = Path(__file__).resolve().parents[1]

CONFIG_FILE = BASE_DIR / "config" / "walk_forward.yaml"


class ParameterSelector:
    def __init__(self, config=None):

        if config is None:
            with open(CONFIG_FILE, "r", encoding="utf-8") as file:
                config = yaml.safe_load(file)["walk_forward"]

        self.config = config

        self.rules = config["selection"]

        self.validate_configuration()

    def validate_configuration(self):

        safety = self.config["safety"]

        if safety["live_trading_enabled"]:
            raise RuntimeError("Live trading is prohibited.")

        if not safety["historical_simulation_only"]:
            raise RuntimeError("Historical simulation mode required.")

        if safety["automatic_strategy_activation"]:
            raise RuntimeError("Automatic strategy activation prohibited.")

        if safety["allow_test_data_for_selection"]:
            raise RuntimeError("Test data cannot be used for selection.")

        if not safety["require_forward_validation"]:
            raise RuntimeError("Forward validation is required.")

        if not self.rules["allow_no_selection"]:
            raise RuntimeError("Forced parameter selection is prohibited.")

        if not self.config["symbols"]:
            raise ValueError("No research symbols configured.")

        if len(set(self.config["symbols"])) != len(self.config["symbols"]):
            raise ValueError("Duplicate research symbols.")

        for name, values in self.config["parameter_candidates"].items():
            if not values:
                raise ValueError(f"No candidates for {name}.")

            if len(values) != len(set(values)):
                raise ValueError(f"Duplicate candidates for {name}.")

            for value in values:
                if not math.isfinite(value):
                    raise ValueError(f"Invalid candidate for {name}.")

                if value <= 0:
                    raise ValueError(f"Candidate must be positive: {name}")

                if name == "minimum_score" and value > 100:
                    raise ValueError("Minimum score cannot exceed 100.")

    def generate_candidates(self):

        parameters = self.config["parameter_candidates"]

        names = list(parameters.keys())

        values = [parameters[name] for name in names]

        candidates = []

        for index, combination in enumerate(product(*values), start=1):
            candidate = dict(zip(names, combination))

            candidate["candidate_id"] = f"C{index:03d}"

            candidates.append(candidate)

        return candidates

    def calculate_training_metrics(self, summaries, trades):

        summaries = pd.DataFrame(summaries)

        trades = pd.DataFrame(trades)

        if summaries.empty:
            raise ValueError("Training summaries cannot be empty.")

        required_summary = [
            "symbol",
            "total_return",
            "total_trades",
            "max_drawdown",
            "net_profit",
        ]

        for column in required_summary:
            if column not in summaries.columns:
                raise ValueError(f"Missing training column: {column}")

        if summaries["symbol"].duplicated().any():
            raise ValueError("Duplicate symbol training results.")

        expected_symbols = set(self.config["symbols"])

        actual_symbols = set(summaries["symbol"])

        if actual_symbols != expected_symbols:
            raise ValueError("Incomplete training symbol coverage.")

        numeric_columns = ["total_return", "total_trades", "max_drawdown", "net_profit"]

        if not np.isfinite(summaries[numeric_columns].to_numpy(dtype=float)).all():
            raise ValueError("Invalid training summary values.")

        if (summaries["total_trades"] < 0).any():
            raise ValueError("Negative trade count.")

        if (summaries["max_drawdown"] < 0).any():
            raise ValueError("Negative drawdown.")

        expected_trades = int(summaries["total_trades"].sum())

        if len(trades) != expected_trades:
            raise ValueError("Training trade count mismatch.")

        if trades.empty:
            gross_profit = 0.0
            gross_loss = 0.0
            net_profit = 0.0
            profit_factor = 0.0
            expectancy = 0.0
            win_rate = 0.0
            risk_breaches = 0
            risk_breach_percent = 0.0

        else:
            required_trades = ["symbol", "pnl", "planned_risk"]

            for column in required_trades:
                if column not in trades.columns:
                    raise ValueError(f"Missing trade column: {column}")

            if not set(trades["symbol"]).issubset(expected_symbols):
                raise ValueError("Unexpected training trade symbol.")

            if not np.isfinite(
                trades[["pnl", "planned_risk"]].to_numpy(dtype=float)
            ).all():
                raise ValueError("Invalid training trade values.")

            if (trades["planned_risk"] <= 0).any():
                raise ValueError("Invalid planned risk.")

            pnl = trades["pnl"]

            gross_profit = float(pnl[pnl > 0].sum())

            gross_loss = abs(float(pnl[pnl < 0].sum()))

            net_profit = float(pnl.sum())

            if gross_loss > 0:
                profit_factor = gross_profit / gross_loss

            elif gross_profit > 0:
                profit_factor = float("inf")

            else:
                profit_factor = 0.0

            expectancy = float(pnl.mean())

            win_rate = float((pnl > 0).mean() * 100)

            actual_loss = (-pnl).clip(lower=0)

            breaches = actual_loss > trades["planned_risk"] + 0.000001

            risk_breaches = int(breaches.sum())

            risk_breach_percent = float(breaches.mean() * 100)

        if not math.isclose(
            net_profit,
            float(summaries["net_profit"].sum()),
            abs_tol=0.000001,
            rel_tol=0,
        ):
            raise ValueError("Training PnL does not reconcile.")

        return {
            "symbols_tested": len(summaries),
            "total_trades": expected_trades,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "net_profit": net_profit,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "win_rate": win_rate,
            "average_return": float(summaries["total_return"].mean()),
            "median_return": float(summaries["total_return"].median()),
            "maximum_drawdown": float(summaries["max_drawdown"].max()),
            "profitable_symbols": int((summaries["total_return"] > 0).sum()),
            "risk_breaches": risk_breaches,
            "risk_breach_percent": risk_breach_percent,
        }

    def qualify_candidate(self, metrics):

        rules = self.rules

        failures = []

        finite_fields = [
            "total_trades",
            "expectancy",
            "average_return",
            "maximum_drawdown",
            "risk_breach_percent",
        ]

        for field in finite_fields:
            if not math.isfinite(metrics[field]):
                failures.append(f"Invalid metric: {field}")

        profit_factor = metrics["profit_factor"]

        if math.isnan(profit_factor) or profit_factor < 0:
            failures.append("Invalid profit factor")

        if metrics["total_trades"] < rules["minimum_training_trades"]:
            failures.append("Insufficient training trades")

        if profit_factor < rules["minimum_profit_factor"]:
            failures.append("Profit factor below threshold")

        if metrics["expectancy"] <= rules["minimum_expectancy"]:
            failures.append("Expectancy below threshold")

        if metrics["average_return"] <= rules["minimum_average_return_percent"]:
            failures.append("Average return below threshold")

        if metrics["maximum_drawdown"] > rules["maximum_drawdown_percent"]:
            failures.append("Drawdown exceeds threshold")

        if metrics["risk_breach_percent"] > rules["maximum_risk_breach_percent"]:
            failures.append("Risk breach rate too high")

        if failures:
            return False, "; ".join(failures)

        return True, "Passed training requirements"

    def evaluate_candidate(self, candidate, summaries, trades, data_role="TRAINING"):

        if data_role != "TRAINING":
            raise RuntimeError("Only training data may be evaluated.")

        metrics = self.calculate_training_metrics(summaries, trades)

        approved, reason = self.qualify_candidate(metrics)

        return {
            **candidate,
            **metrics,
            "data_role": "TRAINING",
            "qualified": approved,
            "qualification_reason": reason,
            "trading_authorized": False,
        }

    def select(self, evaluations):

        if not evaluations:
            return {
                "status": "NO_SELECTION",
                "reason": "No candidate evaluations available",
                "selected_candidate": None,
                "trading_authorized": False,
            }

        for evaluation in evaluations:
            if evaluation.get("data_role") != "TRAINING":
                raise RuntimeError("Non-training results cannot be selected.")

        qualified = []

        for evaluation in evaluations:
            approved, _ = self.qualify_candidate(evaluation)

            if approved and evaluation.get("qualified") is True:
                qualified.append(evaluation)

        if not qualified:
            return {
                "status": "NO_SELECTION",
                "reason": "No configuration passed training requirements",
                "selected_candidate": None,
                "trading_authorized": False,
            }

        ranked = sorted(
            qualified,
            key=lambda row: (
                -row["average_return"],
                -row["expectancy"],
                row["maximum_drawdown"],
                row["risk_breach_percent"],
                row["candidate_id"],
            ),
        )

        selected = ranked[0].copy()

        return {
            "status": "SELECTED_FOR_HISTORICAL_TEST",
            "reason": "Highest ranked qualified training configuration",
            "selected_candidate": selected,
            "qualified_candidates": len(qualified),
            "forward_validation_required": True,
            "trading_authorized": False,
        }


if __name__ == "__main__":
    selector = ParameterSelector()

    candidates = selector.generate_candidates()

    print("=" * 70)
    print("JARVIS PARAMETER SELECTION ENGINE")
    print("VERSION 0.8.5F")
    print("=" * 70)

    print(f"Parameter combinations: {len(candidates)}")

    print()

    print(pd.DataFrame(candidates).to_string(index=False))

    print()

    print("Training-only selection: ENABLED")
    print("Forced selection: DISABLED")
    print("Automatic activation: DISABLED")
    print("Live trading: DISABLED")
    print("Forward validation: REQUIRED")
