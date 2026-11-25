from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
CONFIG_FILE = BASE_DIR / "config" / "strategies.yaml"


class StrategyEngine:
    def __init__(self):

        with open(CONFIG_FILE, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)

    @property
    def settings(self):
        return self.config["settings"]

    def load_latest_rankings(self):

        files = sorted(DATA_DIR.glob("opportunity_rankings_*.csv"))

        if not files:
            raise FileNotFoundError("No opportunity ranking files found.")

        latest = files[-1]

        print(f"\nLoading: {latest.name}")

        return pd.read_csv(latest)

    def momentum_strategy(self, row):

        cfg = self.config["momentum"]

        if not cfg["enabled"]:
            return False

        return (
            row["trend"] == "BULLISH"
            and cfg["minimum_rsi"] <= row["rsi"] <= cfg["maximum_rsi"]
            and row["momentum"] >= cfg["minimum_momentum"]
            and row["relative_volume"] >= cfg["minimum_relative_volume"]
            and row["macd"] > row["macd_signal"]
            and row["ema_9"] > row["ema_21"]
        )

    def breakout_strategy(self, row):

        cfg = self.config["breakout"]

        if not cfg["enabled"]:
            return False

        return (
            row["trend"] == "BULLISH"
            and row["close"] > row["resistance"]
            and row["relative_volume"] >= cfg["minimum_relative_volume"]
            and cfg["minimum_rsi"] <= row["rsi"] <= cfg["maximum_rsi"]
        )

    def pullback_strategy(self, row):

        cfg = self.config["pullback"]

        if not cfg["enabled"]:
            return False

        ema_distance = abs(row["close"] - row["ema_21"]) / row["ema_21"] * 100

        return (
            row["sma_20"] > row["sma_50"]
            and row["close"] > row["sma_200"]
            and cfg["minimum_rsi"] <= row["rsi"] <= cfg["maximum_rsi"]
            and ema_distance <= cfg["maximum_ema_distance_percent"]
        )

    def no_trade(self, row, reason):

        return {
            "symbol": row["symbol"],
            "strategy": "NONE",
            "signal": "NO_TRADE",
            "entry_price": 0.0,
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "risk_reward": 0.0,
            "score": row["score"],
            "reason": reason,
        }

    def generate_signal(self, row):

        required = [
            "close",
            "score",
            "rsi",
            "atr",
            "momentum",
            "relative_volume",
            "macd",
            "macd_signal",
            "ema_9",
            "ema_21",
            "sma_20",
            "sma_50",
            "sma_200",
            "resistance",
        ]

        values = row[required].astype(float)

        if not np.isfinite(values).all():
            return self.no_trade(row, "Invalid market data.")

        if row["close"] <= 0 or row["atr"] <= 0 or row["ema_21"] <= 0:
            return self.no_trade(row, "Invalid price or volatility.")

        if row["score"] < self.settings["minimum_score"]:
            return self.no_trade(row, "Opportunity score below minimum.")

        atr_percent = row["atr"] / row["close"] * 100

        if atr_percent > self.settings["maximum_atr_percent"]:
            return self.no_trade(row, "Volatility exceeds strategy limit.")

        strategy = None

        if self.breakout_strategy(row):
            strategy = "BREAKOUT"

        elif self.momentum_strategy(row):
            strategy = "MOMENTUM"

        elif self.pullback_strategy(row):
            strategy = "PULLBACK"

        if strategy is None:
            return self.no_trade(row, "No strategy conditions satisfied.")

        entry = float(row["close"])

        stop_distance = float(row["atr"]) * self.settings["stop_atr_multiplier"]

        stop_loss = entry - stop_distance

        if stop_loss <= 0:
            return self.no_trade(row, "Invalid calculated stop-loss.")

        reward_ratio = self.settings["minimum_risk_reward"]

        take_profit = entry + stop_distance * reward_ratio

        return {
            "symbol": row["symbol"],
            "strategy": strategy,
            "signal": "BUY",
            "entry_price": round(entry, 4),
            "stop_loss": round(stop_loss, 4),
            "take_profit": round(take_profit, 4),
            "risk_reward": reward_ratio,
            "score": row["score"],
            "reason": f"{strategy} conditions satisfied.",
        }

    def analyze(self, rankings):

        results = []

        for _, row in rankings.iterrows():
            signal = self.generate_signal(row)

            results.append(signal)

            print(
                f"{signal['symbol']:<6} "
                f"{signal['signal']:<10} "
                f"{signal['strategy']:<12} "
                f"Score: {signal['score']:>6.2f}"
            )

        return pd.DataFrame(results)

    def display_summary(self, results):

        candidates = results[results["signal"] == "BUY"]

        print("\n" + "=" * 75)
        print("                  STRATEGY SUMMARY")
        print("=" * 75)

        print(f"Symbols Analyzed : {len(results)}")
        print(f"Buy Signals      : {len(candidates)}")
        print(f"No Trade         : {len(results) - len(candidates)}")

        if candidates.empty:
            print("\nNo qualifying trading setups found.")

            return

        columns = [
            "symbol",
            "strategy",
            "entry_price",
            "stop_loss",
            "take_profit",
            "risk_reward",
            "score",
        ]

        print("\nHYPOTHETICAL TRADE CANDIDATES")
        print("-" * 75)

        print(candidates[columns].round(2).to_string(index=False))

    def save_results(self, results):

        DATA_DIR.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

        filename = DATA_DIR / f"strategy_signals_{timestamp}.csv"

        results.to_csv(filename, index=False)

        print(f"\nResults saved to: {filename}")


if __name__ == "__main__":
    print("=" * 75)
    print("                 JARVIS STRATEGY ENGINE")
    print("                       VERSION 0.7")
    print("=" * 75)

    print("Mode: RESEARCH ONLY")
    print("Order Execution: DISABLED")

    engine = StrategyEngine()

    rankings = engine.load_latest_rankings()

    results = engine.analyze(rankings)

    engine.display_summary(results)

    engine.save_results(results)

    print("\n" + "=" * 75)
    print("STRATEGY ENGINE: SUCCESSFUL")
    print("NO ORDERS EXECUTED")
    print("=" * 75)
