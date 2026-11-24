from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import yaml


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"

CONFIG_FILE = BASE_DIR / "config" / "scoring.yaml"


class OpportunityRanker:

    def __init__(self):

        with open(
            CONFIG_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            self.config = yaml.safe_load(file)

        self.weights = self.config["weights"]

        self.thresholds = self.config["thresholds"]

        if sum(self.weights.values()) != 100:
            raise ValueError(
                "Scoring weights must total 100."
            )

    def load_latest_scan(self):

        files = sorted(
            DATA_DIR.glob("market_scan_*.csv")
        )

        if not files:

            raise FileNotFoundError(
                "No market scan CSV files found."
            )

        latest_file = files[-1]

        print(f"\nLoading: {latest_file.name}")

        return pd.read_csv(latest_file)

    def calculate_score(self, row):

        scores = {}

        # 1. TREND STRENGTH

        trend_values = {
            "BULLISH": 1.0,
            "NEUTRAL": 0.4,
            "BEARISH": 0.0
        }

        scores["trend"] = (
            trend_values.get(row["trend"], 0)
            * self.weights["trend"]
        )

        # 2. RSI CONDITIONS

        rsi = row["rsi"]

        if 50 <= rsi <= 65:
            rsi_factor = 1.0

        elif 40 <= rsi < 50:
            rsi_factor = 0.6

        elif 65 < rsi <= 70:
            rsi_factor = 0.6

        elif 30 <= rsi < 40:
            rsi_factor = 0.3

        else:
            rsi_factor = 0.1

        scores["rsi"] = (
            rsi_factor * self.weights["rsi"]
        )

        # 3. MACD MOMENTUM

        if (
            row["macd"] > row["macd_signal"]
            and row["macd"] > 0
        ):

            macd_factor = 1.0

        elif row["macd"] > row["macd_signal"]:

            macd_factor = 0.6

        elif row["macd"] > 0:

            macd_factor = 0.3

        else:

            macd_factor = 0.0

        scores["macd"] = (
            macd_factor * self.weights["macd"]
        )

        # 4. PRICE MOMENTUM

        momentum = row["momentum"]

        momentum_factor = np.clip(
            momentum / 5,
            0,
            1
        )

        scores["momentum"] = (
            momentum_factor
            * self.weights["momentum"]
        )

        # 5. RELATIVE VOLUME

        if momentum > 0:

            volume_factor = np.clip(
                row["relative_volume"] / 2,
                0,
                1
            )

        else:

            volume_factor = 0.0

        scores["volume"] = (
            volume_factor * self.weights["volume"]
        )

        # 6. MOVING AVERAGE ALIGNMENT

        conditions = [

            row["close"] > row["sma_20"],

            row["close"] > row["sma_50"],

            row["close"] > row["sma_200"],

            row["ema_9"] > row["ema_21"]

        ]

        ma_factor = sum(conditions) / len(conditions)

        scores["moving_averages"] = (
            ma_factor
            * self.weights["moving_averages"]
        )

        # 7. VOLATILITY CONDITIONS

        atr_percent = (
            row["atr"] / row["close"]
        ) * 100

        if atr_percent <= 2:

            volatility_factor = 1.0

        elif atr_percent <= 4:

            volatility_factor = 0.7

        elif atr_percent <= 6:

            volatility_factor = 0.4

        else:

            volatility_factor = 0.1

        scores["volatility"] = (
            volatility_factor
            * self.weights["volatility"]
        )

        total_score = round(
            float(sum(scores.values())),
            2
        )

        return total_score, scores, atr_percent

    def classify(self, score):

        if score >= self.thresholds["high_priority"]:

            return "HIGH PRIORITY"

        if score >= self.thresholds["strong"]:

            return "STRONG"

        if score >= self.thresholds["moderate"]:

            return "MODERATE"

        if score >= self.thresholds["low"]:

            return "LOW PRIORITY"

        return "WEAK"

    def rank(self, data):

        results = []

        required = [
            "close", "rsi", "macd",
            "macd_signal", "momentum",
            "relative_volume", "sma_20",
            "sma_50", "sma_200",
            "ema_9", "ema_21", "atr"
        ]

        for _, row in data.iterrows():

            values = row[required].astype(float)

            if not np.isfinite(values).all():

                print(
                    f"Skipping {row['symbol']}: "
                    "Invalid data"
                )

                continue

            if row["close"] <= 0 or row["atr"] < 0:

                print(
                    f"Skipping {row['symbol']}: "
                    "Invalid price or ATR"
                )

                continue

            score, breakdown, atr_percent = (
                self.calculate_score(row)
            )

            result = row.to_dict()

            result["score"] = score

            result["classification"] = (
                self.classify(score)
            )

            result["atr_percent"] = round(
                atr_percent,
                2
            )

            for name, points in breakdown.items():

                result[f"score_{name}"] = round(
                    float(points),
                    2
                )

            results.append(result)

        if not results:

            raise ValueError(
                "No valid symbols available for ranking."
            )

        ranked = pd.DataFrame(results)

        ranked = ranked.sort_values(
            "score",
            ascending=False
        ).reset_index(drop=True)

        ranked.insert(
            0,
            "rank",
            range(1, len(ranked) + 1)
        )

        return ranked

    def save_results(self, ranked):

        timestamp = datetime.now(
            timezone.utc
        ).strftime("%Y%m%d_%H%M%S")

        filename = (
            DATA_DIR /
            f"opportunity_rankings_{timestamp}.csv"
        )

        ranked.to_csv(
            filename,
            index=False
        )

        print(f"\nResults saved to: {filename}")

    def display_results(self, ranked):

        top_count = self.config["settings"][
            "top_candidates"
        ]

        columns = [
            "rank",
            "symbol",
            "close",
            "trend",
            "rsi",
            "score",
            "classification"
        ]

        print("\nTOP RESEARCH CANDIDATES")

        print("-" * 85)

        print(
            ranked[columns]
            .head(top_count)
            .round(2)
            .to_string(index=False)
        )

        print("\nCLASSIFICATION SUMMARY")

        print("-" * 85)

        print(
            ranked["classification"]
            .value_counts()
            .to_string()
        )

        print("\nHIGHEST SCORING SYMBOL")

        print("-" * 85)

        best = ranked.iloc[0]

        print(f"Symbol          : {best['symbol']}")
        print(f"Score           : {best['score']}/100")
        print(f"Classification  : {best['classification']}")
        print(f"Trend           : {best['trend']}")
        print(f"RSI             : {best['rsi']:.2f}")
        print(f"ATR Percentage  : {best['atr_percent']:.2f}%")

        print("\nSCORE BREAKDOWN")

        for component, maximum in self.weights.items():

            points = best[f"score_{component}"]

            print(
                f"{component:<20}: "
                f"{points:>6.2f}/{maximum}"
            )


if __name__ == "__main__":

    print("=" * 85)
    print("                    JARVIS OPPORTUNITY RANKER")
    print("                            VERSION 0.5")
    print("=" * 85)

    ranker = OpportunityRanker()

    market_data = ranker.load_latest_scan()

    ranked = ranker.rank(market_data)

    ranker.display_results(ranked)

    ranker.save_results(ranked)

    print("\n" + "=" * 85)
    print("                  OPPORTUNITY RANKER: SUCCESSFUL")
    print("=" * 85)