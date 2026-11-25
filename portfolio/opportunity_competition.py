from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

import pandas as pd

from intelligence.opportunity_ranker import OpportunityRanker
from portfolio.capital_manager import CapitalManager
from portfolio.historical_data import HistoricalDataEngine
from portfolio.models import (
    PortfolioExecutionResult,
    PortfolioOpportunity,
    PortfolioRejection,
    positive_decimal,
    validate_score,
    validate_session,
)
from portfolio.portfolio_config import PortfolioConfig
from strategies.strategy_engine import StrategyEngine

# ============================================================
# COMPETITION RESULTS
# ============================================================


@dataclass(frozen=True, slots=True)
class OpportunityBatch:
    session: date
    opportunities: tuple
    rejections: tuple
    evaluated_symbols: int

    @property
    def qualified_count(self):

        return len(self.opportunities)

    @property
    def rejected_count(self):

        return len(self.rejections)


@dataclass(frozen=True, slots=True)
class CompetitionResult:
    session: date
    opportunities: tuple
    executions: tuple
    rejections: tuple

    @property
    def executed_count(self):

        return sum(result.executed for result in self.executions)

    @property
    def rejected_count(self):

        return len(self.rejections)

    @property
    def executed_symbols(self):

        return tuple(result.symbol for result in self.executions if result.executed)


# ============================================================
# OPPORTUNITY COMPETITION ENGINE
# ============================================================


class OpportunityCompetitionEngine:
    COLUMN_MAPPING = {
        "SMA_20": "sma_20",
        "SMA_50": "sma_50",
        "SMA_200": "sma_200",
        "EMA_9": "ema_9",
        "EMA_21": "ema_21",
        "RSI_14": "rsi",
        "MACD": "macd",
        "MACD_SIGNAL": "macd_signal",
        "ATR_14": "atr",
        "RELATIVE_VOLUME": "relative_volume",
        "ROC_10": "momentum",
        "SUPPORT": "support",
        "RESISTANCE": "resistance",
        "TREND": "trend",
    }

    REQUIRED_FIELDS = (
        "symbol",
        "signal_date",
        "close",
        "sma_20",
        "sma_50",
        "sma_200",
        "ema_9",
        "ema_21",
        "rsi",
        "macd",
        "macd_signal",
        "atr",
        "relative_volume",
        "momentum",
        "support",
        "resistance",
        "trend",
    )

    STRATEGY_NAMES = ("MOMENTUM", "BREAKOUT", "PULLBACK")

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(self, config=None, ranker=None, strategy_engine=None):

        if config is None:
            config = PortfolioConfig()

        if not isinstance(config, PortfolioConfig):
            raise ValueError("Invalid portfolio configuration.")

        config.validate()

        self._config = config

        settings = config.config

        self.symbols = tuple(settings["universe"]["symbols"])

        selection = settings["opportunity_selection"]

        strategy = settings["strategy"]

        self.minimum_score = validate_score(selection["minimum_score"])

        self.maximum_new_positions = selection["maximum_new_positions_per_session"]

        self.stop_atr_multiplier = positive_decimal(
            strategy["stop_atr_multiplier"], "stop_atr_multiplier"
        )

        self.risk_reward_ratio = positive_decimal(
            strategy["risk_reward_ratio"], "risk_reward_ratio"
        )

        self.enabled_strategies = frozenset(
            name for name in self.STRATEGY_NAMES if strategy[f"{name.lower()}_enabled"]
        )

        self.ranker = ranker if ranker is not None else OpportunityRanker()

        self.strategy_engine = (
            strategy_engine if strategy_engine is not None else StrategyEngine()
        )

        self.synchronize_strategy_settings()

    # ========================================================
    # STRATEGY CONFIGURATION
    # ========================================================

    def synchronize_strategy_settings(self):

        settings = self.strategy_engine.settings

        settings["minimum_score"] = float(self.minimum_score)

        settings["stop_atr_multiplier"] = float(self.stop_atr_multiplier)

        settings["minimum_risk_reward"] = float(self.risk_reward_ratio)

        for name in self.STRATEGY_NAMES:
            self.strategy_engine.config[name.lower()]["enabled"] = (
                name in self.enabled_strategies
            )

    # ========================================================
    # SIGNAL NORMALIZATION
    # ========================================================

    def normalize_signal(self, symbol, row, session):

        validate_session(session, "session")

        if symbol not in self.symbols:
            raise ValueError(f"Symbol outside universe: {symbol}")

        if not isinstance(row, pd.Series):
            raise ValueError(f"{symbol}: signal must be a pandas Series.")

        if row.index.has_duplicates:
            raise ValueError(f"{symbol}: duplicate signal fields.")

        normalized = row.rename(index=self.COLUMN_MAPPING).copy(deep=True)

        if normalized.index.has_duplicates:
            raise ValueError(f"{symbol}: conflicting signal fields.")

        missing = set(self.REQUIRED_FIELDS) - set(normalized.index)

        if missing:
            raise ValueError(f"{symbol}: missing signal fields {sorted(missing)}")

        if normalized["symbol"] != symbol:
            raise ValueError(f"{symbol}: signal symbol mismatch.")

        signal_date = normalized["signal_date"]

        validate_session(signal_date, "signal_date")

        if signal_date >= session:
            raise ValueError(f"{symbol}: signal must precede execution.")

        from portfolio.models import to_decimal

        for field in self.REQUIRED_FIELDS:
            if field in ("symbol", "signal_date", "trend"):
                continue

            value = to_decimal(normalized[field], field)

            normalized[field] = float(value)

        positive_fields = (
            "close",
            "sma_20",
            "sma_50",
            "sma_200",
            "ema_9",
            "ema_21",
            "atr",
            "support",
            "resistance",
        )

        for field in positive_fields:
            positive_decimal(normalized[field], field)

        if not 0 <= normalized["rsi"] <= 100:
            raise ValueError(f"{symbol}: invalid RSI.")

        if normalized["relative_volume"] < 0:
            raise ValueError(f"{symbol}: invalid relative volume.")

        if normalized["trend"] not in ("BULLISH", "BEARISH", "NEUTRAL"):
            raise ValueError(f"{symbol}: invalid trend.")

        return normalized

    # ========================================================
    # REJECTION RECORD
    # ========================================================

    @staticmethod
    def reject(session, symbol, strategy, score, reason):

        return PortfolioRejection(
            session=session,
            symbol=symbol,
            strategy=strategy,
            score=score,
            reason=reason,
        )

    # ========================================================
    # INDIVIDUAL OPPORTUNITY EVALUATION
    # ========================================================

    def evaluate_signal(self, symbol, row, session):

        normalized = self.normalize_signal(symbol, row, session)

        score, components, atr_percent = self.ranker.calculate_score(normalized)

        score = validate_score(score)

        normalized["score"] = float(score)

        normalized["atr_percent"] = atr_percent

        if score < self.minimum_score:
            return self.reject(
                session, symbol, "NONE", score, "Minimum opportunity score not reached."
            )

        signal = self.strategy_engine.generate_signal(normalized)

        if not isinstance(signal, Mapping):
            raise ValueError(f"{symbol}: invalid strategy response.")

        if signal.get("symbol") != symbol:
            raise ValueError(f"{symbol}: strategy symbol mismatch.")

        action = signal.get("signal")

        strategy = signal.get("strategy", "NONE")

        if action == "NO_TRADE":
            return self.reject(
                session,
                symbol,
                strategy,
                score,
                str(signal.get("reason", "No qualifying strategy.")),
            )

        if action != "BUY":
            raise ValueError(f"{symbol}: unsupported strategy action.")

        if strategy not in self.STRATEGY_NAMES:
            raise ValueError(f"{symbol}: unknown strategy.")

        if strategy not in self.enabled_strategies:
            return self.reject(
                session, symbol, strategy, score, "Strategy is disabled."
            )

        return PortfolioOpportunity(
            symbol=symbol,
            strategy=strategy,
            signal_date=normalized["signal_date"],
            score=score,
            reference_price=str(normalized["close"]),
            atr=str(normalized["atr"]),
            stop_atr_multiplier=self.stop_atr_multiplier,
            risk_reward_ratio=self.risk_reward_ratio,
        )

    # ========================================================
    # RANK OPPORTUNITIES
    # ========================================================

    def rank_opportunities(self, opportunities):

        opportunities = tuple(opportunities)

        for opportunity in opportunities:
            if not isinstance(opportunity, PortfolioOpportunity):
                raise ValueError("Invalid portfolio opportunity.")

        symbols = [opportunity.symbol for opportunity in opportunities]

        if len(symbols) != len(set(symbols)):
            raise ValueError("Duplicate opportunity symbols.")

        return tuple(sorted(opportunities, key=lambda item: (-item.score, item.symbol)))

    # ========================================================
    # BUILD DAILY OPPORTUNITY BATCH
    # ========================================================

    def build_batch(self, historical_data, session):

        validate_session(session, "session")

        self._config.validate()

        if not isinstance(historical_data, HistoricalDataEngine):
            raise ValueError("Invalid historical data engine.")

        if not historical_data.loaded:
            raise RuntimeError("Historical data has not been loaded.")

        if set(historical_data.symbols) != set(self.symbols):
            raise ValueError("Historical universe mismatch.")

        previous = historical_data.get_previous_session(session)

        signals = historical_data.get_session_signals(session)

        if set(signals) != set(self.symbols):
            raise ValueError("Incomplete historical signal universe.")

        opportunities = []

        rejections = []

        for symbol in self.symbols:
            row = signals[symbol]

            if row["signal_date"] != previous:
                raise ValueError(f"{symbol}: stale or incorrect signal date.")

            result = self.evaluate_signal(symbol, row, session)

            if isinstance(result, PortfolioOpportunity):
                opportunities.append(result)

            elif isinstance(result, PortfolioRejection):
                rejections.append(result)

            else:
                raise RuntimeError("Unexpected evaluation result.")

        ranked = self.rank_opportunities(opportunities)

        if len(ranked) + len(rejections) != len(self.symbols):
            raise RuntimeError("Opportunity accounting mismatch.")

        return OpportunityBatch(
            session=session,
            opportunities=ranked,
            rejections=tuple(rejections),
            evaluated_symbols=len(self.symbols),
        )

    # ========================================================
    # EXECUTE RANKED COMPETITION
    # ========================================================

    def execute_batch(self, batch, capital_manager, opening_prices):

        self._config.validate()

        if not isinstance(batch, OpportunityBatch):
            raise ValueError("Invalid opportunity batch.")

        if not isinstance(capital_manager, CapitalManager):
            raise ValueError("Invalid capital manager.")

        if capital_manager.session != batch.session:
            raise ValueError("Capital manager session mismatch.")

        if capital_manager._config.config != self._config.config:
            raise ValueError("Capital manager configuration mismatch.")

        if not isinstance(opening_prices, Mapping):
            raise ValueError("Opening prices must be a mapping.")

        ranked = self.rank_opportunities(batch.opportunities)

        if ranked != batch.opportunities:
            raise ValueError("Opportunity batch is not correctly ranked.")

        if batch.evaluated_symbols != len(self.symbols):
            raise ValueError("Incomplete opportunity batch.")

        if not all(isinstance(item, PortfolioRejection) for item in batch.rejections):
            raise ValueError("Invalid rejection records.")

        accounted_symbols = [item.symbol for item in batch.opportunities] + [
            item.symbol for item in batch.rejections
        ]

        if len(accounted_symbols) != len(self.symbols) or set(accounted_symbols) != set(
            self.symbols
        ):
            raise ValueError("Opportunity batch accounting mismatch.")

        for rejection in batch.rejections:
            if rejection.session != batch.session:
                raise ValueError("Rejection session mismatch.")

        validated_prices = {}

        # Validate every candidate before modifying capital.

        for opportunity in ranked:
            if opportunity.signal_date >= batch.session:
                raise ValueError("Invalid opportunity signal timing.")

            if opportunity.symbol not in opening_prices:
                raise ValueError(f"Missing opening price: {opportunity.symbol}")

            validated_prices[opportunity.symbol] = positive_decimal(
                opening_prices[opportunity.symbol], "opening_price"
            )

        executions = []

        rejections = list(batch.rejections)

        successful_entries = 0

        for opportunity in ranked:
            if successful_entries >= self.maximum_new_positions:
                rejections.append(
                    self.reject(
                        batch.session,
                        opportunity.symbol,
                        opportunity.strategy,
                        opportunity.score,
                        "Competition entry limit reached.",
                    )
                )

                continue

            result = capital_manager.open_position(
                opportunity, validated_prices[opportunity.symbol]
            )

            if not isinstance(result, PortfolioExecutionResult):
                raise RuntimeError("Invalid capital execution result.")

            executions.append(result)

            if result.executed:
                successful_entries += 1

            else:
                rejections.append(
                    self.reject(
                        batch.session,
                        opportunity.symbol,
                        opportunity.strategy,
                        opportunity.score,
                        result.reason,
                    )
                )

        if successful_entries + len(rejections) != batch.evaluated_symbols:
            raise RuntimeError("Competition result accounting mismatch.")

        return CompetitionResult(
            session=batch.session,
            opportunities=ranked,
            executions=tuple(executions),
            rejections=tuple(rejections),
        )

    # ========================================================
    # REPORTING
    # ========================================================

    def display_batch(self, batch):

        print("=" * 75)
        print("JARVIS OPPORTUNITY COMPETITION")
        print("VERSION 0.8.6.7")
        print("=" * 75)

        print("Session:", batch.session)

        print("Evaluated symbols:", batch.evaluated_symbols)

        print("Qualified opportunities:", batch.qualified_count)

        print("Rejected opportunities:", batch.rejected_count)

        print()

        print("RANKED OPPORTUNITIES")

        if not batch.opportunities:
            print("No qualifying opportunities.")

        for rank, opportunity in enumerate(batch.opportunities, start=1):
            print(
                f"{rank:2}. "
                f"{opportunity.symbol:6} | "
                f"{opportunity.strategy:10} | "
                f"Score: {opportunity.score:.2f}"
            )

        print()

        print("REJECTED OPPORTUNITIES")

        for rejection in batch.rejections:
            print(
                f"{rejection.symbol:6} | "
                f"Score: {rejection.score:.2f} | "
                f"{rejection.reason}"
            )

        print()

        print("Historical simulation only")
        print("Broker orders: DISABLED")
        print("Live trading: DISABLED")

    def display_result(self, result):

        print("=" * 75)
        print("PORTFOLIO COMPETITION RESULTS")
        print("=" * 75)

        print("Session:", result.session)

        print("Successful entries:", result.executed_count)

        print("Rejected opportunities:", result.rejected_count)

        print()

        for execution in result.executions:
            status = "EXECUTED" if execution.executed else "REJECTED"

            print(f"{execution.symbol:6} | {status:8} | {execution.reason}")

        print()

        print("Historical simulation only")
        print("Broker orders: DISABLED")
        print("Live trading: DISABLED")


# ============================================================
# STANDALONE RESEARCH
# ============================================================


if __name__ == "__main__":
    config = PortfolioConfig()

    historical = HistoricalDataEngine(config=config)

    print("Loading historical market data...")

    historical.load()

    competition = OpportunityCompetitionEngine(config=config)

    session = historical.sessions[-1]

    batch = competition.build_batch(historical, session)

    competition.display_batch(batch)
