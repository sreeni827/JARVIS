from dataclasses import dataclass
from decimal import Decimal
from collections.abc import Mapping

from portfolio.capital_manager import CapitalManager
from portfolio.portfolio_config import PortfolioConfig

from portfolio.models import (
    PortfolioPosition,
    positive_decimal,
    validate_session
)


D = Decimal
ZERO = D("0")


# ============================================================
# HISTORICAL MARKET BAR
# ============================================================


@dataclass(frozen=True, slots=True)
class PositionMarketBar:

    session: object
    symbol: str
    opening: Decimal
    high: Decimal
    low: Decimal
    close: Decimal

    def __post_init__(self):

        validate_session(self.session, "session")

        if not isinstance(self.symbol, str):
            raise ValueError("Invalid symbol.")

        if not self.symbol.strip():
            raise ValueError("Symbol cannot be empty.")

        if self.symbol != self.symbol.strip().upper():
            raise ValueError("Symbol must be normalized.")

        for name in ("opening", "high", "low", "close"):

            value = positive_decimal(
                getattr(self, name),
                name
            )

            object.__setattr__(self, name, value)

        if self.high < max(self.opening, self.close):
            raise ValueError("Invalid market high.")

        if self.low > min(self.opening, self.close):
            raise ValueError("Invalid market low.")

        if self.low > self.high:
            raise ValueError("Market low exceeds high.")


# ============================================================
# POSITION EXIT DECISION
# ============================================================


@dataclass(frozen=True, slots=True)
class PositionExitDecision:

    triggered: bool
    symbol: str
    reason: str = "HOLD"
    reference_price: Decimal = ZERO
    phase: str = "NONE"


# ============================================================
# POSITION MANAGER
# ============================================================


class PositionManager:

    def __init__(self, capital_manager=None, config=None):

        if config is not None and not isinstance(
            config,
            PortfolioConfig
        ):
            raise ValueError("Invalid portfolio configuration.")

        if capital_manager is None:

            capital_manager = CapitalManager(config=config)

        if not isinstance(capital_manager, CapitalManager):
            raise ValueError("Invalid Capital Manager.")

        self.capital = capital_manager

        # Use the same validated configuration as the
        # Capital Manager to avoid conflicting settings.

        self._config = self.capital._config

        self._config.validate()

        if config is not None:

            if config.config != self._config.config:
                raise ValueError(
                    "Position and Capital Manager configuration mismatch."
                )

        settings = self._config.config

        self.maximum_holding_sessions = settings[
            "execution"
        ]["maximum_holding_sessions"]

        self.liquidate_at_end = settings[
            "execution"
        ]["liquidate_at_end"]

        self.liquidate_on_drawdown = settings[
            "risk"
        ]["liquidate_on_drawdown_breach"]

        self._opening_processed_session = None

        self._intraday_processed_session = None

        self._closing_processed_session = None

    # ========================================================
    # VALIDATION
    # ========================================================

    @staticmethod
    def validate_position(position):

        if not isinstance(position, PortfolioPosition):
            raise ValueError("Invalid portfolio position.")

    def validate_position_bar(self, position, bar):

        self.validate_position(position)

        if not isinstance(bar, PositionMarketBar):
            raise ValueError("Invalid historical market bar.")

        if position.symbol != bar.symbol:
            raise ValueError("Position and market symbol mismatch.")

        if bar.session < position.entry_date:
            raise ValueError("Market session precedes entry.")

    def validate_bars(self, bars):

        self.capital.require_session()

        if not isinstance(bars, Mapping):
            raise ValueError("Market bars must be a mapping.")

        # Validate all required bars before changing
        # any portfolio positions.

        for position in self.capital.state.positions:

            if position.symbol not in bars:

                raise ValueError(
                    f"Missing market bar: {position.symbol}"
                )

            bar = bars[position.symbol]

            self.validate_position_bar(position, bar)

            if bar.session != self.capital.session:

                raise ValueError(
                    f"Market session mismatch: {position.symbol}"
                )

        return True

    @staticmethod
    def hold(position):

        return PositionExitDecision(
            triggered=False,
            symbol=position.symbol
        )

    @staticmethod
    def exit_decision(position, price, reason, phase):

        return PositionExitDecision(
            triggered=True,
            symbol=position.symbol,
            reason=reason,
            reference_price=price,
            phase=phase
        )

    # ========================================================
    # OPENING EXIT EVALUATION
    # ========================================================

    def evaluate_opening_exit(
        self,
        position,
        bar,
        forced_exit=False
    ):

        self.validate_position_bar(position, bar)

        if type(forced_exit) is not bool:
            raise ValueError("Invalid forced exit flag.")

        # Newly entered positions cannot have an
        # overnight gap relative to their own entry.

        if position.entry_date == bar.session:
            return self.hold(position)

        if bar.opening <= position.stop_price:

            return self.exit_decision(
                position,
                bar.opening,
                "STOP_GAP",
                "OPENING"
            )

        if bar.opening >= position.target_price:

            return self.exit_decision(
                position,
                position.target_price,
                "TAKE_PROFIT_GAP",
                "OPENING"
            )

        if forced_exit:

            return self.exit_decision(
                position,
                bar.opening,
                "DRAWDOWN_EXIT",
                "OPENING"
            )

        return self.hold(position)

    # ========================================================
    # INTRADAY EXIT EVALUATION
    # ========================================================

    def evaluate_intraday_exit(self, position, bar):

        self.validate_position_bar(position, bar)

        # Daily OHLC data does not reveal whether the
        # stop or target occurred first.
        # Use the conservative stop-first assumption.

        if bar.low <= position.stop_price:

            # A new position may have a stop above
            # the raw opening price because of slippage.

            reference_price = min(
                position.stop_price,
                bar.opening
            )

            return self.exit_decision(
                position,
                reference_price,
                "STOP_LOSS",
                "INTRADAY"
            )

        if bar.high >= position.target_price:

            return self.exit_decision(
                position,
                position.target_price,
                "TAKE_PROFIT",
                "INTRADAY"
            )

        return self.hold(position)

    # ========================================================
    # CLOSING EXIT EVALUATION
    # ========================================================

    def evaluate_closing_exit(
        self,
        position,
        bar,
        final_session=False
    ):

        self.validate_position_bar(position, bar)

        if type(final_session) is not bool:
            raise ValueError("Invalid final session flag.")

        if (
            position.holding_sessions
            >= self.maximum_holding_sessions
        ):

            return self.exit_decision(
                position,
                bar.close,
                "TIME_EXIT",
                "CLOSING"
            )

        if final_session and self.liquidate_at_end:

            return self.exit_decision(
                position,
                bar.close,
                "END_OF_BACKTEST",
                "CLOSING"
            )

        return self.hold(position)

    # ========================================================
    # COMPLETE EXIT EVALUATION
    # ========================================================

    def evaluate_exit(
        self,
        position,
        bar,
        forced_exit=False,
        final_session=False
    ):

        if type(forced_exit) is not bool:
            raise ValueError("Invalid forced exit flag.")

        if type(final_session) is not bool:
            raise ValueError("Invalid final session flag.")

        decision = self.evaluate_opening_exit(
            position,
            bar,
            forced_exit=forced_exit
        )

        if decision.triggered:
            return decision

        decision = self.evaluate_intraday_exit(
            position,
            bar
        )

        if decision.triggered:
            return decision

        return self.evaluate_closing_exit(
            position,
            bar,
            final_session=final_session
        )

    # ========================================================
    # SIMULATED EXIT EXECUTION
    # ========================================================

    def execute_decision(self, decision):

        if not isinstance(decision, PositionExitDecision):
            raise ValueError("Invalid exit decision.")

        if not decision.triggered:
            raise ValueError("Cannot execute a HOLD decision.")

        result = self.capital.close_position(
            decision.symbol,
            decision.reference_price,
            decision.reason
        )

        if not result.executed:

            raise RuntimeError(
                f"Position exit failed: {result.reason}"
            )

        return result

    # ========================================================
    # PROCESS OPENING EXITS
    # ========================================================

    def process_opening_exits(self, bars):

        self.validate_bars(bars)

        session = self.capital.session

        if self._opening_processed_session == session:
            raise RuntimeError("Opening exits already processed.")

        # Capture the halt status before evaluating
        # any opening transactions.

        forced_exit = (
            self.capital.state.trading_halted
            and self.liquidate_on_drawdown
        )

        positions = tuple(
            self.capital.state.positions
        )

        if any(
            position.entry_date == session
            for position in positions
        ):

            raise RuntimeError(
                "Opening exits must precede new entries."
            )

        decisions = []

        for position in positions:

            decision = self.evaluate_opening_exit(
                position,
                bars[position.symbol],
                forced_exit=forced_exit
            )

            if decision.triggered:
                decisions.append(decision)

        results = tuple(
            self.execute_decision(decision)
            for decision in decisions
        )

        self._opening_processed_session = session

        return results

    # ========================================================
    # PROCESS INTRADAY EXITS
    # ========================================================

    def process_intraday_exits(self, bars):

        self.validate_bars(bars)

        session = self.capital.session

        if self._opening_processed_session != session:

            raise RuntimeError(
                "Process opening exits first."
            )

        if self._intraday_processed_session == session:

            raise RuntimeError(
                "Intraday exits already processed."
            )

        decisions = []

        for position in self.capital.state.positions:

            decision = self.evaluate_intraday_exit(
                position,
                bars[position.symbol]
            )

            if decision.triggered:
                decisions.append(decision)

        results = tuple(
            self.execute_decision(decision)
            for decision in decisions
        )

        self._intraday_processed_session = session

        return results

    # ========================================================
    # PROCESS CLOSING EXITS
    # ========================================================

    def process_closing_exits(
        self,
        bars,
        final_session=False
    ):

        self.validate_bars(bars)

        if type(final_session) is not bool:
            raise ValueError("Invalid final session flag.")

        session = self.capital.session

        if self._intraday_processed_session != session:

            raise RuntimeError(
                "Process intraday exits first."
            )

        if self._closing_processed_session == session:

            raise RuntimeError(
                "Closing exits already processed."
            )

        decisions = []

        for position in self.capital.state.positions:

            decision = self.evaluate_closing_exit(
                position,
                bars[position.symbol],
                final_session=final_session
            )

            if decision.triggered:
                decisions.append(decision)

        results = tuple(
            self.execute_decision(decision)
            for decision in decisions
        )

        self._closing_processed_session = session

        return results

    # ========================================================
    # MARK REMAINING POSITIONS
    # ========================================================

    def mark_open_positions(self, bars):

        self.validate_bars(bars)

        prices = {
            position.symbol: bars[position.symbol].opening
            for position in self.capital.state.positions
        }

        return self.capital.mark_to_market(prices)

    def mark_closing_positions(self, bars):

        self.validate_bars(bars)

        if (
            self._closing_processed_session
            != self.capital.session
        ):

            raise RuntimeError(
                "Process closing exits before closing valuation."
            )

        prices = {
            position.symbol: bars[position.symbol].close
            for position in self.capital.state.positions
        }

        return self.capital.mark_to_market(prices)

    # ========================================================
    # SUMMARY
    # ========================================================

    def display_summary(self):

        print("=" * 70)
        print("JARVIS PORTFOLIO POSITION MANAGER")
        print("VERSION 0.8.6.5")
        print("=" * 70)

        print(
            "Maximum holding sessions:",
            self.maximum_holding_sessions
        )

        print("Stop-target conflict: STOP_FIRST")
        print("Opening gap protection: ENABLED")
        print("Opening exit evaluation: ENABLED")
        print("Intraday exit evaluation: ENABLED")
        print("Closing exit evaluation: ENABLED")

        print(
            "Drawdown liquidation:",
            self.liquidate_on_drawdown
        )

        print(
            "End-of-backtest liquidation:",
            self.liquidate_at_end
        )

        print("Historical simulation only")
        print("Broker orders: DISABLED")
        print("Live trading: DISABLED")


if __name__ == "__main__":

    manager = PositionManager()

    manager.display_summary()