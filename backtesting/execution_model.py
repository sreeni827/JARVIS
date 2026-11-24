from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ExitDecision:

    triggered: bool
    price: float = 0.0
    reason: str = "HOLD"


class ExecutionModel:

    def __init__(
        self,
        slippage_percent=0.05,
        fee_percent=0.02
    ):

        for value in (
            slippage_percent,
            fee_percent
        ):

            if not math.isfinite(value):

                raise ValueError(
                    "Execution costs must be finite."
                )

            if not 0 <= value < 100:

                raise ValueError(
                    "Invalid execution cost."
                )

        self.slippage = slippage_percent / 100
        self.fee = fee_percent / 100

    @staticmethod
    def validate_positive(*values):

        if not all(
            math.isfinite(value) and value > 0
            for value in values
        ):

            raise ValueError(
                "Prices and quantities must be positive."
            )

    def buy_price(self, market_price):

        self.validate_positive(market_price)

        return market_price * (
            1 + self.slippage
        )

    def sell_price(self, market_price):

        self.validate_positive(market_price)

        return market_price * (
            1 - self.slippage
        )

    def entry_cost(self, quantity, price):

        self.validate_positive(
            quantity,
            price
        )

        return quantity * price * (
            1 + self.fee
        )

    def exit_proceeds(self, quantity, price):

        self.validate_positive(
            quantity,
            price
        )

        return quantity * price * (
            1 - self.fee
        )

    def calculate_pnl(
        self,
        quantity,
        entry_price,
        exit_price
    ):

        return (
            self.exit_proceeds(
                quantity,
                exit_price
            )
            -
            self.entry_cost(
                quantity,
                entry_price
            )
        )

    def evaluate_long_exit(
        self,
        opening,
        high,
        low,
        close,
        stop,
        target,
        holding_days,
        maximum_holding_days,
        final_session=False,
        forced_exit=False
    ):

        self.validate_positive(
            opening,
            high,
            low,
            close,
            stop,
            target
        )

        if (
            high < max(opening, close)
            or low > min(opening, close)
            or low > high
        ):

            raise ValueError(
                "Invalid OHLC candle."
            )

        if stop >= target:

            raise ValueError(
                "Stop must be below target."
            )

        if (
            holding_days < 1
            or maximum_holding_days < 1
        ):

            raise ValueError(
                "Invalid holding period."
            )

        # Overnight gap below stop.

        if opening <= stop:

            return ExitDecision(
                True,
                opening,
                "STOP_GAP"
            )

        # Overnight gap above target.
        # Conservative target-price assumption.

        if opening >= target:

            return ExitDecision(
                True,
                target,
                "TAKE_PROFIT_GAP"
            )

        # Previously scheduled risk liquidation.

        if forced_exit:

            return ExitDecision(
                True,
                opening,
                "DRAWDOWN_EXIT"
            )

        # Conservative stop-first assumption
        # when both levels occur in one candle.

        if low <= stop:

            return ExitDecision(
                True,
                stop,
                "STOP_LOSS"
            )

        if high >= target:

            return ExitDecision(
                True,
                target,
                "TAKE_PROFIT"
            )

        if holding_days >= maximum_holding_days:

            return ExitDecision(
                True,
                close,
                "TIME_EXIT"
            )

        if final_session:

            return ExitDecision(
                True,
                close,
                "END_OF_BACKTEST"
            )

        return ExitDecision(False)