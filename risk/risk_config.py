import os
from pathlib import Path
from decimal import Decimal
from dataclasses import dataclass

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[1]

load_dotenv(BASE_DIR / ".env", override=True)


def get_decimal(name, default):

    value = Decimal(os.getenv(name, default))

    if not value.is_finite() or value <= 0:
        raise ValueError(f"Invalid configuration: {name}")

    return value


def get_bool(name, default="false"):

    value = os.getenv(name, default).strip().lower()

    if value not in {"true", "false"}:
        raise ValueError(f"Invalid boolean: {name}")

    return value == "true"


@dataclass(frozen=True)
class RiskConfig:

    starting_capital: Decimal
    max_daily_loss_percent: Decimal
    max_drawdown_percent: Decimal
    risk_per_trade_percent: Decimal
    max_position_percent: Decimal
    max_total_exposure_percent: Decimal
    min_order_value: Decimal
    max_open_positions: int

    enabled: bool
    emergency_stop: bool
    live_trading_enabled: bool
    margin_enabled: bool
    short_selling_enabled: bool

    trading_mode: str

    @classmethod
    def load(cls):

        config = cls(

            starting_capital=get_decimal(
                "STARTING_CAPITAL", "100"
            ),

            max_daily_loss_percent=get_decimal(
                "MAX_DAILY_LOSS_PERCENT", "1"
            ),

            max_drawdown_percent=get_decimal(
                "MAX_DRAWDOWN_PERCENT", "5"
            ),

            risk_per_trade_percent=get_decimal(
                "RISK_PER_TRADE_PERCENT", "0.5"
            ),

            max_position_percent=get_decimal(
                "MAX_POSITION_PERCENT", "20"
            ),

            max_total_exposure_percent=get_decimal(
                "MAX_TOTAL_EXPOSURE_PERCENT", "60"
            ),

            min_order_value=get_decimal(
                "MIN_ORDER_VALUE", "1"
            ),

            max_open_positions=int(
                os.getenv("MAX_OPEN_POSITIONS", "3")
            ),

            enabled=get_bool(
                "RISK_GUARDIAN_ENABLED", "true"
            ),

            emergency_stop=get_bool(
                "EMERGENCY_STOP"
            ),

            live_trading_enabled=get_bool(
                "LIVE_TRADING_ENABLED"
            ),

            margin_enabled=get_bool(
                "MARGIN_ENABLED"
            ),

            short_selling_enabled=get_bool(
                "SHORT_SELLING_ENABLED"
            ),

            trading_mode=os.getenv(
                "TRADING_MODE", "paper"
            ).lower()
        )

        config.validate()

        return config

    def validate(self):

        percentages = [

            self.max_daily_loss_percent,
            self.max_drawdown_percent,
            self.risk_per_trade_percent,
            self.max_position_percent,
            self.max_total_exposure_percent

        ]

        if any(value > 100 for value in percentages):
            raise ValueError("Invalid risk percentage.")

        if self.max_open_positions < 1:
            raise ValueError("Invalid position limit.")

        if self.risk_per_trade_percent > self.max_daily_loss_percent:
            raise ValueError(
                "Trade risk exceeds daily loss limit."
            )

        if self.max_position_percent > self.max_total_exposure_percent:
            raise ValueError(
                "Position limit exceeds exposure limit."
            )