from pathlib import Path
from copy import deepcopy
from datetime import date
from decimal import Decimal

import math
import re
import yaml

from risk.risk_config import RiskConfig


BASE_DIR = Path(__file__).resolve().parents[1]

CONFIG_FILE = BASE_DIR / "config" / "portfolio.yaml"


class PortfolioConfig:

    REQUIRED_SECTIONS = {

        "name",
        "version",
        "account",
        "historical_data",
        "universe",
        "risk",
        "execution",
        "opportunity_selection",
        "strategy",
        "benchmarks",
        "reporting",
        "safety"

    }

    def __init__(self, config=None, risk_config=None):

        if config is None:

            with open(
                CONFIG_FILE,
                "r",
                encoding="utf-8"
            ) as file:

                document = yaml.safe_load(file)

            if not isinstance(document, dict):

                raise ValueError(
                    "Invalid portfolio configuration document."
                )

            if "portfolio" not in document:

                raise ValueError(
                    "Missing portfolio configuration."
                )

            config = document["portfolio"]

        self._config = deepcopy(config)

        self.risk_config = (
            risk_config
            if risk_config is not None
            else RiskConfig.load()
        )

        self.validate()


    @property
    def config(self):

        return deepcopy(self._config)


    def get_section(self, name):

        return deepcopy(self._config[name])


    @staticmethod
    def require_keys(section, keys, name):

        if not isinstance(section, dict):

            raise ValueError(
                f"{name} must be a dictionary."
            )

        missing = set(keys) - set(section)

        if missing:

            raise ValueError(
                f"{name} missing keys: {sorted(missing)}"
            )


    @staticmethod
    def validate_number(
        value,
        name,
        minimum=None,
        maximum=None,
        strictly_positive=False
    ):

        if isinstance(value, bool):

            raise ValueError(
                f"{name} must be numeric."
            )

        if not isinstance(
            value,
            (int, float, Decimal)
        ):

            raise ValueError(
                f"{name} must be numeric."
            )

        value = float(value)

        if not math.isfinite(value):

            raise ValueError(
                f"{name} must be finite."
            )

        if strictly_positive and value <= 0:

            raise ValueError(
                f"{name} must be greater than zero."
            )

        if minimum is not None and value < minimum:

            raise ValueError(
                f"{name} must be at least {minimum}."
            )

        if maximum is not None and value > maximum:

            raise ValueError(
                f"{name} cannot exceed {maximum}."
            )

        return value


    @staticmethod
    def validate_integer(
        value,
        name,
        minimum=1,
        maximum=None
    ):

        if type(value) is not int:

            raise ValueError(
                f"{name} must be an integer."
            )

        if value < minimum:

            raise ValueError(
                f"{name} must be at least {minimum}."
            )

        if maximum is not None and value > maximum:

            raise ValueError(
                f"{name} cannot exceed {maximum}."
            )

        return value


    @staticmethod
    def validate_boolean(value, name):

        if type(value) is not bool:

            raise ValueError(
                f"{name} must be a boolean."
            )

        return value


    @staticmethod
    def validate_string(value, name):

        if not isinstance(value, str):

            raise ValueError(
                f"{name} must be a string."
            )

        if not value.strip():

            raise ValueError(
                f"{name} cannot be empty."
            )

        return value


    @classmethod
    def require_boolean(cls, value, expected, name):

        cls.validate_boolean(value, name)

        if value is not expected:

            raise ValueError(
                f"{name} must be {expected}."
            )


    @staticmethod
    def validate_symbols(symbols, name):

        if not isinstance(symbols, list):

            raise ValueError(
                f"{name} must be a list."
            )

        if not symbols:

            raise ValueError(
                f"{name} cannot be empty."
            )

        for symbol in symbols:

            if not isinstance(symbol, str):

                raise ValueError(
                    f"Invalid symbol in {name}."
                )

            if not re.fullmatch(
                r"[A-Z][A-Z0-9.-]{0,9}",
                symbol
            ):

                raise ValueError(
                    f"Invalid symbol: {symbol}"
                )

        if len(symbols) != len(set(symbols)):

            raise ValueError(
                f"Duplicate symbols in {name}."
            )


    @staticmethod
    def validate_date(value, name):

        if not isinstance(value, str):

            raise ValueError(
                f"{name} must be a date string."
            )

        if not re.fullmatch(
            r"\d{4}-\d{2}-\d{2}",
            value
        ):

            raise ValueError(
                f"{name} must use YYYY-MM-DD."
            )

        try:

            return date.fromisoformat(value)

        except ValueError as error:

            raise ValueError(
                f"Invalid date: {name}"
            ) from error


    def validate_structure(self):

        self.require_keys(

            self._config,

            self.REQUIRED_SECTIONS,

            "Portfolio configuration"

        )

        self.validate_string(
            self._config["name"],
            "name"
        )

        self.validate_string(
            self._config["version"],
            "version"
        )


    def validate_account(self):

        account = self._config["account"]

        self.require_keys(

            account,

            [

                "initial_capital",
                "currency",
                "fractional_shares",
                "quantity_precision",
                "monthly_contribution",
                "simulate_contributions",
                "margin_enabled",
                "short_selling_enabled"

            ],

            "Account"

        )

        self.validate_number(

            account["initial_capital"],

            "initial_capital",

            strictly_positive=True

        )

        self.validate_number(

            account["monthly_contribution"],

            "monthly_contribution",

            minimum=0

        )

        if account["currency"] != "USD":

            raise ValueError(
                "Only USD accounts are supported."
            )

        self.require_boolean(
            account["fractional_shares"],
            True,
            "fractional_shares"
        )

        self.validate_integer(

            account["quantity_precision"],

            "quantity_precision",

            minimum=1,

            maximum=6

        )

        self.require_boolean(
            account["simulate_contributions"],
            False,
            "simulate_contributions"
        )

        self.require_boolean(
            account["margin_enabled"],
            False,
            "margin_enabled"
        )

        self.require_boolean(
            account["short_selling_enabled"],
            False,
            "short_selling_enabled"
        )


    def validate_historical_data(self):

        historical = self._config["historical_data"]

        self.require_keys(

            historical,

            [

                "historical_days",
                "warmup_candles",
                "start_date",
                "end_date",
                "timeframe",
                "require_aligned_sessions"

            ],

            "Historical data"

        )

        self.validate_integer(

            historical["historical_days"],

            "historical_days",

            minimum=365

        )

        self.validate_integer(

            historical["warmup_candles"],

            "warmup_candles",

            minimum=200

        )

        start = self.validate_date(

            historical["start_date"],

            "start_date"

        )

        end = self.validate_date(

            historical["end_date"],

            "end_date"

        )

        if start >= end:

            raise ValueError(
                "Historical start date must precede end date."
            )

        if historical["timeframe"] != "DAY":

            raise ValueError(
                "Only daily candles are currently supported."
            )

        self.require_boolean(

            historical["require_aligned_sessions"],

            True,

            "require_aligned_sessions"

        )


    def validate_universe(self):

        universe = self._config["universe"]

        self.require_keys(

            universe,

            ["symbols"],

            "Universe"

        )

        self.validate_symbols(

            universe["symbols"],

            "Trading universe"

        )


    def validate_risk(self):

        risk = self._config["risk"]

        percentages = [

            "risk_per_trade_percent",
            "maximum_daily_loss_percent",
            "maximum_drawdown_percent",
            "maximum_position_percent",
            "maximum_total_exposure_percent",
            "maximum_aggregate_open_risk_percent"

        ]

        required = percentages + [

            "maximum_open_positions",
            "minimum_order_value",
            "use_starting_capital_ceiling",
            "allow_duplicate_positions",
            "allow_pyramiding",
            "halt_new_entries_on_daily_loss",
            "halt_on_maximum_drawdown",
            "liquidate_on_drawdown_breach"

        ]

        self.require_keys(

            risk,

            required,

            "Risk"

        )

        for name in percentages:

            self.validate_number(

                risk[name],

                name,

                strictly_positive=True,

                maximum=100

            )

        self.validate_integer(

            risk["maximum_open_positions"],

            "maximum_open_positions"

        )

        self.validate_number(

            risk["minimum_order_value"],

            "minimum_order_value",

            strictly_positive=True

        )

        if (

            risk["risk_per_trade_percent"]

            > risk["maximum_aggregate_open_risk_percent"]

        ):

            raise ValueError(
                "Per-trade risk exceeds aggregate open risk."
            )

        if (

            risk["maximum_aggregate_open_risk_percent"]

            > risk["maximum_daily_loss_percent"]

        ):

            raise ValueError(
                "Aggregate open risk exceeds daily loss threshold."
            )

        if (

            risk["maximum_daily_loss_percent"]

            > risk["maximum_drawdown_percent"]

        ):

            raise ValueError(
                "Daily loss threshold exceeds drawdown threshold."
            )

        if (

            risk["maximum_position_percent"]

            > risk["maximum_total_exposure_percent"]

        ):

            raise ValueError(
                "Position allocation exceeds total exposure."
            )

        capital = self._config["account"]["initial_capital"]

        maximum_position = (

            capital

            * risk["maximum_position_percent"]

            / 100

        )

        if risk["minimum_order_value"] > maximum_position:

            raise ValueError(
                "Minimum order exceeds maximum position value."
            )

        required_true = [

            "use_starting_capital_ceiling",
            "halt_new_entries_on_daily_loss",
            "halt_on_maximum_drawdown",
            "liquidate_on_drawdown_breach"

        ]

        required_false = [

            "allow_duplicate_positions",
            "allow_pyramiding"

        ]

        for name in required_true:

            self.require_boolean(
                risk[name],
                True,
                name
            )

        for name in required_false:

            self.require_boolean(
                risk[name],
                False,
                name
            )


    def validate_execution(self):

        execution = self._config["execution"]

        self.require_keys(

            execution,

            [

                "slippage_percent",
                "transaction_fee_percent",
                "entry_timing",
                "stop_target_conflict",
                "maximum_holding_sessions",
                "liquidate_at_end",
                "allow_same_session_reentry"

            ],

            "Execution"

        )

        for name in [

            "slippage_percent",
            "transaction_fee_percent"

        ]:

            self.validate_number(

                execution[name],

                name,

                minimum=0,

                maximum=10

            )

        if execution["entry_timing"] != "NEXT_SESSION_OPEN":

            raise ValueError(
                "Unsupported entry timing."
            )

        if execution["stop_target_conflict"] != "STOP_FIRST":

            raise ValueError(
                "Unsupported stop-target conflict policy."
            )

        self.validate_integer(

            execution["maximum_holding_sessions"],

            "maximum_holding_sessions"

        )

        self.require_boolean(

            execution["liquidate_at_end"],

            True,

            "liquidate_at_end"

        )

        self.require_boolean(

            execution["allow_same_session_reentry"],

            False,

            "allow_same_session_reentry"

        )


    def validate_opportunity_selection(self):

        selection = self._config["opportunity_selection"]

        self.require_keys(

            selection,

            [

                "minimum_score",
                "ranking_method",
                "ranking_direction",
                "tie_breaker",
                "maximum_new_positions_per_session"

            ],

            "Opportunity selection"

        )

        self.validate_number(

            selection["minimum_score"],

            "minimum_score",

            minimum=0,

            maximum=100

        )

        if selection["ranking_method"] != "OPPORTUNITY_SCORE":

            raise ValueError(
                "Unsupported ranking method."
            )

        if selection["ranking_direction"] != "DESCENDING":

            raise ValueError(
                "Unsupported ranking direction."
            )

        if selection["tie_breaker"] != "SYMBOL_ASCENDING":

            raise ValueError(
                "Unsupported tie-breaking method."
            )

        self.validate_integer(

            selection["maximum_new_positions_per_session"],

            "maximum_new_positions_per_session",

            maximum=self._config["risk"][
                "maximum_open_positions"
            ]

        )


    def validate_strategy(self):

        strategy = self._config["strategy"]

        self.require_keys(

            strategy,

            [

                "momentum_enabled",
                "breakout_enabled",
                "pullback_enabled",
                "stop_atr_multiplier",
                "risk_reward_ratio",
                "research_only"

            ],

            "Strategy"

        )

        for name in [

            "momentum_enabled",
            "breakout_enabled",
            "pullback_enabled"

        ]:

            self.validate_boolean(
                strategy[name],
                name
            )

        self.validate_number(

            strategy["stop_atr_multiplier"],

            "stop_atr_multiplier",

            strictly_positive=True

        )

        self.validate_number(

            strategy["risk_reward_ratio"],

            "risk_reward_ratio",

            minimum=1

        )

        self.require_boolean(

            strategy["research_only"],

            True,

            "research_only"

        )


    def validate_benchmarks(self):

        benchmarks = self._config["benchmarks"]

        self.require_keys(

            benchmarks,

            [

                "enabled",
                "symbols",
                "include_cash_benchmark"

            ],

            "Benchmarks"

        )

        self.validate_boolean(

            benchmarks["enabled"],

            "benchmarks.enabled"

        )

        self.validate_boolean(

            benchmarks["include_cash_benchmark"],

            "include_cash_benchmark"

        )

        self.validate_symbols(

            benchmarks["symbols"],

            "Benchmark symbols"

        )


    def validate_reporting(self):

        reporting = self._config["reporting"]

        fields = [

            "save_trades",
            "save_equity_curve",
            "save_daily_snapshots",
            "save_rejected_opportunities",
            "save_risk_events",
            "save_position_history"

        ]

        self.require_keys(

            reporting,

            fields,

            "Reporting"

        )

        for name in fields:

            self.require_boolean(

                reporting[name],

                True,

                name

            )


    def validate_safety(self):

        safety = self._config["safety"]

        required_true = [

            "historical_simulation_only",
            "require_forward_validation",
            "fail_on_missing_market_data",
            "fail_on_accounting_mismatch"

        ]

        required_false = [

            "live_trading_enabled",
            "broker_order_submission_enabled",
            "automatic_strategy_activation"

        ]

        self.require_keys(

            safety,

            required_true + required_false,

            "Safety"

        )

        for name in required_true:

            self.require_boolean(

                safety[name],

                True,

                name

            )

        for name in required_false:

            self.require_boolean(

                safety[name],

                False,

                name

            )


    def validate_risk_synchronization(self):

        risk = self._config["risk"]

        account = self._config["account"]

        guardian = self.risk_config

        comparisons = {

            "initial_capital": (

                account["initial_capital"],

                guardian.starting_capital

            ),

            "risk_per_trade_percent": (

                risk["risk_per_trade_percent"],

                guardian.risk_per_trade_percent

            ),

            "maximum_daily_loss_percent": (

                risk["maximum_daily_loss_percent"],

                guardian.max_daily_loss_percent

            ),

            "maximum_drawdown_percent": (

                risk["maximum_drawdown_percent"],

                guardian.max_drawdown_percent

            ),

            "maximum_position_percent": (

                risk["maximum_position_percent"],

                guardian.max_position_percent

            ),

            "maximum_total_exposure_percent": (

                risk["maximum_total_exposure_percent"],

                guardian.max_total_exposure_percent

            ),

            "maximum_open_positions": (

                risk["maximum_open_positions"],

                guardian.max_open_positions

            ),

            "minimum_order_value": (

                risk["minimum_order_value"],

                guardian.min_order_value

            )

        }

        for name, values in comparisons.items():

            portfolio_value, guardian_value = values

            if Decimal(str(portfolio_value)) != Decimal(
                str(guardian_value)
            ):

                raise ValueError(

                    f"Risk configuration conflict: {name}. "

                    f"Portfolio={portfolio_value}, "

                    f"Risk Guardian={guardian_value}"

                )

        if guardian.trading_mode != "paper":

            raise ValueError(
                "Risk Guardian must remain in paper mode."
            )

        if not guardian.enabled:

            raise ValueError(
                "Risk Guardian must remain enabled."
            )

        if guardian.emergency_stop:

            raise ValueError(
                "Emergency stop is active."
            )

        if guardian.live_trading_enabled:

            raise ValueError(
                "Live trading must remain disabled."
            )

        if guardian.margin_enabled:

            raise ValueError(
                "Margin must remain disabled."
            )

        if guardian.short_selling_enabled:

            raise ValueError(
                "Short selling must remain disabled."
            )


    def validate(self):

        self.validate_structure()

        self.validate_account()

        self.validate_historical_data()

        self.validate_universe()

        self.validate_risk()

        self.validate_execution()

        self.validate_opportunity_selection()

        self.validate_strategy()

        self.validate_benchmarks()

        self.validate_reporting()

        self.validate_safety()

        self.validate_risk_synchronization()

        return True


    def display_summary(self):

        config = self._config

        account = config["account"]

        risk = config["risk"]

        capital = account["initial_capital"]

        print("=" * 80)

        print("JARVIS PORTFOLIO CONFIGURATION VALIDATOR")

        print("VERSION 0.8.6.2")

        print("=" * 80)

        print("\nACCOUNT")

        print(f"Portfolio: {config['name']}")

        print(f"Starting capital: ${capital:.2f}")

        print(f"Currency: {account['currency']}")

        print(
            "Trading universe:",
            len(config["universe"]["symbols"])
        )

        print("\nPORTFOLIO RISK LIMITS")

        print(
            "Maximum positions:",
            risk["maximum_open_positions"]
        )

        print(
            "Maximum position value:",
            f"${capital * risk['maximum_position_percent'] / 100:.2f}"
        )

        print(
            "Maximum total exposure:",
            f"${capital * risk['maximum_total_exposure_percent'] / 100:.2f}"
        )

        print(
            "Maximum planned risk per trade:",
            f"${capital * risk['risk_per_trade_percent'] / 100:.2f}"
        )

        print(
            "Maximum aggregate planned open risk:",
            f"${capital * risk['maximum_aggregate_open_risk_percent'] / 100:.2f}"
        )

        print(
            "Initial daily loss threshold:",
            f"${capital * risk['maximum_daily_loss_percent'] / 100:.2f}"
        )

        print(
            "Initial drawdown threshold:",
            f"${capital * risk['maximum_drawdown_percent'] / 100:.2f}"
        )

        print("\nVALIDATION")

        print("Configuration structure: PASSED")

        print("Account validation: PASSED")

        print("Historical configuration: PASSED")

        print("Trading universe: PASSED")

        print("Risk validation: PASSED")

        print("Execution validation: PASSED")

        print("Opportunity selection: PASSED")

        print("Strategy validation: PASSED")

        print("Benchmark validation: PASSED")

        print("Reporting validation: PASSED")

        print("Safety validation: PASSED")

        print("Risk Guardian synchronization: PASSED")

        print("\nSAFETY STATUS")

        print("Historical simulation only: ENABLED")

        print("Risk Guardian: ENABLED")

        print("Margin: DISABLED")

        print("Short selling: DISABLED")

        print("Live trading: DISABLED")

        print("Broker order submission: DISABLED")

        print("Automatic strategy activation: DISABLED")

        print("\nPORTFOLIO CONFIGURATION VALIDATED")


if __name__ == "__main__":

    PortfolioConfig().display_summary()