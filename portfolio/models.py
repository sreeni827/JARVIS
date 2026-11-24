from dataclasses import dataclass
from datetime import date
from decimal import Decimal, ROUND_DOWN
import re


ZERO = Decimal("0")
ONE = Decimal("1")
HUNDRED = Decimal("100")

QUANTITY_STEP = Decimal("0.000001")


# ============================================================
# VALIDATION UTILITIES
# ============================================================


def to_decimal(value, name="value"):

    if isinstance(value, bool):

        raise ValueError(
            f"{name} must be numeric."
        )

    if not isinstance(
        value,
        (Decimal, int, float, str)
    ):

        raise ValueError(
            f"{name} must be numeric."
        )

    try:

        result = Decimal(str(value))

    except Exception as error:

        raise ValueError(
            f"Invalid decimal: {name}"
        ) from error

    if not result.is_finite():

        raise ValueError(
            f"{name} must be finite."
        )

    return result


def positive_decimal(value, name):

    result = to_decimal(value, name)

    if result <= ZERO:

        raise ValueError(
            f"{name} must be greater than zero."
        )

    return result


def nonnegative_decimal(value, name):

    result = to_decimal(value, name)

    if result < ZERO:

        raise ValueError(
            f"{name} cannot be negative."
        )

    return result


def validate_symbol(symbol):

    if not isinstance(symbol, str):

        raise ValueError(
            "Symbol must be a string."
        )

    if not re.fullmatch(
        r"[A-Z][A-Z0-9.-]{0,9}",
        symbol
    ):

        raise ValueError(
            f"Invalid symbol: {symbol}"
        )

    return symbol


def validate_text(value, name):

    if not isinstance(value, str):

        raise ValueError(
            f"{name} must be a string."
        )

    if not value.strip():

        raise ValueError(
            f"{name} cannot be empty."
        )

    return value


def validate_session(value, name):

    if type(value) is not date:

        raise ValueError(
            f"{name} must be a date."
        )

    return value


def validate_integer(value, name, minimum=0):

    if type(value) is not int:

        raise ValueError(
            f"{name} must be an integer."
        )

    if value < minimum:

        raise ValueError(
            f"{name} must be at least {minimum}."
        )

    return value


def validate_quantity(value):

    quantity = positive_decimal(
        value,
        "quantity"
    )

    rounded = quantity.quantize(
        QUANTITY_STEP,
        rounding=ROUND_DOWN
    )

    if quantity != rounded:

        raise ValueError(
            "Quantity exceeds six decimal places."
        )

    return quantity


def validate_score(value):

    score = nonnegative_decimal(
        value,
        "score"
    )

    if score > HUNDRED:

        raise ValueError(
            "Score cannot exceed 100."
        )

    return score


def normalize_fields(
    instance,
    positive=(),
    nonnegative=()
):

    for name in positive:

        object.__setattr__(

            instance,

            name,

            positive_decimal(
                getattr(instance, name),
                name
            )

        )

    for name in nonnegative:

        object.__setattr__(

            instance,

            name,

            nonnegative_decimal(
                getattr(instance, name),
                name
            )

        )


# ============================================================
# OPEN PORTFOLIO POSITION
# ============================================================


@dataclass(frozen=True, slots=True)
class PortfolioPosition:

    symbol: str

    strategy: str

    entry_date: date

    quantity: Decimal

    entry_price: Decimal

    stop_price: Decimal

    target_price: Decimal

    current_price: Decimal

    planned_risk: Decimal

    entry_fee: Decimal = ZERO

    holding_sessions: int = 1

    def __post_init__(self):

        validate_symbol(self.symbol)

        validate_text(
            self.strategy,
            "strategy"
        )

        validate_session(
            self.entry_date,
            "entry_date"
        )

        object.__setattr__(
            self,
            "quantity",
            validate_quantity(self.quantity)
        )

        normalize_fields(

            self,

            positive=(

                "entry_price",
                "stop_price",
                "target_price",
                "current_price",
                "planned_risk"

            ),

            nonnegative=("entry_fee",)

        )

        if not (
            self.stop_price
            < self.entry_price
            < self.target_price
        ):

            raise ValueError(
                "Long position requires stop < entry < target."
            )

        validate_integer(

            self.holding_sessions,

            "holding_sessions",

            minimum=1

        )

    @property
    def entry_notional(self):

        return (
            self.quantity
            * self.entry_price
        )

    @property
    def entry_cost(self):

        return (
            self.entry_notional
            + self.entry_fee
        )

    @property
    def market_value(self):

        return (
            self.quantity
            * self.current_price
        )

    @property
    def unrealized_pnl(self):

        return (
            self.market_value
            - self.entry_cost
        )

    @property
    def unrealized_return_percent(self):

        return (
            self.unrealized_pnl
            / self.entry_cost
            * HUNDRED
        )

    @property
    def risk_reward_ratio(self):

        return (

            (self.target_price - self.entry_price)

            /

            (self.entry_price - self.stop_price)

        )


# ============================================================
# COMPLETED PORTFOLIO TRADE
# ============================================================


@dataclass(frozen=True, slots=True)
class PortfolioTrade:

    symbol: str

    strategy: str

    entry_date: date

    exit_date: date

    quantity: Decimal

    entry_price: Decimal

    exit_price: Decimal

    stop_price: Decimal

    target_price: Decimal

    planned_risk: Decimal

    exit_reason: str

    entry_fee: Decimal = ZERO

    exit_fee: Decimal = ZERO

    holding_sessions: int = 1

    def __post_init__(self):

        validate_symbol(self.symbol)

        validate_text(
            self.strategy,
            "strategy"
        )

        validate_text(
            self.exit_reason,
            "exit_reason"
        )

        validate_session(
            self.entry_date,
            "entry_date"
        )

        validate_session(
            self.exit_date,
            "exit_date"
        )

        if self.exit_date < self.entry_date:

            raise ValueError(
                "Exit date cannot precede entry date."
            )

        object.__setattr__(

            self,

            "quantity",

            validate_quantity(self.quantity)

        )

        normalize_fields(

            self,

            positive=(

                "entry_price",
                "exit_price",
                "stop_price",
                "target_price",
                "planned_risk"

            ),

            nonnegative=(

                "entry_fee",
                "exit_fee"

            )

        )

        if not (
            self.stop_price
            < self.entry_price
            < self.target_price
        ):

            raise ValueError(
                "Invalid long trade price structure."
            )

        if self.exit_fee > self.exit_notional:

            raise ValueError(
                "Exit fee exceeds sale proceeds."
            )

        validate_integer(

            self.holding_sessions,

            "holding_sessions",

            minimum=1

        )

    @property
    def entry_notional(self):

        return self.quantity * self.entry_price

    @property
    def exit_notional(self):

        return self.quantity * self.exit_price

    @property
    def entry_cost(self):

        return self.entry_notional + self.entry_fee

    @property
    def exit_proceeds(self):

        return self.exit_notional - self.exit_fee

    @property
    def total_fees(self):

        return self.entry_fee + self.exit_fee

    @property
    def pnl(self):

        return self.exit_proceeds - self.entry_cost

    @property
    def return_percent(self):

        return (
            self.pnl
            / self.entry_cost
            * HUNDRED
        )

    @property
    def actual_loss(self):

        return max(
            ZERO,
            -self.pnl
        )

    @property
    def risk_multiple(self):

        return self.pnl / self.planned_risk

    @property
    def risk_breached(self):

        return (
            self.actual_loss
            > self.planned_risk
        )


# ============================================================
# SHARED PORTFOLIO ACCOUNT STATE
# ============================================================


@dataclass(frozen=True, slots=True)
class PortfolioAccountState:

    initial_capital: Decimal

    cash: Decimal

    day_start_equity: Decimal

    peak_equity: Decimal

    positions: tuple = ()

    realized_pnl: Decimal = ZERO

    trading_halted: bool = False

    daily_entry_halted: bool = False

    def __post_init__(self):

        normalize_fields(

            self,

            positive=(

                "initial_capital",
                "day_start_equity",
                "peak_equity"

            ),

            nonnegative=("cash",)

        )

        object.__setattr__(

            self,

            "realized_pnl",

            to_decimal(
                self.realized_pnl,
                "realized_pnl"
            )

        )

        if not isinstance(
            self.positions,
            (tuple, list)
        ):

            raise ValueError(
                "Positions must be a tuple or list."
            )

        positions = tuple(self.positions)

        for position in positions:

            if not isinstance(
                position,
                PortfolioPosition
            ):

                raise ValueError(
                    "Invalid portfolio position."
                )

        symbols = [
            position.symbol
            for position in positions
        ]

        if len(symbols) != len(set(symbols)):

            raise ValueError(
                "Duplicate portfolio positions."
            )

        object.__setattr__(
            self,
            "positions",
            positions
        )

        if self.peak_equity < self.equity:

            raise ValueError(
                "Peak equity cannot be below current equity."
            )

        if self.peak_equity < self.day_start_equity:

            raise ValueError(
                "Peak equity cannot be below day-start equity."
            )

        expected_equity = (

            self.initial_capital

            + self.realized_pnl

            + self.unrealized_pnl

        )

        if self.equity != expected_equity:

            raise ValueError(
                "Portfolio accounting mismatch."
            )

        for name in [

            "trading_halted",
            "daily_entry_halted"

        ]:

            if type(getattr(self, name)) is not bool:

                raise ValueError(
                    f"{name} must be boolean."
                )

    @property
    def market_value(self):

        return sum(

            (
                position.market_value
                for position in self.positions
            ),

            ZERO

        )

    @property
    def equity(self):

        return self.cash + self.market_value

    @property
    def open_positions(self):

        return len(self.positions)

    @property
    def existing_symbols(self):

        return tuple(

            position.symbol

            for position in self.positions

        )

    @property
    def total_exposure(self):

        return self.market_value

    @property
    def exposure_percent(self):

        if self.equity <= ZERO:

            return ZERO

        return (
            self.total_exposure
            / self.equity
            * HUNDRED
        )

    @property
    def aggregate_open_risk(self):

        return sum(

            (
                position.planned_risk
                for position in self.positions
            ),

            ZERO

        )

    @property
    def unrealized_pnl(self):

        return sum(

            (
                position.unrealized_pnl
                for position in self.positions
            ),

            ZERO

        )

    @property
    def total_pnl(self):

        return (
            self.realized_pnl
            + self.unrealized_pnl
        )

    @property
    def daily_loss(self):

        return max(

            ZERO,

            self.day_start_equity - self.equity

        )

    @property
    def daily_loss_percent(self):

        return (

            self.daily_loss

            / self.day_start_equity

            * HUNDRED

        )

    @property
    def drawdown_percent(self):

        return (

            (self.peak_equity - self.equity)

            / self.peak_equity

            * HUNDRED

        )

    @property
    def return_percent(self):

        return (

            (self.equity - self.initial_capital)

            / self.initial_capital

            * HUNDRED

        )


# ============================================================
# DAILY PORTFOLIO SNAPSHOT
# ============================================================


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:

    session: date

    account: PortfolioAccountState

    entries: int = 0

    exits: int = 0

    rejected_opportunities: int = 0

    def __post_init__(self):

        validate_session(
            self.session,
            "session"
        )

        if not isinstance(
            self.account,
            PortfolioAccountState
        ):

            raise ValueError(
                "Invalid portfolio account state."
            )

        for name in [

            "entries",
            "exits",
            "rejected_opportunities"

        ]:

            validate_integer(
                getattr(self, name),
                name
            )

    def to_record(self):

        account = self.account

        return {

            "session": self.session.isoformat(),

            "cash": str(account.cash),

            "market_value": str(
                account.market_value
            ),

            "equity": str(account.equity),

            "realized_pnl": str(
                account.realized_pnl
            ),

            "unrealized_pnl": str(
                account.unrealized_pnl
            ),

            "open_positions": account.open_positions,

            "total_exposure": str(
                account.total_exposure
            ),

            "exposure_percent": str(
                account.exposure_percent
            ),

            "aggregate_open_risk": str(
                account.aggregate_open_risk
            ),

            "daily_loss_percent": str(
                account.daily_loss_percent
            ),

            "drawdown_percent": str(
                account.drawdown_percent
            ),

            "return_percent": str(
                account.return_percent
            ),

            "entries": self.entries,

            "exits": self.exits,

            "rejected_opportunities":
                self.rejected_opportunities,

            "trading_halted":
                account.trading_halted,

            "daily_entry_halted":
                account.daily_entry_halted

        }


# ============================================================
# PORTFOLIO OPPORTUNITY
# ============================================================


@dataclass(frozen=True, slots=True)
class PortfolioOpportunity:

    symbol: str

    strategy: str

    signal_date: date

    score: Decimal

    reference_price: Decimal

    atr: Decimal

    stop_atr_multiplier: Decimal

    risk_reward_ratio: Decimal

    def __post_init__(self):

        validate_symbol(self.symbol)

        validate_text(
            self.strategy,
            "strategy"
        )

        validate_session(
            self.signal_date,
            "signal_date"
        )

        object.__setattr__(

            self,

            "score",

            validate_score(self.score)

        )

        normalize_fields(

            self,

            positive=(

                "reference_price",
                "atr",
                "stop_atr_multiplier",
                "risk_reward_ratio"

            )

        )

        if self.risk_reward_ratio < ONE:

            raise ValueError(
                "Risk-reward ratio must be at least 1."
            )

    @property
    def stop_distance(self):

        return (
            self.atr
            * self.stop_atr_multiplier
        )

    def prices_for_entry(self, entry_price):

        entry = positive_decimal(

            entry_price,

            "entry_price"

        )

        stop = entry - self.stop_distance

        if stop <= ZERO:

            raise ValueError(
                "Calculated stop price is invalid."
            )

        target = (

            entry

            + self.stop_distance

            * self.risk_reward_ratio

        )

        return stop, target


# ============================================================
# PORTFOLIO RISK EVENT
# ============================================================


@dataclass(frozen=True, slots=True)
class PortfolioRiskEvent:

    session: date

    event_type: str

    reason: str

    observed_value: Decimal

    threshold: Decimal

    symbol: str | None = None

    def __post_init__(self):

        validate_session(
            self.session,
            "session"
        )

        validate_text(
            self.event_type,
            "event_type"
        )

        validate_text(
            self.reason,
            "reason"
        )

        normalize_fields(

            self,

            nonnegative=(

                "observed_value",
                "threshold"

            )

        )

        if self.symbol is not None:

            validate_symbol(self.symbol)


# ============================================================
# REJECTED PORTFOLIO OPPORTUNITY
# ============================================================


@dataclass(frozen=True, slots=True)
class PortfolioRejection:

    session: date

    symbol: str

    strategy: str

    score: Decimal

    reason: str

    def __post_init__(self):

        validate_session(
            self.session,
            "session"
        )

        validate_symbol(self.symbol)

        validate_text(
            self.strategy,
            "strategy"
        )

        validate_text(
            self.reason,
            "reason"
        )

        object.__setattr__(

            self,

            "score",

            validate_score(self.score)

        )


# ============================================================
# PORTFOLIO EXECUTION RESULT
# ============================================================


@dataclass(frozen=True, slots=True)
class PortfolioExecutionResult:

    executed: bool

    symbol: str

    reason: str

    position: PortfolioPosition | None = None

    trade: PortfolioTrade | None = None

    def __post_init__(self):

        if type(self.executed) is not bool:

            raise ValueError(
                "executed must be boolean."
            )

        validate_symbol(self.symbol)

        validate_text(
            self.reason,
            "reason"
        )

        if self.position is not None:

            if not isinstance(
                self.position,
                PortfolioPosition
            ):

                raise ValueError(
                    "Invalid execution position."
                )

            if self.position.symbol != self.symbol:

                raise ValueError(
                    "Execution position symbol mismatch."
                )

        if self.trade is not None:

            if not isinstance(
                self.trade,
                PortfolioTrade
            ):

                raise ValueError(
                    "Invalid execution trade."
                )

            if self.trade.symbol != self.symbol:

                raise ValueError(
                    "Execution trade symbol mismatch."
                )

        records = (

            int(self.position is not None)

            + int(self.trade is not None)

        )

        if self.executed and records != 1:

            raise ValueError(
                "Successful execution requires one record."
            )

        if not self.executed and records != 0:

            raise ValueError(
                "Rejected execution cannot contain records."
            )