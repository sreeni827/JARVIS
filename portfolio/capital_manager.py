from dataclasses import dataclass, replace
from decimal import Decimal, ROUND_DOWN
from collections.abc import Mapping

from portfolio.portfolio_config import PortfolioConfig

from portfolio.models import (
    PortfolioAccountState,
    PortfolioPosition,
    PortfolioTrade,
    PortfolioOpportunity,
    PortfolioExecutionResult,
    PortfolioRiskEvent,
    positive_decimal,
    nonnegative_decimal,
    validate_session
)


D = Decimal
ZERO = D("0")
HUNDRED = D("100")


# ============================================================
# CAPITAL ALLOCATION DECISION
# ============================================================


@dataclass(frozen=True, slots=True)
class CapitalAllocation:

    approved: bool

    reason: str

    quantity: Decimal = ZERO

    entry_price: Decimal = ZERO

    stop_price: Decimal = ZERO

    target_price: Decimal = ZERO

    entry_fee: Decimal = ZERO

    position_value: Decimal = ZERO

    entry_cost: Decimal = ZERO

    planned_risk: Decimal = ZERO


# ============================================================
# SHARED CAPITAL MANAGER
# ============================================================


class CapitalManager:

    def __init__(self, config=None):

        self._config = (
            config if config is not None
            else PortfolioConfig()
        )

        if not isinstance(
            self._config,
            PortfolioConfig
        ):

            raise ValueError(
                "Invalid portfolio configuration."
            )

        self._config.validate()

        settings = self._config.config

        risk = settings["risk"]

        execution = settings["execution"]

        account = settings["account"]

        self.initial_capital = D(
            str(account["initial_capital"])
        )

        self.maximum_positions = risk[
            "maximum_open_positions"
        ]

        self.maximum_position_percent = D(
            str(risk["maximum_position_percent"])
        )

        self.maximum_exposure_percent = D(
            str(risk["maximum_total_exposure_percent"])
        )

        self.risk_per_trade_percent = D(
            str(risk["risk_per_trade_percent"])
        )

        self.maximum_open_risk_percent = D(
            str(
                risk[
                    "maximum_aggregate_open_risk_percent"
                ]
            )
        )

        self.maximum_daily_loss_percent = D(
            str(risk["maximum_daily_loss_percent"])
        )

        self.maximum_drawdown_percent = D(
            str(risk["maximum_drawdown_percent"])
        )

        self.minimum_order_value = D(
            str(risk["minimum_order_value"])
        )

        self.slippage = D(
            str(execution["slippage_percent"])
        ) / HUNDRED

        self.fee_rate = D(
            str(execution["transaction_fee_percent"])
        ) / HUNDRED

        self.quantity_step = D("1").scaleb(
            -account["quantity_precision"]
        )

        self.symbols = frozenset(
            settings["universe"]["symbols"]
        )

        self.minimum_score = D(
            str(
                settings["opportunity_selection"][
                    "minimum_score"
                ]
            )
        )

        self.maximum_new_positions = settings[
            "opportunity_selection"
        ]["maximum_new_positions_per_session"]

        self.enabled_strategies = frozenset(
            name.upper()
            for name in (
                "momentum",
                "breakout",
                "pullback"
            )
            if settings["strategy"][
                f"{name}_enabled"
            ]
        )

        self.stop_atr_multiplier = D(
            str(
                settings["strategy"][
                    "stop_atr_multiplier"
                ]
            )
        )

        self.risk_reward_ratio = D(
            str(
                settings["strategy"][
                    "risk_reward_ratio"
                ]
            )
        )

        self._state = PortfolioAccountState(

            initial_capital=self.initial_capital,

            cash=self.initial_capital,

            day_start_equity=self.initial_capital,

            peak_equity=self.initial_capital

        )

        self._session = None

        self._trades = []

        self._risk_events = []

        self._closed_symbols = set()

        self._session_entries = 0

    # ========================================================
    # READ-ONLY ACCOUNT ACCESS
    # ========================================================

    @property
    def state(self):

        return self._state

    @property
    def session(self):

        return self._session

    @property
    def trades(self):

        return tuple(self._trades)

    @property
    def risk_events(self):

        return tuple(self._risk_events)

    @property
    def capital_base(self):

        return min(
            self.state.equity,
            self.initial_capital
        )

    @property
    def maximum_position_value(self):

        return (
            self.capital_base
            * self.maximum_position_percent
            / HUNDRED
        )

    @property
    def maximum_exposure(self):

        return (
            self.capital_base
            * self.maximum_exposure_percent
            / HUNDRED
        )

    @property
    def maximum_trade_risk(self):

        return (
            self.capital_base
            * self.risk_per_trade_percent
            / HUNDRED
        )

    @property
    def maximum_aggregate_risk(self):

        return (
            self.capital_base
            * self.maximum_open_risk_percent
            / HUNDRED
        )

    @property
    def remaining_exposure(self):

        return max(

            ZERO,

            self.maximum_exposure
            - self.state.total_exposure

        )

    @property
    def remaining_open_risk(self):

        return max(

            ZERO,

            self.maximum_aggregate_risk
            - self.state.aggregate_open_risk

        )

    @property
    def daily_loss_limit(self):

        return (

            self.state.day_start_equity

            * self.maximum_daily_loss_percent

            / HUNDRED

        )

    @property
    def remaining_daily_risk(self):

        return max(

            ZERO,

            self.daily_loss_limit

            - self.state.daily_loss

            - self.state.aggregate_open_risk

        )

    @property
    def remaining_drawdown_risk(self):

        minimum_equity = (

            self.state.peak_equity

            * (
                D("1")
                - self.maximum_drawdown_percent
                / HUNDRED
            )

        )

        return max(

            ZERO,

            self.state.equity

            - minimum_equity

            - self.state.aggregate_open_risk

        )

    # ========================================================
    # SESSION MANAGEMENT
    # ========================================================

    def require_session(self):

        if self._session is None:

            raise RuntimeError(
                "Start a portfolio session first."
            )

    def start_session(self, session):

        validate_session(session, "session")

        if (
            self._session is not None
            and session <= self._session
        ):

            raise ValueError(
                "Sessions must advance chronologically."
            )

        positions = tuple(

            replace(

                position,

                holding_sessions=(
                    position.holding_sessions + 1
                )

            )

            for position in self.state.positions

        )

        updated = replace(

            self.state,

            day_start_equity=self.state.equity,

            daily_entry_halted=False,

            positions=positions

        )

        self._state = updated

        self._session = session

        self._closed_symbols.clear()

        self._session_entries = 0

        self.evaluate_risk_thresholds()

        return self.state

    # ========================================================
    # ACCOUNT STATE TRANSITION
    # ========================================================

    def build_state(
        self,
        cash,
        positions,
        realized_pnl
    ):

        positions = tuple(positions)

        market_value = sum(

            (
                position.market_value
                for position in positions
            ),

            ZERO

        )

        equity = cash + market_value

        peak = max(

            self.state.peak_equity,

            equity

        )

        return replace(

            self.state,

            cash=cash,

            positions=positions,

            realized_pnl=realized_pnl,

            peak_equity=peak

        )

    # ========================================================
    # RISK THRESHOLD MONITORING
    # ========================================================

    def evaluate_risk_thresholds(self):

        self.require_session()

        daily_breach = (

            self.state.daily_loss_percent

            >= self.maximum_daily_loss_percent

        )

        drawdown_breach = (

            self.state.drawdown_percent

            >= self.maximum_drawdown_percent

        )

        events = []

        if (
            daily_breach
            and not self.state.daily_entry_halted
        ):

            events.append(

                PortfolioRiskEvent(

                    session=self.session,

                    event_type="DAILY_LOSS",

                    reason="Daily loss threshold reached",

                    observed_value=(
                        self.state.daily_loss_percent
                    ),

                    threshold=(
                        self.maximum_daily_loss_percent
                    )

                )

            )

        if (
            drawdown_breach
            and not self.state.trading_halted
        ):

            events.append(

                PortfolioRiskEvent(

                    session=self.session,

                    event_type="MAXIMUM_DRAWDOWN",

                    reason="Maximum drawdown reached",

                    observed_value=(
                        self.state.drawdown_percent
                    ),

                    threshold=(
                        self.maximum_drawdown_percent
                    )

                )

            )

        self._state = replace(

            self.state,

            daily_entry_halted=(

                self.state.daily_entry_halted

                or daily_breach

            ),

            trading_halted=(

                self.state.trading_halted

                or drawdown_breach

            )

        )

        self._risk_events.extend(events)

        return tuple(events)

    # ========================================================
    # ENTRY ALLOCATION
    # ========================================================

    @staticmethod
    def reject(reason):

        return CapitalAllocation(

            approved=False,

            reason=reason

        )

    def calculate_allocation(
        self,
        opportunity,
        market_price
    ):

        self.require_session()

        self._config.validate()

        if not isinstance(
            opportunity,
            PortfolioOpportunity
        ):

            raise ValueError(
                "Invalid portfolio opportunity."
            )

        market_price = positive_decimal(

            market_price,

            "market_price"

        )

        if opportunity.signal_date >= self.session:

            return self.reject(
                "Signal must precede execution session."
            )

        if opportunity.symbol not in self.symbols:

            return self.reject(
                "Symbol is outside trading universe."
            )

        if opportunity.strategy not in (
            self.enabled_strategies
        ):

            return self.reject(
                "Strategy is disabled."
            )

        if opportunity.score < self.minimum_score:

            return self.reject(
                "Minimum opportunity score not reached."
            )

        if (
            opportunity.stop_atr_multiplier
            != self.stop_atr_multiplier

            or opportunity.risk_reward_ratio
            != self.risk_reward_ratio
        ):

            return self.reject(
                "Opportunity strategy parameters mismatch."
            )

        if self.state.trading_halted:

            return self.reject(
                "Portfolio trading is halted."
            )

        if self.state.daily_entry_halted:

            return self.reject(
                "Daily entry halt is active."
            )

        if (
            opportunity.symbol
            in self.state.existing_symbols
        ):

            return self.reject(
                "Position already exists."
            )

        if opportunity.symbol in self._closed_symbols:

            return self.reject(
                "Same-session reentry is prohibited."
            )

        if (
            self.state.open_positions
            >= self.maximum_positions
        ):

            return self.reject(
                "Maximum open positions reached."
            )

        if (
            self._session_entries
            >= self.maximum_new_positions
        ):

            return self.reject(
                "Session entry limit reached."
            )

        entry_price = (

            market_price * (D("1") + self.slippage)

        )

        try:

            stop_price, target_price = (
                opportunity.prices_for_entry(
                    entry_price
                )
            )

        except ValueError as error:

            return self.reject(str(error))

        cost_per_share = (

            entry_price * (D("1") + self.fee_rate)

        )

        stop_fill = (

            stop_price * (D("1") - self.slippage)

        )

        stop_proceeds = (

            stop_fill * (D("1") - self.fee_rate)

        )

        risk_per_share = (

            cost_per_share - stop_proceeds

        )

        risk_budget = min(

            self.maximum_trade_risk,

            self.remaining_open_risk,

            self.remaining_daily_risk,

            self.remaining_drawdown_risk

        )

        if risk_budget <= ZERO:

            return self.reject(
                "No remaining planned risk budget."
            )

        if self.remaining_exposure <= ZERO:

            return self.reject(
                "Maximum exposure reached."
            )

        if self.state.cash < self.minimum_order_value:

            return self.reject(
                "Insufficient available cash."
            )

        quantity = min(

            risk_budget / risk_per_share,

            self.maximum_position_value / entry_price,

            self.remaining_exposure / entry_price,

            self.state.cash / cost_per_share

        ).quantize(

            self.quantity_step,

            rounding=ROUND_DOWN

        )

        if quantity <= ZERO:

            return self.reject(
                "Calculated quantity is zero."
            )

        # Entry fees immediately reduce equity. Recalculate
        # allocation ceilings using the post-entry capital.

        while quantity > ZERO:

            position_value = quantity * entry_price

            entry_fee = position_value * self.fee_rate

            entry_cost = position_value + entry_fee

            planned_risk = quantity * risk_per_share

            post_equity = self.state.equity - entry_fee

            post_capital = min(

                post_equity,

                self.initial_capital

            )

            position_limit = (

                post_capital

                * self.maximum_position_percent

                / HUNDRED

            )

            exposure_limit = (

                post_capital

                * self.maximum_exposure_percent

                / HUNDRED

            )

            trade_risk_limit = (

                post_capital

                * self.risk_per_trade_percent

                / HUNDRED

            )

            aggregate_risk_limit = (

                post_capital

                * self.maximum_open_risk_percent

                / HUNDRED

            )

            if (

                entry_cost <= self.state.cash

                and position_value <= position_limit

                and (
                    self.state.total_exposure
                    + position_value
                    <= exposure_limit
                )

                and planned_risk <= trade_risk_limit

                and (
                    self.state.aggregate_open_risk
                    + planned_risk
                    <= aggregate_risk_limit
                )

                and planned_risk <= risk_budget

            ):

                break

            reduced = min(

                quantity - self.quantity_step,

                position_limit / entry_price,

                max(
                    ZERO,
                    exposure_limit
                    - self.state.total_exposure
                ) / entry_price,

                trade_risk_limit / risk_per_share,

                max(
                    ZERO,
                    aggregate_risk_limit
                    - self.state.aggregate_open_risk
                ) / risk_per_share

            )

            quantity = reduced.quantize(

                self.quantity_step,

                rounding=ROUND_DOWN

            )

        if quantity <= ZERO:

            return self.reject(
                "No affordable quantity within risk limits."
            )

        if position_value < self.minimum_order_value:

            return self.reject(
                "Position value below minimum order value."
            )

        return CapitalAllocation(

            approved=True,

            reason="Capital allocation approved",

            quantity=quantity,

            entry_price=entry_price,

            stop_price=stop_price,

            target_price=target_price,

            entry_fee=entry_fee,

            position_value=position_value,

            entry_cost=entry_cost,

            planned_risk=planned_risk

        )

    # ========================================================
    # SIMULATED POSITION ENTRY
    # ========================================================

    def open_position(
        self,
        opportunity,
        market_price
    ):

        allocation = self.calculate_allocation(

            opportunity,

            market_price

        )

        if not allocation.approved:

            return PortfolioExecutionResult(

                executed=False,

                symbol=opportunity.symbol,

                reason=allocation.reason

            )

        position = PortfolioPosition(

            symbol=opportunity.symbol,

            strategy=opportunity.strategy,

            entry_date=self.session,

            quantity=allocation.quantity,

            entry_price=allocation.entry_price,

            stop_price=allocation.stop_price,

            target_price=allocation.target_price,

            current_price=allocation.entry_price,

            planned_risk=allocation.planned_risk,

            entry_fee=allocation.entry_fee

        )

        updated = self.build_state(

            cash=self.state.cash - position.entry_cost,

            positions=self.state.positions + (position,),

            realized_pnl=self.state.realized_pnl

        )

        self._state = updated

        self._session_entries += 1

        self.evaluate_risk_thresholds()

        return PortfolioExecutionResult(

            executed=True,

            symbol=position.symbol,

            reason="Historical entry executed",

            position=position

        )

    # ========================================================
    # MARK OPEN POSITIONS TO MARKET
    # ========================================================

    def mark_to_market(self, prices):

        self.require_session()

        if not isinstance(prices, Mapping):

            raise ValueError(
                "Market prices must be a mapping."
            )

        updated_positions = []

        for position in self.state.positions:

            if position.symbol not in prices:

                raise ValueError(

                    f"Missing market price: "
                    f"{position.symbol}"

                )

            price = positive_decimal(

                prices[position.symbol],

                f"{position.symbol} market price"

            )

            updated_positions.append(

                replace(

                    position,

                    current_price=price

                )

            )

        updated = self.build_state(

            cash=self.state.cash,

            positions=updated_positions,

            realized_pnl=self.state.realized_pnl

        )

        self._state = updated

        self.evaluate_risk_thresholds()

        return self.state

    # ========================================================
    # SIMULATED POSITION EXIT
    # ========================================================

    def close_position(
        self,
        symbol,
        market_price,
        reason
    ):

        self.require_session()

        market_price = positive_decimal(

            market_price,

            "market_price"

        )

        position = next(

            (

                position

                for position in self.state.positions

                if position.symbol == symbol

            ),

            None

        )

        if position is None:

            return PortfolioExecutionResult(

                executed=False,

                symbol=symbol,

                reason="No open position found"

            )

        exit_price = (

            market_price * (D("1") - self.slippage)

        )

        exit_fee = (

            position.quantity

            * exit_price

            * self.fee_rate

        )

        trade = PortfolioTrade(

            symbol=position.symbol,

            strategy=position.strategy,

            entry_date=position.entry_date,

            exit_date=self.session,

            quantity=position.quantity,

            entry_price=position.entry_price,

            exit_price=exit_price,

            stop_price=position.stop_price,

            target_price=position.target_price,

            planned_risk=position.planned_risk,

            exit_reason=reason,

            entry_fee=position.entry_fee,

            exit_fee=exit_fee,

            holding_sessions=position.holding_sessions

        )

        remaining_positions = tuple(

            item

            for item in self.state.positions

            if item.symbol != symbol

        )

        updated = self.build_state(

            cash=self.state.cash + trade.exit_proceeds,

            positions=remaining_positions,

            realized_pnl=(
                self.state.realized_pnl + trade.pnl
            )

        )

        self._state = updated

        self._trades.append(trade)

        self._closed_symbols.add(symbol)

        if trade.risk_breached:

            self._risk_events.append(

                PortfolioRiskEvent(

                    session=self.session,

                    event_type="PLANNED_RISK_BREACH",

                    reason=(
                        "Actual loss exceeded planned risk"
                    ),

                    observed_value=trade.actual_loss,

                    threshold=trade.planned_risk,

                    symbol=symbol

                )

            )

        self.evaluate_risk_thresholds()

        return PortfolioExecutionResult(

            executed=True,

            symbol=symbol,

            reason="Historical exit executed",

            trade=trade

        )

    # ========================================================
    # ACCOUNT SUMMARY
    # ========================================================

    def display_summary(self):

        account = self.state

        print("=" * 70)

        print("JARVIS SHARED CAPITAL MANAGER")

        print("VERSION 0.8.6.4")

        print("=" * 70)

        print(f"Session: {self.session}")

        print(f"Initial capital: ${account.initial_capital:.2f}")

        print(f"Cash: ${account.cash:.6f}")

        print(f"Market value: ${account.market_value:.6f}")

        print(f"Equity: ${account.equity:.6f}")

        print(f"Realized P&L: ${account.realized_pnl:.6f}")

        print(f"Unrealized P&L: ${account.unrealized_pnl:.6f}")

        print(f"Open positions: {account.open_positions}")

        print(f"Exposure: {account.exposure_percent:.4f}%")

        print(f"Aggregate risk: ${account.aggregate_open_risk:.6f}")

        print(f"Remaining exposure: ${self.remaining_exposure:.6f}")

        print(f"Remaining open risk: ${self.remaining_open_risk:.6f}")

        print(f"Daily loss: {account.daily_loss_percent:.4f}%")

        print(f"Drawdown: {account.drawdown_percent:.4f}%")

        print(f"Completed trades: {len(self.trades)}")

        print(f"Risk events: {len(self.risk_events)}")

        print(f"Daily entry halt: {account.daily_entry_halted}")

        print(f"Trading halted: {account.trading_halted}")

        print()

        print("Historical simulation only")

        print("Broker orders: DISABLED")

        print("Live trading: DISABLED")


if __name__ == "__main__":

    manager = CapitalManager()

    manager.display_summary()