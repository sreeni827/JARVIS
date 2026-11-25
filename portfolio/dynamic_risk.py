from dataclasses import dataclass
from decimal import Decimal

from portfolio.accounting_context import (
    isolated_accounting,
)

from portfolio.decimal_utils import (
    as_decimal,
    normalize_money,
    normalize_rate,
    calculate_sell_price,
    calculate_notional,
    calculate_fee,
    calculate_exit_proceeds,
)

from portfolio.models import (
    PortfolioPosition,
    PortfolioAccountState,
)


D = Decimal

ZERO = D("0")
HUNDRED = D("100")


# =========================================================
# POSITION RISK REPORT
# =========================================================


@dataclass(frozen=True, slots=True)
class PositionRiskReport:

    symbol: str

    quantity: Decimal

    entry_price: Decimal

    current_price: Decimal

    stop_price: Decimal

    exit_reference: Decimal

    expected_exit_price: Decimal

    current_market_value: Decimal

    expected_exit_notional: Decimal

    expected_exit_fee: Decimal

    expected_exit_proceeds: Decimal

    original_planned_risk: Decimal

    dynamic_stop_risk: Decimal

    reservation_difference: Decimal

    stop_breached: bool

    @property
    @isolated_accounting
    def under_reserved(self):

        return (
            self.dynamic_stop_risk
            > self.original_planned_risk
        )

    @property
    @isolated_accounting
    def over_reserved(self):

        return (
            self.dynamic_stop_risk
            < self.original_planned_risk
        )

    @property
    @isolated_accounting
    def risk_multiplier(self):

        return (
            self.dynamic_stop_risk
            / self.original_planned_risk
        )

    @property
    @isolated_accounting
    def classification(self):

        if self.under_reserved:

            return "UNDER_RESERVED"

        if self.over_reserved:

            return "OVER_RESERVED"

        return "EXACT"

    @isolated_accounting
    def to_record(self):

        return {
            "symbol": self.symbol,
            "quantity": str(self.quantity),
            "entry_price": str(self.entry_price),
            "current_price": str(self.current_price),
            "stop_price": str(self.stop_price),
            "exit_reference": str(self.exit_reference),
            "expected_exit_price": str(
                self.expected_exit_price
            ),
            "current_market_value": str(
                self.current_market_value
            ),
            "expected_exit_notional": str(
                self.expected_exit_notional
            ),
            "expected_exit_fee": str(
                self.expected_exit_fee
            ),
            "expected_exit_proceeds": str(
                self.expected_exit_proceeds
            ),
            "original_planned_risk": str(
                self.original_planned_risk
            ),
            "dynamic_stop_risk": str(
                self.dynamic_stop_risk
            ),
            "reservation_difference": str(
                self.reservation_difference
            ),
            "risk_multiplier": str(
                self.risk_multiplier
            ),
            "stop_breached": self.stop_breached,
            "classification": self.classification,
        }


# =========================================================
# PORTFOLIO RISK REPORT
# =========================================================


@dataclass(frozen=True, slots=True)
class PortfolioRiskReport:

    equity: Decimal

    maximum_aggregate_risk: Decimal

    original_aggregate_risk: Decimal

    dynamic_aggregate_risk: Decimal

    reservation_difference: Decimal

    positions: tuple[PositionRiskReport, ...]

    @property
    @isolated_accounting
    def open_positions(self):

        return len(self.positions)

    @property
    @isolated_accounting
    def remaining_capacity(self):

        return max(
            ZERO,
            self.maximum_aggregate_risk
            - self.dynamic_aggregate_risk,
        )

    @property
    @isolated_accounting
    def excess_risk(self):

        return max(
            ZERO,
            self.dynamic_aggregate_risk
            - self.maximum_aggregate_risk,
        )

    @property
    @isolated_accounting
    def risk_budget_exceeded(self):

        return (
            self.dynamic_aggregate_risk
            > self.maximum_aggregate_risk
        )

    @property
    @isolated_accounting
    def risk_budget_exhausted(self):

        return (
            self.dynamic_aggregate_risk
            >= self.maximum_aggregate_risk
        )

    @property
    @isolated_accounting
    def dynamic_risk_percent(self):

        return (
            self.dynamic_aggregate_risk
            / self.equity
            * HUNDRED
        )

    @property
    @isolated_accounting
    def original_risk_percent(self):

        return (
            self.original_aggregate_risk
            / self.equity
            * HUNDRED
        )

    @property
    @isolated_accounting
    def budget_utilization_percent(self):

        if self.maximum_aggregate_risk == ZERO:

            return None

        return (
            self.dynamic_aggregate_risk
            / self.maximum_aggregate_risk
            * HUNDRED
        )

    @property
    @isolated_accounting
    def breached_stop_symbols(self):

        return tuple(
            position.symbol
            for position in self.positions
            if position.stop_breached
        )

    @property
    @isolated_accounting
    def under_reserved_symbols(self):

        return tuple(
            position.symbol
            for position in self.positions
            if position.under_reserved
        )

    @property
    @isolated_accounting
    def over_reserved_symbols(self):

        return tuple(
            position.symbol
            for position in self.positions
            if position.over_reserved
        )

    @property
    @isolated_accounting
    def requires_entry_restriction(self):

        return (
            self.risk_budget_exhausted
            or bool(self.breached_stop_symbols)
        )

    @isolated_accounting
    def to_record(self):

        utilization = self.budget_utilization_percent

        return {
            "equity": str(self.equity),
            "open_positions": self.open_positions,
            "maximum_aggregate_risk": str(
                self.maximum_aggregate_risk
            ),
            "original_aggregate_risk": str(
                self.original_aggregate_risk
            ),
            "dynamic_aggregate_risk": str(
                self.dynamic_aggregate_risk
            ),
            "reservation_difference": str(
                self.reservation_difference
            ),
            "remaining_capacity": str(
                self.remaining_capacity
            ),
            "excess_risk": str(
                self.excess_risk
            ),
            "dynamic_risk_percent": str(
                self.dynamic_risk_percent
            ),
            "original_risk_percent": str(
                self.original_risk_percent
            ),
            "budget_utilization_percent": (
                str(utilization)
                if utilization is not None
                else None
            ),
            "risk_budget_exceeded": (
                self.risk_budget_exceeded
            ),
            "risk_budget_exhausted": (
                self.risk_budget_exhausted
            ),
            "breached_stop_symbols": list(
                self.breached_stop_symbols
            ),
            "under_reserved_symbols": list(
                self.under_reserved_symbols
            ),
            "over_reserved_symbols": list(
                self.over_reserved_symbols
            ),
            "requires_entry_restriction": (
                self.requires_entry_restriction
            ),
            "positions": [
                position.to_record()
                for position in self.positions
            ],
        }


# =========================================================
# DYNAMIC RISK CALCULATOR
# =========================================================


@dataclass(frozen=True, slots=True)
class DynamicRiskCalculator:

    slippage_rate: Decimal

    fee_rate: Decimal

    @isolated_accounting
    def __post_init__(self):

        object.__setattr__(
            self,
            "slippage_rate",
            normalize_rate(self.slippage_rate),
        )

        object.__setattr__(
            self,
            "fee_rate",
            normalize_rate(self.fee_rate),
        )

    @isolated_accounting
    def calculate_position_risk(
        self,
        position,
    ):

        if not isinstance(
            position,
            PortfolioPosition,
        ):

            raise TypeError(
                "Expected PortfolioPosition."
            )

        exit_reference = min(
            position.current_price,
            position.stop_price,
        )

        expected_exit_price = (
            calculate_sell_price(
                exit_reference,
                self.slippage_rate,
            )
        )

        expected_exit_notional = (
            calculate_notional(
                position.quantity,
                expected_exit_price,
            )
        )

        expected_exit_fee = calculate_fee(
            expected_exit_notional,
            self.fee_rate,
        )

        expected_exit_proceeds = (
            calculate_exit_proceeds(
                position.quantity,
                expected_exit_price,
                self.fee_rate,
            )
        )

        current_market_value = (
            position.market_value
        )

        dynamic_stop_risk = max(
            ZERO,
            normalize_money(
                current_market_value
                - expected_exit_proceeds
            ),
        )

        reservation_difference = (
            normalize_money(
                dynamic_stop_risk
                - position.planned_risk
            )
        )

        return PositionRiskReport(
            symbol=position.symbol,
            quantity=position.quantity,
            entry_price=position.entry_price,
            current_price=position.current_price,
            stop_price=position.stop_price,
            exit_reference=exit_reference,
            expected_exit_price=expected_exit_price,
            current_market_value=current_market_value,
            expected_exit_notional=expected_exit_notional,
            expected_exit_fee=expected_exit_fee,
            expected_exit_proceeds=expected_exit_proceeds,
            original_planned_risk=position.planned_risk,
            dynamic_stop_risk=dynamic_stop_risk,
            reservation_difference=reservation_difference,
            stop_breached=(
                position.current_price
                <= position.stop_price
            ),
        )

    @isolated_accounting
    def calculate_portfolio_risk(
        self,
        account,
        maximum_aggregate_risk,
    ):

        if not isinstance(
            account,
            PortfolioAccountState,
        ):

            raise TypeError(
                "Expected PortfolioAccountState."
            )

        maximum_aggregate_risk = as_decimal(
            maximum_aggregate_risk
        )

        if maximum_aggregate_risk < ZERO:

            raise ValueError(
                "Aggregate risk budget cannot be negative."
            )

        if account.equity <= ZERO:

            raise ValueError(
                "Portfolio equity must be positive."
            )

        reports = tuple(
            self.calculate_position_risk(position)
            for position in account.positions
        )

        original_aggregate_risk = normalize_money(
            sum(
                (
                    report.original_planned_risk
                    for report in reports
                ),
                ZERO,
            )
        )

        dynamic_aggregate_risk = normalize_money(
            sum(
                (
                    report.dynamic_stop_risk
                    for report in reports
                ),
                ZERO,
            )
        )

        if original_aggregate_risk != (
            account.aggregate_open_risk
        ):

            raise RuntimeError(
                "Original risk reconciliation failed."
            )

        reservation_difference = (
            normalize_money(
                dynamic_aggregate_risk
                - original_aggregate_risk
            )
        )

        return PortfolioRiskReport(
            equity=account.equity,
            maximum_aggregate_risk=maximum_aggregate_risk,
            original_aggregate_risk=original_aggregate_risk,
            dynamic_aggregate_risk=dynamic_aggregate_risk,
            reservation_difference=reservation_difference,
            positions=reports,
        )


def display_summary():

    print("=" * 70)
    print("JARVIS DYNAMIC RISK CALCULATOR")
    print("VERSION 0.8.6.7.2D.2")
    print("=" * 70)

    print("Original planned risk: PRESERVED")
    print("Dynamic stop risk: ENABLED")
    print("Exit slippage accounting: ENABLED")
    print("Exit fee accounting: ENABLED")
    print("Below-stop valuation: ENABLED")
    print("Breached-stop detection: ENABLED")
    print("Aggregate risk reporting: ENABLED")
    print("Decimal context isolation: ENABLED")
    print("Capital Manager integration: PENDING")
    print("Historical simulation only")
    print("Broker order submission: DISABLED")


if __name__ == "__main__":

    display_summary()