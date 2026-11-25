import json
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import ROUND_DOWN, Decimal
from pathlib import Path

from risk.risk_config import RiskConfig

BASE_DIR = Path(__file__).resolve().parents[1]

LOG_FILE = BASE_DIR / "logs" / "risk_guardian.jsonl"

ZERO = Decimal(0)
HUNDRED = Decimal(100)


@dataclass
class AccountState:
    equity: Decimal
    cash: Decimal
    day_start_equity: Decimal
    peak_equity: Decimal
    open_positions: int
    total_exposure: Decimal

    existing_symbols: tuple = ()


@dataclass
class TradeRequest:
    symbol: str
    entry_price: Decimal
    stop_price: Decimal

    side: str = "buy"


@dataclass
class RiskDecision:
    approved: bool
    symbol: str
    reason: str

    quantity: Decimal = ZERO
    position_value: Decimal = ZERO
    estimated_risk: Decimal = ZERO


class RiskGuardian:
    def __init__(self, config=None):

        self.config = config or RiskConfig.load()

    def log_decision(self, decision):

        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "symbol": decision.symbol,
            "approved": decision.approved,
            "reason": decision.reason,
            "quantity": str(decision.quantity),
            "position_value": str(decision.position_value),
            "estimated_risk": str(decision.estimated_risk),
        }

        with open(LOG_FILE, "a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")

        return decision

    def reject(self, symbol, reason):

        return self.log_decision(
            RiskDecision(approved=False, symbol=symbol, reason=reason)
        )

    def evaluate(self, account, trade):

        cfg = self.config

        symbol = trade.symbol.strip().upper()

        # SYSTEM SAFETY

        if not cfg.enabled:
            return self.reject(symbol, "Risk Guardian disabled. Entries blocked.")

        if cfg.emergency_stop:
            return self.reject(symbol, "Emergency stop activated.")

        if cfg.trading_mode != "paper":
            return self.reject(symbol, "Only paper trading is permitted.")

        if cfg.live_trading_enabled:
            return self.reject(symbol, "Live trading is prohibited.")

        if cfg.margin_enabled or cfg.short_selling_enabled:
            return self.reject(symbol, "Margin and short selling are prohibited.")

        # ACCOUNT VALIDATION

        account_values = [
            account.equity,
            account.cash,
            account.day_start_equity,
            account.peak_equity,
            account.total_exposure,
        ]

        if not all(value.is_finite() for value in account_values):
            return self.reject(symbol, "Invalid account data.")

        if (
            account.equity <= ZERO
            or account.cash < ZERO
            or account.day_start_equity <= ZERO
            or account.peak_equity <= ZERO
            or account.total_exposure < ZERO
            or account.open_positions < 0
        ):
            return self.reject(symbol, "Invalid account state.")

        if account.peak_equity < account.equity:
            return self.reject(symbol, "Account peak equity is inconsistent.")

        # DAILY LOSS PROTECTION

        daily_loss = max(ZERO, account.day_start_equity - account.equity)

        daily_loss_limit = (
            account.day_start_equity * cfg.max_daily_loss_percent / HUNDRED
        )

        if daily_loss >= daily_loss_limit:
            return self.reject(symbol, "Maximum daily loss reached.")

        # DRAWDOWN PROTECTION

        drawdown = (
            (account.peak_equity - account.equity) / account.peak_equity * HUNDRED
        )

        if drawdown >= cfg.max_drawdown_percent:
            return self.reject(symbol, "Maximum drawdown reached.")

        # POSITION COUNT

        if account.open_positions >= cfg.max_open_positions:
            return self.reject(symbol, "Maximum open positions reached.")

        if symbol in account.existing_symbols:
            return self.reject(symbol, "Position already exists.")

        # TRADE VALIDATION

        if not symbol:
            return self.reject(symbol, "Missing stock symbol.")

        if trade.side.lower() != "buy":
            return self.reject(symbol, "Only long entry requests are permitted.")

        if not (trade.entry_price.is_finite() and trade.stop_price.is_finite()):
            return self.reject(symbol, "Invalid trade prices.")

        if trade.entry_price <= ZERO:
            return self.reject(symbol, "Invalid entry price.")

        if trade.stop_price <= ZERO or trade.stop_price >= trade.entry_price:
            return self.reject(symbol, "Invalid protective stop price.")

        # CAPITAL RESTRICTION

        capital = min(account.equity, cfg.starting_capital)

        risk_budget = capital * cfg.risk_per_trade_percent / HUNDRED

        remaining_daily_risk = daily_loss_limit - daily_loss

        remaining_drawdown_risk = account.equity - account.peak_equity * (
            Decimal(1) - cfg.max_drawdown_percent / HUNDRED
        )

        risk_budget = min(risk_budget, remaining_daily_risk, remaining_drawdown_risk)

        max_position_value = capital * cfg.max_position_percent / HUNDRED

        max_total_exposure = capital * cfg.max_total_exposure_percent / HUNDRED

        remaining_exposure = max_total_exposure - account.total_exposure

        if remaining_exposure <= ZERO:
            return self.reject(symbol, "Maximum portfolio exposure reached.")

        if account.cash < cfg.min_order_value:
            return self.reject(symbol, "Insufficient available cash.")

        # RISK-BASED POSITION SIZING

        risk_per_share = trade.entry_price - trade.stop_price

        quantity_by_risk = risk_budget / risk_per_share

        quantity_by_position = max_position_value / trade.entry_price

        quantity_by_cash = account.cash / trade.entry_price

        quantity_by_exposure = remaining_exposure / trade.entry_price

        quantity = min(
            quantity_by_risk,
            quantity_by_position,
            quantity_by_cash,
            quantity_by_exposure,
        )

        quantity = quantity.quantize(Decimal("0.000001"), rounding=ROUND_DOWN)

        position_value = quantity * trade.entry_price

        estimated_risk = quantity * risk_per_share

        if quantity <= ZERO or position_value < cfg.min_order_value:
            return self.reject(symbol, "Calculated position is below minimum size.")

        # APPROVED SIMULATED REQUEST

        return self.log_decision(
            RiskDecision(
                approved=True,
                symbol=symbol,
                reason="Passed configured risk checks.",
                quantity=quantity,
                position_value=position_value,
                estimated_risk=estimated_risk,
            )
        )
