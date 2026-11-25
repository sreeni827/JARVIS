from dataclasses import replace
from decimal import Decimal

import pytest

from risk.risk_config import RiskConfig
from risk.risk_guardian import AccountState, RiskGuardian, TradeRequest

D = Decimal


@pytest.fixture
def config():

    return RiskConfig(
        starting_capital=D("100"),
        max_daily_loss_percent=D("1"),
        max_drawdown_percent=D("5"),
        risk_per_trade_percent=D("0.5"),
        max_position_percent=D("20"),
        max_total_exposure_percent=D("60"),
        min_order_value=D("1"),
        max_open_positions=3,
        enabled=True,
        emergency_stop=False,
        live_trading_enabled=False,
        margin_enabled=False,
        short_selling_enabled=False,
        trading_mode="paper",
    )


@pytest.fixture
def account():

    return AccountState(
        equity=D("100"),
        cash=D("100"),
        day_start_equity=D("100"),
        peak_equity=D("100"),
        open_positions=0,
        total_exposure=D("0"),
    )


@pytest.fixture
def trade():

    return TradeRequest(symbol="TEST", entry_price=D("100"), stop_price=D("95"))


@pytest.fixture(autouse=True)
def isolate_logs(tmp_path, monkeypatch):

    monkeypatch.setattr("risk.risk_guardian.LOG_FILE", tmp_path / "risk_test.jsonl")


def test_valid_trade(config, account, trade):

    result = RiskGuardian(config).evaluate(account, trade)

    assert result.approved
    assert result.quantity == D("0.1")
    assert result.position_value == D("10")
    assert result.estimated_risk == D("0.5")


def test_daily_loss_limit(config, account, trade):

    account = replace(account, equity=D("99"))

    result = RiskGuardian(config).evaluate(account, trade)

    assert not result.approved


def test_drawdown_limit(config, account, trade):

    account = replace(account, equity=D("95"), day_start_equity=D("95"))

    result = RiskGuardian(config).evaluate(account, trade)

    assert not result.approved


def test_maximum_positions(config, account, trade):

    account = replace(account, open_positions=3)

    result = RiskGuardian(config).evaluate(account, trade)

    assert not result.approved


def test_emergency_stop(config, account, trade):

    config = replace(config, emergency_stop=True)

    result = RiskGuardian(config).evaluate(account, trade)

    assert not result.approved


def test_live_trading_blocked(config, account, trade):

    config = replace(config, live_trading_enabled=True)

    result = RiskGuardian(config).evaluate(account, trade)

    assert not result.approved


def test_invalid_stop(config, account, trade):

    trade = replace(trade, stop_price=D("105"))

    result = RiskGuardian(config).evaluate(account, trade)

    assert not result.approved


def test_insufficient_cash(config, account, trade):

    account = replace(account, cash=D("0.50"))

    result = RiskGuardian(config).evaluate(account, trade)

    assert not result.approved


def test_exposure_limit(config, account, trade):

    account = replace(account, total_exposure=D("60"))

    result = RiskGuardian(config).evaluate(account, trade)

    assert not result.approved


def test_short_selling_blocked(config, account, trade):

    trade = replace(trade, side="sell")

    result = RiskGuardian(config).evaluate(account, trade)

    assert not result.approved


def test_duplicate_position(config, account, trade):

    account = replace(account, existing_symbols=("TEST",), open_positions=1)

    result = RiskGuardian(config).evaluate(account, trade)

    assert not result.approved


def test_maximum_position_size(config, account, trade):

    trade = replace(trade, stop_price=D("99"))

    result = RiskGuardian(config).evaluate(account, trade)

    assert result.approved
    assert result.position_value <= D("20")


def test_guardian_disabled(config, account, trade):

    config = replace(config, enabled=False)

    result = RiskGuardian(config).evaluate(account, trade)

    assert not result.approved


def test_remaining_daily_risk(config, account, trade):

    account = replace(account, equity=D("99.20"))

    result = RiskGuardian(config).evaluate(account, trade)

    assert result.approved
    assert result.estimated_risk <= D("0.20")


def test_capital_ceiling(config, account, trade):

    account = replace(
        account,
        equity=D("1000"),
        cash=D("1000"),
        day_start_equity=D("1000"),
        peak_equity=D("1000"),
    )

    result = RiskGuardian(config).evaluate(account, trade)

    assert result.approved
    assert result.estimated_risk <= D("0.50")
    assert result.position_value <= D("20")
