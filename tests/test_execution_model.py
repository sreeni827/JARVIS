import pytest

from backtesting.execution_model import ExecutionModel


@pytest.fixture
def model():

    return ExecutionModel()


def evaluate(model, **overrides):

    parameters = {
        "opening": 100,
        "high": 103,
        "low": 98,
        "close": 101,
        "stop": 95,
        "target": 110,
        "holding_days": 1,
        "maximum_holding_days": 20,
        "final_session": False,
        "forced_exit": False,
    }

    parameters.update(overrides)

    return model.evaluate_long_exit(**parameters)


def test_buy_slippage(model):

    assert model.buy_price(100) == pytest.approx(100.05)


def test_sell_slippage(model):

    assert model.sell_price(100) == pytest.approx(99.95)


def test_entry_fee(model):

    assert model.entry_cost(1, 100) == pytest.approx(100.02)


def test_exit_fee(model):

    assert model.exit_proceeds(1, 100) == pytest.approx(99.98)


def test_round_trip_cost(model):

    entry = model.buy_price(100)

    exit_price = model.sell_price(100)

    pnl = model.calculate_pnl(1, entry, exit_price)

    assert pnl == pytest.approx(-0.14)


def test_normal_stop_loss(model):

    result = evaluate(model, low=94)

    assert result.triggered
    assert result.price == 95
    assert result.reason == "STOP_LOSS"


def test_normal_take_profit(model):

    result = evaluate(model, high=112)

    assert result.triggered
    assert result.price == 110
    assert result.reason == "TAKE_PROFIT"


def test_gap_below_stop(model):

    result = evaluate(model, opening=90, high=94, low=88, close=92)

    assert result.price == 90
    assert result.reason == "STOP_GAP"


def test_gap_above_target(model):

    result = evaluate(model, opening=115, high=118, low=112, close=116)

    assert result.price == 110
    assert result.reason == "TAKE_PROFIT_GAP"


def test_stop_and_target_same_day(model):

    result = evaluate(model, high=115, low=90)

    assert result.price == 95
    assert result.reason == "STOP_LOSS"


def test_no_exit(model):

    result = evaluate(model)

    assert not result.triggered
    assert result.reason == "HOLD"


def test_time_exit(model):

    result = evaluate(model, holding_days=20)

    assert result.price == 101
    assert result.reason == "TIME_EXIT"


def test_end_of_backtest(model):

    result = evaluate(model, final_session=True)

    assert result.price == 101
    assert result.reason == "END_OF_BACKTEST"


def test_forced_exit_at_open(model):

    result = evaluate(model, forced_exit=True)

    assert result.price == 100
    assert result.reason == "DRAWDOWN_EXIT"


def test_gap_priority_over_forced_exit(model):

    result = evaluate(model, opening=90, high=94, low=88, close=92, forced_exit=True)

    assert result.price == 90
    assert result.reason == "STOP_GAP"


def test_invalid_price(model):

    with pytest.raises(ValueError):
        model.buy_price(-100)


def test_invalid_ohlc(model):

    with pytest.raises(ValueError):
        evaluate(model, high=90)


def test_invalid_stop_target(model):

    with pytest.raises(ValueError):
        evaluate(model, stop=120, target=110)


def test_invalid_slippage():

    with pytest.raises(ValueError):
        ExecutionModel(slippage_percent=-1)


def test_fractional_share_pnl(model):

    pnl = model.calculate_pnl(quantity=0.1, entry_price=100, exit_price=110)

    assert pnl == pytest.approx(0.9958)
