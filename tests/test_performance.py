import pytest

from backtesting.performance import PerformanceMetrics


def test_profitable_backtest():

    trades = [{"pnl": 2.0}, {"pnl": -1.0}, {"pnl": 3.0}]

    equity = [100, 102, 101, 104]

    result = PerformanceMetrics.calculate(trades, equity, 100)

    assert result["final_capital"] == 104
    assert result["net_profit"] == 4
    assert result["total_return"] == 4
    assert result["total_trades"] == 3
    assert result["winning_trades"] == 2
    assert result["losing_trades"] == 1
    assert result["profit_factor"] == 5


def test_losing_backtest():

    trades = [{"pnl": -2.0}, {"pnl": -3.0}]

    equity = [100, 98, 95]

    result = PerformanceMetrics.calculate(trades, equity, 100)

    assert result["net_profit"] == -5
    assert result["total_return"] == -5
    assert result["win_rate"] == 0
    assert result["profit_factor"] == 0
    assert result["max_drawdown"] == 5


def test_no_trades():

    result = PerformanceMetrics.calculate([], [100, 100], 100)

    assert result["total_trades"] == 0
    assert result["net_profit"] == 0
    assert result["win_rate"] == 0
    assert result["max_drawdown"] == 0


def test_breakeven_trade():

    result = PerformanceMetrics.calculate([{"pnl": 0}], [100, 100], 100)

    assert result["breakeven_trades"] == 1
    assert result["winning_trades"] == 0
    assert result["losing_trades"] == 0


def test_maximum_drawdown():

    equity = [100, 110, 99, 105]

    result = PerformanceMetrics.calculate([], equity, 100)

    assert result["max_drawdown"] == pytest.approx(10)


def test_profit_factor():

    trades = [{"pnl": 4}, {"pnl": 2}, {"pnl": -3}]

    result = PerformanceMetrics.calculate(trades, [100, 104, 106, 103], 100)

    assert result["profit_factor"] == 2


def test_expectancy():

    trades = [{"pnl": 1}, {"pnl": -0.5}, {"pnl": 1}, {"pnl": -0.5}]

    result = PerformanceMetrics.calculate(trades, [100, 101, 100.5, 101.5, 101], 100)

    assert result["expectancy"] == pytest.approx(0.25)


def test_all_winning_trades():

    trades = [{"pnl": 1}, {"pnl": 2}]

    result = PerformanceMetrics.calculate(trades, [100, 101, 103], 100)

    assert result["win_rate"] == 100
    assert result["profit_factor"] == float("inf")


def test_average_win_and_loss():

    trades = [{"pnl": 2}, {"pnl": 4}, {"pnl": -1}, {"pnl": -3}]

    result = PerformanceMetrics.calculate(trades, [100, 102, 106, 105, 102], 100)

    assert result["average_win"] == 3
    assert result["average_loss"] == -2


def test_equity_recovery():

    equity = [100, 90, 100, 110]

    result = PerformanceMetrics.calculate([], equity, 100)

    assert result["max_drawdown"] == 10
    assert result["total_return"] == 10
