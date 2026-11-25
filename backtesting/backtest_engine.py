from datetime import UTC, datetime
from decimal import ROUND_DOWN, Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from backtesting.execution_model import ExecutionModel
from backtesting.performance import PerformanceMetrics
from backtesting.simulation_guardian import SimulationRiskGuardian
from intelligence.opportunity_ranker import OpportunityRanker
from market.indicators import TechnicalIndicators
from market.market_data import MarketDataEngine
from risk.risk_guardian import AccountState, TradeRequest
from strategies.strategy_engine import StrategyEngine

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"

CONFIG_FILE = BASE_DIR / "config" / "backtesting.yaml"

D = Decimal


class BacktestEngine:
    def __init__(self):

        with open(CONFIG_FILE, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)["backtesting"]

        self.market = MarketDataEngine()

        self.ranker = OpportunityRanker()

        self.strategy = StrategyEngine()

        self.guardian = SimulationRiskGuardian()

        self.execution = ExecutionModel(
            slippage_percent=self.config["slippage_percent"],
            fee_percent=self.config["transaction_fee_percent"],
        )

    def prepare_data(self, symbol):

        history = self.market.get_historical_data(
            symbol, days=self.config["historical_days"]
        )

        data = TechnicalIndicators.calculate(history)

        if isinstance(data.index, pd.MultiIndex):
            data = data.reset_index(level=0)

        else:
            data["symbol"] = symbol

        data = data.rename(
            columns={
                "SMA_20": "sma_20",
                "SMA_50": "sma_50",
                "SMA_200": "sma_200",
                "EMA_9": "ema_9",
                "EMA_21": "ema_21",
                "RSI_14": "rsi",
                "MACD": "macd",
                "MACD_SIGNAL": "macd_signal",
                "ATR_14": "atr",
                "RELATIVE_VOLUME": "relative_volume",
                "ROC_10": "momentum",
                "SUPPORT": "support",
                "RESISTANCE": "resistance",
                "TREND": "trend",
            }
        )

        data = data.sort_index()

        prices = data[["open", "high", "low", "close"]]

        if data.index.has_duplicates or not np.isfinite(prices.to_numpy()).all():
            raise ValueError("Duplicate candles or non-finite prices.")

        if (prices <= 0).any().any():
            raise ValueError("Invalid historical prices.")

        if (
            (data["high"] < data["open"]).any()
            or (data["high"] < data["close"]).any()
            or (data["low"] > data["open"]).any()
            or (data["low"] > data["close"]).any()
            or (data["low"] > data["high"]).any()
        ):
            raise ValueError("Invalid OHLC candles.")

        return data

    def run(self, symbol):

        print(f"\nBacktesting {symbol}...")

        data = self.prepare_data(symbol)

        warmup = self.config["warmup_candles"]

        if len(data) <= warmup + 1:
            raise ValueError("Insufficient historical data.")

        initial = float(self.config["initial_capital"])

        cash = initial

        peak_equity = initial

        previous_equity = initial

        equity_curve = [initial]

        equity_dates = [data.index[warmup - 1]]

        trades = []

        position = None

        trading_halted = False

        pending_forced_exit = False

        daily_loss_events = 0

        drawdown_events = 0

        def close_position(position, raw_exit_price, timestamp, reason):

            exit_price = self.execution.sell_price(raw_exit_price)

            proceeds = self.execution.exit_proceeds(position["quantity"], exit_price)

            pnl = proceeds - position["cost"]

            record = {
                "symbol": symbol,
                "strategy": position["strategy"],
                "entry_date": position["entry_date"],
                "exit_date": str(timestamp),
                "entry_price": position["entry"],
                "exit_price": exit_price,
                "quantity": position["quantity"],
                "stop_price": position["stop"],
                "target_price": position["target"],
                "planned_risk": position["planned_risk"],
                "pnl": pnl,
                "exit_reason": reason,
            }

            return proceeds, record

        for i in range(warmup, len(data)):
            today = data.iloc[i]

            yesterday = data.iloc[i - 1]

            timestamp = data.index[i]

            day_start_equity = previous_equity

            # ENTRY EVALUATION
            # Use only the previous completed daily candle.

            if position is None and not trading_halted:
                row = yesterday.copy()

                score, _, _ = self.ranker.calculate_score(row)

                row["score"] = score

                signal = self.strategy.generate_signal(row)

                if signal["signal"] == "BUY":
                    entry = self.execution.buy_price(float(today["open"]))

                    stop_distance = (
                        float(yesterday["atr"]) * self.config["stop_atr_multiplier"]
                    )

                    stop = entry - stop_distance

                    target = entry + stop_distance * self.config["risk_reward_ratio"]

                    account = AccountState(
                        equity=D(str(cash)),
                        cash=D(str(cash)),
                        day_start_equity=D(str(day_start_equity)),
                        peak_equity=D(str(peak_equity)),
                        open_positions=0,
                        total_exposure=D("0"),
                    )

                    request = TradeRequest(
                        symbol=symbol,
                        entry_price=D(str(entry)),
                        stop_price=D(str(stop)),
                    )

                    decision = self.guardian.evaluate(account, request)

                    if decision.approved:
                        quantity = decision.quantity

                        cost_per_share = self.execution.entry_cost(1, entry)

                        stop_fill = self.execution.sell_price(stop)

                        stop_proceeds = self.execution.exit_proceeds(1, stop_fill)

                        risk_per_share = D(str(cost_per_share - stop_proceeds))

                        if risk_per_share > 0:
                            quantity = min(
                                quantity,
                                decision.estimated_risk / risk_per_share,
                                D(str(cash)) / D(str(cost_per_share)),
                            )

                            quantity = quantity.quantize(
                                D("0.000001"), rounding=ROUND_DOWN
                            )

                        quantity = float(quantity)

                        cost = (
                            self.execution.entry_cost(quantity, entry)
                            if quantity > 0
                            else 0.0
                        )

                        minimum_order = float(self.guardian.config.min_order_value)

                        if (
                            quantity > 0
                            and quantity * entry >= minimum_order
                            and cost <= cash
                        ):
                            cash -= cost

                            position = {
                                "entry_date": str(timestamp),
                                "entry": entry,
                                "stop": stop,
                                "target": target,
                                "quantity": quantity,
                                "cost": cost,
                                "strategy": signal["strategy"],
                                "planned_risk": (quantity * float(risk_per_share)),
                                "entry_index": i,
                            }

            # EXIT EVALUATION
            # Use the dedicated execution simulator.

            if position is not None:
                exit_decision = self.execution.evaluate_long_exit(
                    opening=float(today["open"]),
                    high=float(today["high"]),
                    low=float(today["low"]),
                    close=float(today["close"]),
                    stop=position["stop"],
                    target=position["target"],
                    holding_days=(i - position["entry_index"] + 1),
                    maximum_holding_days=(self.config["maximum_holding_days"]),
                    final_session=(i == len(data) - 1),
                    forced_exit=pending_forced_exit,
                )

                if exit_decision.triggered:
                    proceeds, trade = close_position(
                        position, exit_decision.price, timestamp, exit_decision.reason
                    )

                    cash += proceeds

                    trades.append(trade)

                    position = None

                    pending_forced_exit = False

            # DAILY MARK-TO-MARKET EQUITY

            equity = cash

            if position is not None:
                equity += position["quantity"] * float(today["close"])

            peak_equity = max(peak_equity, equity)

            drawdown = (peak_equity - equity) / peak_equity * 100

            # DAILY LOSS MONITORING

            daily_loss = max(0.0, (day_start_equity - equity) / day_start_equity * 100)

            daily_limit = float(self.guardian.config.max_daily_loss_percent)

            if daily_loss >= daily_limit:
                daily_loss_events += 1

            # DRAWDOWN PROTECTION
            # Schedule liquidation for the next session.

            drawdown_limit = float(self.guardian.config.max_drawdown_percent)

            if drawdown >= drawdown_limit:
                if not trading_halted:
                    drawdown_events += 1

                trading_halted = True

                if position is not None:
                    pending_forced_exit = True

            equity_curve.append(equity)

            equity_dates.append(timestamp)

            previous_equity = equity

        # PERFORMANCE CALCULATIONS

        metrics = PerformanceMetrics.calculate(trades, equity_curve, initial)

        benchmark_entry = float(data.iloc[warmup]["open"])

        benchmark_exit = float(data.iloc[-1]["close"])

        metrics["benchmark_return"] = (benchmark_exit / benchmark_entry - 1) * 100

        metrics["symbol"] = symbol

        metrics["start_date"] = str(data.index[warmup])

        metrics["end_date"] = str(data.index[-1])

        metrics["trading_halted"] = trading_halted

        metrics["daily_loss_events"] = daily_loss_events

        metrics["drawdown_events"] = drawdown_events

        metrics["execution_version"] = "0.8.5B"

        curve = pd.DataFrame(
            {"timestamp": equity_dates, "equity": equity_curve, "symbol": symbol}
        )

        return metrics, trades, curve

    def run_all(self):

        summaries = []

        all_trades = []

        all_curves = []

        for symbol in self.config["symbols"]:
            try:
                metrics, trades, curve = self.run(symbol)

                summaries.append(metrics)

                all_trades.extend(trades)

                all_curves.append(curve)

                print(
                    f"Return: "
                    f"{metrics['total_return']:.2f}% | "
                    f"Trades: "
                    f"{metrics['total_trades']} | "
                    f"Win Rate: "
                    f"{metrics['win_rate']:.2f}% | "
                    f"Drawdown: "
                    f"{metrics['max_drawdown']:.2f}%"
                )

            except Exception as error:
                print(f"{symbol} FAILED: {type(error).__name__}: {error}")

        if not summaries:
            raise RuntimeError("No backtests completed successfully.")

        results = pd.DataFrame(summaries)

        results = results.sort_values("total_return", ascending=False)

        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

        DATA_DIR.mkdir(parents=True, exist_ok=True)

        # SAVE PERFORMANCE SUMMARY

        summary_file = DATA_DIR / f"backtest_summary_{timestamp}.csv"

        results.to_csv(summary_file, index=False)

        # SAVE TRADE HISTORY

        trade_columns = [
            "symbol",
            "strategy",
            "entry_date",
            "exit_date",
            "entry_price",
            "exit_price",
            "quantity",
            "stop_price",
            "target_price",
            "planned_risk",
            "pnl",
            "exit_reason",
        ]

        trade_file = DATA_DIR / f"backtest_trades_{timestamp}.csv"

        pd.DataFrame(all_trades, columns=trade_columns).to_csv(trade_file, index=False)

        # SAVE EQUITY HISTORY

        equity_file = DATA_DIR / f"backtest_equity_{timestamp}.csv"

        pd.concat(all_curves, ignore_index=True).to_csv(equity_file, index=False)

        print(f"\nSaved: {summary_file.name}")

        print(f"Saved: {trade_file.name}")

        print(f"Saved: {equity_file.name}")

        # DISPLAY PERFORMANCE SUMMARY

        print("\n" + "=" * 100)

        print("BACKTEST PERFORMANCE SUMMARY")

        print("=" * 100)

        columns = [
            "symbol",
            "final_capital",
            "total_return",
            "total_trades",
            "win_rate",
            "profit_factor",
            "max_drawdown",
            "benchmark_return",
            "daily_loss_events",
            "drawdown_events",
        ]

        print(results[columns].round(2).to_string(index=False))

        print(f"\nSuccessful backtests: {len(results)}/{len(self.config['symbols'])}")

        return results


if __name__ == "__main__":
    print("=" * 100)

    print("              JARVIS BACKTESTING ENGINE")

    print("                    VERSION 0.8.5B")

    print("=" * 100)

    print("Mode: HISTORICAL SIMULATION")

    print("Real Order Execution: DISABLED")

    engine = BacktestEngine()

    engine.run_all()

    print("\n" + "=" * 100)

    print("BACKTESTING COMPLETE")

    print("NO BROKER ORDERS EXECUTED")

    print("=" * 100)
