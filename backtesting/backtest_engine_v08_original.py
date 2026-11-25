from datetime import UTC, datetime
from decimal import ROUND_DOWN, Decimal
from pathlib import Path

import pandas as pd
import yaml

from backtesting.performance import PerformanceMetrics
from intelligence.opportunity_ranker import OpportunityRanker
from market.indicators import TechnicalIndicators
from market.market_data import MarketDataEngine
from risk.risk_guardian import AccountState, RiskGuardian, TradeRequest
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

        self.guardian = RiskGuardian()

        self.slippage = self.config["slippage_percent"] / 100

        self.fee = self.config["transaction_fee_percent"] / 100

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

        return data.sort_index()

    def run(self, symbol):

        print(f"\nBacktesting {symbol}...")

        data = self.prepare_data(symbol)

        warmup = self.config["warmup_candles"]

        if len(data) <= warmup + 1:
            raise ValueError("Insufficient historical data.")

        initial = float(self.config["initial_capital"])

        cash = initial

        peak_equity = initial

        equity_curve = [initial]

        equity_dates = [data.index[warmup - 1]]

        trades = []

        position = None

        trading_halted = False

        previous_equity = initial

        def close_position(position, raw_exit_price, timestamp, reason):

            exit_price = raw_exit_price * (1 - self.slippage)

            proceeds = position["quantity"] * exit_price * (1 - self.fee)

            pnl = proceeds - position["cost"]

            record = {
                "symbol": symbol,
                "strategy": position["strategy"],
                "entry_date": position["entry_date"],
                "exit_date": str(timestamp),
                "entry_price": position["entry"],
                "exit_price": exit_price,
                "quantity": position["quantity"],
                "pnl": pnl,
                "exit_reason": reason,
            }

            return proceeds, record

        for i in range(warmup, len(data)):
            today = data.iloc[i]

            yesterday = data.iloc[i - 1]

            timestamp = data.index[i]

            day_start_equity = previous_equity

            # GENERATE ENTRY USING PREVIOUS DAY DATA

            if position is None and not trading_halted:
                row = yesterday.copy()

                score, _, _ = self.ranker.calculate_score(row)

                row["score"] = score

                signal = self.strategy.generate_signal(row)

                if signal["signal"] == "BUY":
                    entry = float(today["open"]) * (1 + self.slippage)

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

                        # Include estimated transaction costs
                        # in the planned risk per share.

                        risk_per_share = D(
                            str(
                                entry * (1 + self.fee)
                                - stop * (1 - self.slippage) * (1 - self.fee)
                            )
                        )

                        if risk_per_share > 0:
                            quantity = min(
                                quantity,
                                decision.estimated_risk / risk_per_share,
                                D(str(cash)) / D(str(entry * (1 + self.fee))),
                            )

                            quantity = quantity.quantize(
                                D("0.000001"), rounding=ROUND_DOWN
                            )

                        quantity = float(quantity)

                        cost = quantity * entry * (1 + self.fee)

                        if (
                            quantity > 0
                            and quantity * entry
                            >= float(self.guardian.config.min_order_value)
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
                                "entry_index": i,
                            }

            # MONITOR OPEN POSITION

            if position is not None:
                raw_exit = None

                reason = None

                stop = position["stop"]

                target = position["target"]

                opening = float(today["open"])

                high = float(today["high"])

                low = float(today["low"])

                close = float(today["close"])

                # Price gaps below the stop.

                if opening <= stop:
                    raw_exit = opening

                    reason = "STOP_GAP"

                # Price gaps above the target.

                elif opening >= target:
                    raw_exit = target

                    reason = "TAKE_PROFIT_GAP"

                # If both levels are touched,
                # conservatively assume stop first.

                elif low <= stop:
                    raw_exit = stop

                    reason = "STOP_LOSS"

                elif high >= target:
                    raw_exit = target

                    reason = "TAKE_PROFIT"

                elif (
                    i - position["entry_index"] + 1
                    >= self.config["maximum_holding_days"]
                ):
                    raw_exit = close

                    reason = "TIME_EXIT"

                elif i == len(data) - 1:
                    raw_exit = close

                    reason = "END_OF_BACKTEST"

                if raw_exit is not None:
                    proceeds, trade = close_position(
                        position, raw_exit, timestamp, reason
                    )

                    cash += proceeds

                    trades.append(trade)

                    position = None

            # DAILY MARK-TO-MARKET EQUITY

            equity = cash

            if position is not None:
                equity += position["quantity"] * float(today["close"])

            peak_equity = max(peak_equity, equity)

            drawdown = (peak_equity - equity) / peak_equity * 100

            if drawdown >= float(self.guardian.config.max_drawdown_percent):
                trading_halted = True

                if position is not None:
                    proceeds, trade = close_position(
                        position, float(today["close"]), timestamp, "DRAWDOWN_EXIT"
                    )

                    cash += proceeds

                    trades.append(trade)

                    position = None

                    equity = cash

            equity_curve.append(equity)

            equity_dates.append(timestamp)

            previous_equity = equity

        metrics = PerformanceMetrics.calculate(trades, equity_curve, initial)

        benchmark_entry = float(data.iloc[warmup]["open"])

        benchmark_exit = float(data.iloc[-1]["close"])

        metrics["benchmark_return"] = (benchmark_exit / benchmark_entry - 1) * 100

        metrics["symbol"] = symbol

        metrics["start_date"] = str(data.index[warmup])

        metrics["end_date"] = str(data.index[-1])

        metrics["trading_halted"] = trading_halted

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

        results.to_csv(DATA_DIR / f"backtest_summary_{timestamp}.csv", index=False)

        pd.DataFrame(all_trades).to_csv(
            DATA_DIR / f"backtest_trades_{timestamp}.csv", index=False
        )

        pd.concat(all_curves, ignore_index=True).to_csv(
            DATA_DIR / f"backtest_equity_{timestamp}.csv", index=False
        )

        print("\n" + "=" * 85)

        print("BACKTEST PERFORMANCE SUMMARY")

        print("=" * 85)

        columns = [
            "symbol",
            "final_capital",
            "total_return",
            "total_trades",
            "win_rate",
            "profit_factor",
            "max_drawdown",
            "benchmark_return",
        ]

        print(results[columns].round(2).to_string(index=False))

        print(f"\nSuccessful backtests: {len(results)}/{len(self.config['symbols'])}")

        return results


if __name__ == "__main__":
    print("=" * 85)

    print("                  JARVIS BACKTESTING ENGINE")

    print("                          VERSION 0.8")

    print("=" * 85)

    print("Mode: HISTORICAL SIMULATION")

    print("Real Order Execution: DISABLED")

    engine = BacktestEngine()

    engine.run_all()

    print("\n" + "=" * 85)

    print("BACKTESTING COMPLETE")

    print("NO BROKER ORDERS EXECUTED")

    print("=" * 85)
