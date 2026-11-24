from market.market_data import MarketDataEngine
from market.indicators import TechnicalIndicators


class TechnicalAgent:

    def __init__(self):

        self.market = MarketDataEngine()

    def analyze(self, symbol):

        history = self.market.get_historical_data(
            symbol,
            days=400
        )

        data = TechnicalIndicators.calculate(history)

        latest = data.iloc[-1]

        return {
            "symbol": symbol,
            "timestamp": str(latest.name),
            "close": float(latest["close"]),
            "sma_20": float(latest["SMA_20"]),
            "sma_50": float(latest["SMA_50"]),
            "sma_200": float(latest["SMA_200"]),
            "ema_9": float(latest["EMA_9"]),
            "ema_21": float(latest["EMA_21"]),
            "rsi": float(latest["RSI_14"]),
            "macd": float(latest["MACD"]),
            "macd_signal": float(latest["MACD_SIGNAL"]),
            "bollinger_upper": float(latest["BB_UPPER"]),
            "bollinger_lower": float(latest["BB_LOWER"]),
            "atr": float(latest["ATR_14"]),
            "relative_volume": float(latest["RELATIVE_VOLUME"]),
            "momentum": float(latest["ROC_10"]),
            "support": float(latest["SUPPORT"]),
            "resistance": float(latest["RESISTANCE"]),
            "trend": str(latest["TREND"])
        }


if __name__ == "__main__":

    agent = TechnicalAgent()

    symbol = "NVDA"

    print("=" * 60)
    print("          JARVIS TECHNICAL ANALYSIS AGENT")
    print("                    VERSION 0.3")
    print("=" * 60)

    result = agent.analyze(symbol)

    print(f"\nSymbol           : {result['symbol']}")
    print(f"Data Timestamp   : {result['timestamp']}")
    print(f"Last Daily Close : ${result['close']:.2f}")
    print(f"Trend            : {result['trend']}")

    print("\nMOVING AVERAGES")
    print("-" * 60)

    print(f"SMA 20           : ${result['sma_20']:.2f}")
    print(f"SMA 50           : ${result['sma_50']:.2f}")
    print(f"SMA 200          : ${result['sma_200']:.2f}")
    print(f"EMA 9            : ${result['ema_9']:.2f}")
    print(f"EMA 21           : ${result['ema_21']:.2f}")

    print("\nMOMENTUM")
    print("-" * 60)

    print(f"RSI              : {result['rsi']:.2f}")
    print(f"MACD             : {result['macd']:.2f}")
    print(f"MACD Signal      : {result['macd_signal']:.2f}")
    print(f"Rate of Change   : {result['momentum']:.2f}%")

    print("\nVOLATILITY")
    print("-" * 60)

    print(f"ATR              : ${result['atr']:.2f}")
    print(f"Bollinger Upper  : ${result['bollinger_upper']:.2f}")
    print(f"Bollinger Lower  : ${result['bollinger_lower']:.2f}")

    print("\nVOLUME AND PRICE LEVELS")
    print("-" * 60)

    print(f"Relative Volume  : {result['relative_volume']:.2f}x")
    print(f"Support          : ${result['support']:.2f}")
    print(f"Resistance       : ${result['resistance']:.2f}")

    print("\n" + "=" * 60)
    print("      TECHNICAL ANALYSIS ENGINE: SUCCESSFUL")
    print("=" * 60)