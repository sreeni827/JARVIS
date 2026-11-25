import numpy as np
import pandas as pd


class TechnicalIndicators:
    @staticmethod
    def calculate(data):

        if data.empty:
            raise ValueError("No historical data available.")

        df = data.copy().sort_index()

        if isinstance(df.index, pd.MultiIndex):
            if df.index.get_level_values(0).nunique() != 1:
                raise ValueError("Calculate indicators for one symbol at a time.")

        required = {"open", "high", "low", "close", "volume"}

        if not required.issubset(df.columns):
            raise ValueError("Required OHLCV columns are missing.")

        if len(df) < 200:
            raise ValueError("At least 200 daily candles are required.")

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # SIMPLE MOVING AVERAGES
        df["SMA_20"] = close.rolling(20).mean()
        df["SMA_50"] = close.rolling(50).mean()
        df["SMA_200"] = close.rolling(200).mean()

        # EXPONENTIAL MOVING AVERAGES
        df["EMA_9"] = close.ewm(span=9, adjust=False).mean()

        df["EMA_21"] = close.ewm(span=21, adjust=False).mean()

        # RELATIVE STRENGTH INDEX
        delta = close.diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

        avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)

        rsi = 100 - (100 / (1 + rs))

        rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)

        rsi = rsi.mask((avg_loss == 0) & (avg_gain == 0), 50)

        df["RSI_14"] = rsi

        # MACD
        ema_12 = close.ewm(span=12, adjust=False).mean()

        ema_26 = close.ewm(span=26, adjust=False).mean()

        df["MACD"] = ema_12 - ema_26

        df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()

        df["MACD_HISTOGRAM"] = df["MACD"] - df["MACD_SIGNAL"]

        # BOLLINGER BANDS
        df["BB_MIDDLE"] = close.rolling(20).mean()

        std = close.rolling(20).std(ddof=0)

        df["BB_UPPER"] = df["BB_MIDDLE"] + (2 * std)
        df["BB_LOWER"] = df["BB_MIDDLE"] - (2 * std)

        # AVERAGE TRUE RANGE
        previous_close = close.shift(1)

        true_range = pd.concat(
            [high - low, (high - previous_close).abs(), (low - previous_close).abs()],
            axis=1,
        ).max(axis=1)

        df["ATR_14"] = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

        # RELATIVE VOLUME
        df["AVG_VOLUME_20"] = volume.shift(1).rolling(20).mean()

        df["RELATIVE_VOLUME"] = volume / df["AVG_VOLUME_20"].replace(0, np.nan)

        # RATE OF CHANGE
        df["ROC_10"] = close.pct_change(10) * 100

        # SUPPORT AND RESISTANCE
        df["SUPPORT"] = low.shift(1).rolling(20).min()

        df["RESISTANCE"] = high.shift(1).rolling(20).max()

        # TREND CLASSIFICATION
        bullish = (close > df["SMA_20"]) & (df["SMA_20"] > df["SMA_50"])

        bearish = (close < df["SMA_20"]) & (df["SMA_20"] < df["SMA_50"])

        df["TREND"] = np.select(
            [bullish, bearish], ["BULLISH", "BEARISH"], default="NEUTRAL"
        )

        return df
