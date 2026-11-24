import pandas as pd


class PerformanceMetrics:

    @staticmethod
    def calculate(
        trades,
        equity_curve,
        initial_capital
    ):

        equity = pd.Series(
            equity_curve,
            dtype=float
        )

        final_capital = float(equity.iloc[-1])

        net_profit = (
            final_capital - initial_capital
        )

        total_return = (
            net_profit / initial_capital * 100
        )

        running_peak = equity.cummax()

        drawdown = (
            (running_peak - equity)
            / running_peak
            * 100
        )

        max_drawdown = float(drawdown.max())

        pnl = pd.Series(
            [trade["pnl"] for trade in trades],
            dtype=float
        )

        winners = pnl[pnl > 0]

        losers = pnl[pnl < 0]

        total_trades = len(pnl)

        gross_profit = float(winners.sum())

        gross_loss = abs(float(losers.sum()))

        win_rate = (
            len(winners) / total_trades * 100
            if total_trades else 0
        )

        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss

        elif gross_profit > 0:
            profit_factor = float("inf")

        else:
            profit_factor = 0

        return {

            "initial_capital": initial_capital,

            "final_capital": final_capital,

            "net_profit": net_profit,

            "total_return": total_return,

            "total_trades": total_trades,

            "winning_trades": len(winners),

            "losing_trades": len(losers),

            "breakeven_trades": int((pnl == 0).sum()),

            "win_rate": win_rate,

            "gross_profit": gross_profit,

            "gross_loss": gross_loss,

            "profit_factor": profit_factor,

            "average_win": (
                float(winners.mean())
                if len(winners) else 0
            ),

            "average_loss": (
                float(losers.mean())
                if len(losers) else 0
            ),

            "expectancy": (
                float(pnl.mean())
                if total_trades else 0
            ),

            "max_drawdown": max_drawdown

        }