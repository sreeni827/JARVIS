from pathlib import Path
from itertools import product

import json
import numpy as np
import pandas as pd
import yaml

from backtesting.parameter_selector import ParameterSelector


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"

CONFIG_FILE = BASE_DIR / "config" / "walk_forward.yaml"


class WalkForwardAnalyzer:

    def __init__(self):

        with open(
            CONFIG_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            self.config = yaml.safe_load(file)["walk_forward"]

        self.selector = ParameterSelector(self.config)

        self.symbols = self.config["symbols"]

        self.run_id = None

        self.windows = None
        self.training = None
        self.selections = None
        self.testing = None
        self.trades = None
        self.equity = None


    @staticmethod
    def boolean_series(series):

        values = (
            series.astype(str)
            .str.strip()
            .str.lower()
        )

        if not values.isin(["true", "false"]).all():

            raise ValueError(
                "Invalid boolean values detected."
            )

        return values.eq("true")


    @staticmethod
    def require_columns(frame, columns, name):

        missing = set(columns) - set(frame.columns)

        if missing:

            raise ValueError(
                f"{name} missing columns: {sorted(missing)}"
            )


    def load_reports(self):

        files = sorted(
            DATA_DIR.glob(
                "walk_forward_training_*.csv"
            )
        )

        if not files:

            raise FileNotFoundError(
                "No walk-forward research reports found."
            )

        latest = files[-1]

        self.run_id = latest.stem.removeprefix(
            "walk_forward_training_"
        )

        reports = {

            "windows": "walk_forward_windows",

            "training": "walk_forward_training",

            "selections": "walk_forward_selections",

            "testing": "walk_forward_testing_summary",

            "trades": "walk_forward_testing_trades",

            "equity": "walk_forward_testing_equity"

        }

        for attribute, prefix in reports.items():

            path = DATA_DIR / (
                f"{prefix}_{self.run_id}.csv"
            )

            if not path.exists():

                raise FileNotFoundError(path)

            setattr(
                self,
                attribute,
                pd.read_csv(
                    path,
                    keep_default_na=False
                )
            )

        error_file = DATA_DIR / (
            f"walk_forward_errors_{self.run_id}.json"
        )

        with open(
            error_file,
            "r",
            encoding="utf-8"
        ) as file:

            errors = json.load(file)

        if errors != []:

            raise RuntimeError(
                "Research run contains execution errors."
            )

        self.validate_reports()

        print(f"Research run: {self.run_id}")

        print("Report integrity verification: PASSED")


    def validate_reports(self):

        self.require_columns(

            self.windows,

            ["window_id"],

            "Windows"

        )

        self.require_columns(

            self.training,

            [
                "window_id",
                "candidate_id",
                "minimum_score",
                "stop_atr_multiplier",
                "risk_reward_ratio",
                "total_trades",
                "average_return",
                "profit_factor",
                "expectancy",
                "maximum_drawdown",
                "risk_breach_percent",
                "qualified",
                "data_role",
                "trading_authorized"
            ],

            "Training"

        )

        self.require_columns(

            self.selections,

            [
                "window_id",
                "candidate_id",
                "status",
                "qualified_candidates",
                "trading_authorized"
            ],

            "Selections"

        )

        self.require_columns(

            self.testing,

            [
                "window_id",
                "candidate_id",
                "symbol",
                "total_trades",
                "initial_capital",
                "final_capital",
                "net_profit",
                "total_return",
                "max_drawdown",
                "period_start",
                "period_end",
                "data_role",
                "trading_authorized"
            ],

            "Testing"

        )

        self.require_columns(

            self.trades,

            [
                "window_id",
                "candidate_id",
                "symbol",
                "strategy",
                "pnl",
                "planned_risk",
                "entry_date",
                "exit_date",
                "exit_reason",
                "data_role",
                "trading_authorized"
            ],

            "Trades"

        )

        self.require_columns(

            self.equity,

            [
                "window_id",
                "candidate_id",
                "symbol",
                "timestamp",
                "equity",
                "data_role",
                "trading_authorized"
            ],

            "Equity"

        )

        if self.windows["window_id"].duplicated().any():

            raise ValueError(
                "Duplicate research windows."
            )

        window_ids = self.windows["window_id"].tolist()

        candidates = self.selector.generate_candidates()

        candidate_ids = [
            candidate["candidate_id"]
            for candidate in candidates
        ]

        expected_training = set(
            product(window_ids, candidate_ids)
        )

        actual_training = set(
            self.training[
                ["window_id", "candidate_id"]
            ].itertuples(index=False, name=None)
        )

        if (
            actual_training != expected_training
            or len(self.training) != len(expected_training)
        ):

            raise ValueError(
                "Incomplete or duplicate training evaluations."
            )

        expected_testing = set(
            product(window_ids, self.symbols)
        )

        actual_testing = set(
            self.testing[
                ["window_id", "symbol"]
            ].itertuples(index=False, name=None)
        )

        if (
            actual_testing != expected_testing
            or len(self.testing) != len(expected_testing)
        ):

            raise ValueError(
                "Incomplete or duplicate testing records."
            )

        if (
            len(self.selections) != len(window_ids)
            or set(self.selections["window_id"]) != set(window_ids)
        ):

            raise ValueError(
                "Invalid selection coverage."
            )

        for frame in [

            self.training,
            self.selections,
            self.testing,
            self.trades,
            self.equity

        ]:

            if self.boolean_series(
                frame["trading_authorized"]
            ).any():

                raise RuntimeError(
                    "Unexpected trading authorization."
                )

        if not self.training["data_role"].eq(
            "TRAINING"
        ).all():

            raise ValueError(
                "Invalid training data role."
            )

        for frame in [
            self.testing,
            self.trades,
            self.equity
        ]:

            if not frame["data_role"].eq(
                "TESTING"
            ).all():

                raise ValueError(
                    "Invalid testing data role."
                )

        self.training["qualified"] = self.boolean_series(
            self.training["qualified"]
        )

        training_numeric = [

            "minimum_score",
            "stop_atr_multiplier",
            "risk_reward_ratio",
            "total_trades",
            "average_return",
            "profit_factor",
            "expectancy",
            "maximum_drawdown",
            "risk_breach_percent"

        ]

        for column in training_numeric:

            self.training[column] = pd.to_numeric(
                self.training[column],
                errors="raise"
            )

        for column in training_numeric:

            values = self.training[column].to_numpy(
                dtype=float
            )

            if column == "profit_factor":

                valid = (
                    ~np.isnan(values)
                    & (values >= 0)
                )

            else:

                valid = np.isfinite(values)

            if not valid.all():

                raise ValueError(
                    f"Invalid training metric: {column}"
                )

        for candidate in candidates:

            rows = self.training[
                self.training["candidate_id"]
                == candidate["candidate_id"]
            ]

            for parameter in [

                "minimum_score",
                "stop_atr_multiplier",
                "risk_reward_ratio"

            ]:

                if not np.allclose(
                    rows[parameter],
                    candidate[parameter],
                    atol=1e-9,
                    rtol=0
                ):

                    raise ValueError(
                        "Candidate parameter mismatch."
                    )

        for _, row in self.training.iterrows():

            qualified, _ = (
                self.selector.qualify_candidate(
                    row.to_dict()
                )
            )

            if qualified != bool(row["qualified"]):

                raise ValueError(
                    "Candidate qualification mismatch."
                )

        self.validate_selections()

        self.validate_testing_accounting()


    def validate_selections(self):

        for _, selection in self.selections.iterrows():

            window_id = selection["window_id"]

            training = self.training[
                self.training["window_id"] == window_id
            ]

            expected = self.selector.select(
                training.to_dict("records")
            )

            selected = expected["selected_candidate"]

            expected_id = (

                selected["candidate_id"]

                if selected is not None

                else "NONE"

            )

            if selection["status"] != expected["status"]:

                raise ValueError(
                    f"Selection status mismatch: {window_id}"
                )

            if selection["candidate_id"] != expected_id:

                raise ValueError(
                    f"Selected candidate mismatch: {window_id}"
                )

            qualified_count = int(
                training["qualified"].sum()
            )

            if int(
                selection["qualified_candidates"]
            ) != qualified_count:

                raise ValueError(
                    "Qualified candidate count mismatch."
                )

            for frame in [

                self.testing,
                self.trades,
                self.equity

            ]:

                rows = frame[
                    frame["window_id"] == window_id
                ]

                if not rows["candidate_id"].eq(
                    expected_id
                ).all():

                    raise ValueError(
                        "Testing candidate mismatch."
                    )


    def validate_testing_accounting(self):

        keys = ["window_id", "symbol"]

        numeric = [

            "total_trades",
            "initial_capital",
            "final_capital",
            "net_profit",
            "total_return",
            "max_drawdown"

        ]

        for column in numeric:

            self.testing[column] = pd.to_numeric(
                self.testing[column],
                errors="raise"
            )

            if not np.isfinite(
                self.testing[column]
            ).all():

                raise ValueError(
                    f"Invalid testing metric: {column}"
                )

        for column in ["pnl", "planned_risk"]:

            self.trades[column] = pd.to_numeric(
                self.trades[column],
                errors="raise"
            )

            if not np.isfinite(
                self.trades[column]
            ).all():

                raise ValueError(
                    f"Invalid trade metric: {column}"
                )

        if (self.trades["planned_risk"] <= 0).any():

            raise ValueError(
                "Invalid planned trade risk."
            )

        self.equity["equity"] = pd.to_numeric(
            self.equity["equity"],
            errors="raise"
        )

        if (
            not np.isfinite(
                self.equity["equity"]
            ).all()
            or (self.equity["equity"] <= 0).any()
        ):

            raise ValueError(
                "Invalid testing equity."
            )

        expected_keys = set(
            self.testing[keys].itertuples(
                index=False,
                name=None
            )
        )

        trade_keys = set(
            self.trades[keys].itertuples(
                index=False,
                name=None
            )
        )

        equity_keys = set(
            self.equity[keys].itertuples(
                index=False,
                name=None
            )
        )

        if not trade_keys.issubset(expected_keys):

            raise ValueError(
                "Unexpected testing trade records."
            )

        if equity_keys != expected_keys:

            raise ValueError(
                "Incomplete equity coverage."
            )

        for _, row in self.testing.iterrows():

            window_id = row["window_id"]
            symbol = row["symbol"]

            trades = self.trades[

                (self.trades["window_id"] == window_id)

                & (self.trades["symbol"] == symbol)

            ]

            equity = self.equity[

                (self.equity["window_id"] == window_id)

                & (self.equity["symbol"] == symbol)

            ].copy()

            equity["timestamp"] = pd.to_datetime(
                equity["timestamp"],
                utc=True,
                errors="raise"
            )

            equity = equity.sort_values("timestamp")

            if equity["timestamp"].duplicated().any():

                raise ValueError(
                    "Duplicate equity timestamps."
                )

            if len(trades) != row["total_trades"]:

                raise ValueError(
                    "Testing trade count mismatch."
                )

            pnl = float(trades["pnl"].sum())

            initial = float(row["initial_capital"])

            final = float(row["final_capital"])

            if initial <= 0 or final <= 0:

                raise ValueError(
                    "Invalid account capital."
                )

            if not np.isclose(
                pnl,
                row["net_profit"],
                atol=1e-6,
                rtol=0
            ):

                raise ValueError(
                    "Testing PnL mismatch."
                )

            if not np.isclose(
                initial + pnl,
                final,
                atol=1e-6,
                rtol=0
            ):

                raise ValueError(
                    "Final capital mismatch."
                )

            if not np.isclose(
                equity["equity"].iloc[-1],
                final,
                atol=1e-6,
                rtol=0
            ):

                raise ValueError(
                    "Final equity mismatch."
                )

            calculated_return = (
                (final / initial - 1) * 100
            )

            if not np.isclose(
                calculated_return,
                row["total_return"],
                atol=1e-6,
                rtol=0
            ):

                raise ValueError(
                    "Testing return mismatch."
                )

            curve = np.concatenate(
                [
                    [initial],
                    equity["equity"].to_numpy()
                ]
            )

            peaks = np.maximum.accumulate(curve)

            drawdown = float(
                np.max(
                    (peaks - curve) / peaks * 100
                )
            )

            if not np.isclose(
                drawdown,
                row["max_drawdown"],
                atol=1e-6,
                rtol=0
            ):

                raise ValueError(
                    "Testing drawdown mismatch."
                )

            if not trades.empty:

                entry = pd.to_datetime(
                    trades["entry_date"],
                    utc=True
                ).dt.tz_convert(
                    "America/New_York"
                ).dt.date

                exit_dates = pd.to_datetime(
                    trades["exit_date"],
                    utc=True
                ).dt.tz_convert(
                    "America/New_York"
                ).dt.date

                start = pd.Timestamp(
                    row["period_start"]
                ).date()

                end = pd.Timestamp(
                    row["period_end"]
                ).date()

                if not (

                    (entry >= start)

                    & (exit_dates <= end)

                    & (exit_dates >= entry)

                ).all():

                    raise ValueError(
                        "Trade outside testing period."
                    )

            if row["candidate_id"] == "NONE":

                if (
                    len(trades) != 0
                    or not np.allclose(
                        curve,
                        initial,
                        atol=1e-6,
                        rtol=0
                    )
                ):

                    raise ValueError(
                        "Invalid cash-only simulation."
                    )


    @staticmethod
    def calculate_trade_metrics(trades):

        pnl = trades["pnl"]

        winners = pnl[pnl > 0]

        losers = pnl[pnl < 0]

        gross_profit = float(winners.sum())

        gross_loss = float(-losers.sum())

        total = len(trades)

        if gross_loss > 0:

            profit_factor = gross_profit / gross_loss

        elif gross_profit > 0:

            profit_factor = float("inf")

        else:

            profit_factor = 0.0

        breaches = (

            (-pnl).clip(lower=0)

            > trades["planned_risk"] + 1e-6

        )

        return {

            "total_trades": total,

            "winning_trades": len(winners),

            "losing_trades": len(losers),

            "win_rate": (
                len(winners) / total * 100
                if total else 0.0
            ),

            "net_profit": float(pnl.sum()),

            "profit_factor": profit_factor,

            "expectancy": (
                float(pnl.mean())
                if total else 0.0
            ),

            "risk_breaches": int(breaches.sum()),

            "risk_breach_percent": (
                float(breaches.mean() * 100)
                if total else 0.0
            )

        }


    def analyze_windows(self):

        results = []

        for _, selection in self.selections.iterrows():

            window_id = selection["window_id"]

            candidate_id = selection["candidate_id"]

            testing = self.testing[
                self.testing["window_id"] == window_id
            ]

            trades = self.trades[
                self.trades["window_id"] == window_id
            ]

            metrics = self.calculate_trade_metrics(
                trades
            )

            testing_return = float(
                testing["total_return"].mean()
            )

            training_return = None

            deterioration = None

            if candidate_id != "NONE":

                training = self.training[

                    (self.training["window_id"] == window_id)

                    & (
                        self.training["candidate_id"]
                        == candidate_id
                    )

                ].iloc[0]

                training_return = float(
                    training["average_return"]
                )

                deterioration = (
                    training_return - testing_return
                )

            results.append({

                "window_id": window_id,

                "candidate_id": candidate_id,

                "status": selection["status"],

                "qualified_candidates": int(
                    selection["qualified_candidates"]
                ),

                "training_average_return": training_return,

                "testing_average_return": testing_return,

                "performance_deterioration": deterioration,

                "maximum_drawdown": float(
                    testing["max_drawdown"].max()
                ),

                **metrics,

                "trading_authorized": False

            })

        return pd.DataFrame(results)


    def analyze_parameter_stability(self):

        results = []

        for candidate_id, group in self.training.groupby(
            "candidate_id"
        ):

            first = group.iloc[0]

            qualified = int(
                group["qualified"].sum()
            )

            selected = int(

                (
                    self.selections["candidate_id"]
                    == candidate_id
                ).sum()

            )

            results.append({

                "candidate_id": candidate_id,

                "minimum_score": first["minimum_score"],

                "stop_atr_multiplier": first[
                    "stop_atr_multiplier"
                ],

                "risk_reward_ratio": first[
                    "risk_reward_ratio"
                ],

                "windows_evaluated": len(group),

                "qualified_windows": qualified,

                "qualification_rate": (
                    qualified / len(group) * 100
                ),

                "selected_windows": selected,

                "average_training_return": float(
                    group["average_return"].mean()
                ),

                "worst_training_return": float(
                    group["average_return"].min()
                ),

                "maximum_drawdown": float(
                    group["maximum_drawdown"].max()
                ),

                "trading_authorized": False

            })

        return pd.DataFrame(results).sort_values(

            [
                "qualified_windows",
                "average_training_return"
            ],

            ascending=False

        )


    def analyze_parameter_effectiveness(self):

        results = []

        parameters = [

            "minimum_score",

            "stop_atr_multiplier",

            "risk_reward_ratio"

        ]

        for parameter in parameters:

            controls = [

                "window_id",

                *[
                    name
                    for name in parameters
                    if name != parameter
                ]

            ]

            for group_key, group in self.training.groupby(
                controls
            ):

                unique_returns = group[
                    "average_return"
                ].round(10).nunique()

                unique_counts = group[
                    "total_trades"
                ].nunique()

                unique_factors = group[
                    "profit_factor"
                ].round(10).nunique()

                distinct = max(

                    unique_returns,

                    unique_counts,

                    unique_factors

                )

                results.append({

                    "parameter": parameter,

                    "comparison_group": str(group_key),

                    "candidates": len(group),

                    "unique_returns": unique_returns,

                    "unique_trade_counts": unique_counts,

                    "unique_profit_factors": unique_factors,

                    "status": (

                        "EFFECT_OBSERVED"

                        if distinct > 1

                        else "NO_EFFECT_OBSERVED"

                    )

                })

        return pd.DataFrame(results)


    def analyze_strategies(self):

        results = []

        for strategy, trades in self.trades.groupby(
            "strategy"
        ):

            results.append({

                "strategy": strategy,

                **self.calculate_trade_metrics(trades),

                "trading_authorized": False

            })

        columns = [

            "strategy",
            "total_trades",
            "winning_trades",
            "losing_trades",
            "win_rate",
            "net_profit",
            "profit_factor",
            "expectancy",
            "risk_breaches",
            "risk_breach_percent",
            "trading_authorized"

        ]

        return pd.DataFrame(
            results,
            columns=columns
        )


    def save_report(self, frame, name):

        path = DATA_DIR / (
            f"walk_forward_{name}_{self.run_id}.csv"
        )

        frame.to_csv(
            path,
            index=False
        )

        print(f"Saved: {path.name}")


    def run(self):

        print("=" * 80)

        print("JARVIS WALK-FORWARD INTELLIGENCE ENGINE")

        print("VERSION 0.8.5F.7")

        print("=" * 80)

        self.load_reports()

        windows = self.analyze_windows()

        stability = self.analyze_parameter_stability()

        effectiveness = self.analyze_parameter_effectiveness()

        strategies = self.analyze_strategies()

        active = windows[
            windows["candidate_id"] != "NONE"
        ]

        cash_only = windows[
            windows["candidate_id"] == "NONE"
        ]

        profitable = active[
            active["testing_average_return"] > 0
        ]

        print("\nWINDOW PERFORMANCE\n")

        print(
            windows.round(4).to_string(
                index=False
            )
        )

        print("\nPARAMETER STABILITY\n")

        print(
            stability.round(4).to_string(
                index=False
            )
        )

        print("\nPARAMETER EFFECTIVENESS\n")

        print(

            effectiveness.groupby(

                ["parameter", "status"]

            ).size().to_string()

        )

        print("\nTESTING STRATEGY PERFORMANCE\n")

        print(
            strategies.round(4).to_string(
                index=False
            )
        )

        print("\nRESEARCH SUMMARY\n")

        print("Research windows:", len(windows))

        print("Training evaluations:", len(self.training))

        print(
            "Qualified evaluations:",
            int(self.training["qualified"].sum())
        )

        print("Active windows:", len(active))

        print("Cash-only windows:", len(cash_only))

        print("Profitable active windows:", len(profitable))

        print("Testing trades:", len(self.trades))

        print(

            "Average testing window return:",

            round(
                windows["testing_average_return"].mean(),
                4
            )

        )

        if not active.empty:

            print(

                "Average active testing return:",

                round(
                    active["testing_average_return"].mean(),
                    4
                )

            )

        print("\nAGGREGATE TESTING TRADE METRICS\n")

        metrics = self.calculate_trade_metrics(
            self.trades
        )

        for name, value in metrics.items():

            print(f"{name}: {value}")

        print("\nRESEARCH ASSESSMENT\n")

        if active.empty:

            assessment = "NO_QUALIFIED_CONFIGURATIONS"

        elif profitable.empty:

            assessment = "NO_PROFITABLE_TESTING_WINDOWS"

        else:

            assessment = "FURTHER_VALIDATION_REQUIRED"

        print(assessment)

        print("\nGENERATING INTELLIGENCE REPORTS\n")

        self.save_report(
            windows,
            "window_analysis"
        )

        self.save_report(
            stability,
            "parameter_stability"
        )

        self.save_report(
            effectiveness,
            "parameter_effectiveness"
        )

        self.save_report(
            strategies,
            "strategy_analysis"
        )

        print("\nFORWARD VALIDATION REQUIRED")

        print("NO STRATEGIES AUTHORIZED FOR TRADING")

        print("NO BROKER ORDERS EXECUTED")

        print("WALK-FORWARD INTELLIGENCE COMPLETE")


if __name__ == "__main__":

    WalkForwardAnalyzer().run()