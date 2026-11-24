from pathlib import Path
from datetime import datetime, timezone
import json

from risk.risk_guardian import RiskGuardian


BASE_DIR = Path(__file__).resolve().parents[1]

LOG_FILE = (
    BASE_DIR
    / "logs"
    / "backtesting_risk.jsonl"
)


class SimulationRiskGuardian(RiskGuardian):

    def log_decision(self, decision):

        LOG_FILE.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        record = {
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),

            "mode": "HISTORICAL_SIMULATION",

            "symbol": decision.symbol,

            "approved": decision.approved,

            "reason": decision.reason,

            "quantity": str(
                decision.quantity
            ),

            "position_value": str(
                decision.position_value
            ),

            "estimated_risk": str(
                decision.estimated_risk
            )
        }

        with open(
            LOG_FILE,
            "a",
            encoding="utf-8"
        ) as file:

            file.write(
                json.dumps(record) + "\n"
            )

        return decision