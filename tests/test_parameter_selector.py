from copy import deepcopy

import pytest

from backtesting.parameter_selector import ParameterSelector


@pytest.fixture
def config():

    return {
        "symbols": ["AAA", "BBB"],
        "parameter_candidates": {
            "minimum_score": [55, 65, 75],
            "stop_atr_multiplier": [1.5, 2.0],
            "risk_reward_ratio": [2.0, 3.0],
        },
        "selection": {
            "minimum_training_trades": 30,
            "minimum_profit_factor": 1.25,
            "minimum_expectancy": 0.0,
            "minimum_average_return_percent": 0.0,
            "maximum_drawdown_percent": 5.0,
            "maximum_risk_breach_percent": 10.0,
            "allow_no_selection": True,
        },
        "safety": {
            "historical_simulation_only": True,
            "live_trading_enabled": False,
            "automatic_strategy_activation": False,
            "require_forward_validation": True,
            "allow_test_data_for_selection": False,
        },
    }


@pytest.fixture
def selector(config):

    return ParameterSelector(config)


@pytest.fixture
def valid_metrics():

    return {
        "total_trades": 40,
        "profit_factor": 1.8,
        "expectancy": 0.15,
        "average_return": 2.0,
        "maximum_drawdown": 3.0,
        "risk_breach_percent": 5.0,
    }


@pytest.fixture
def training_data():

    summaries = [
        {
            "symbol": "AAA",
            "total_return": 0.6,
            "total_trades": 2,
            "max_drawdown": 1.0,
            "net_profit": 0.6,
        },
        {
            "symbol": "BBB",
            "total_return": 0.3,
            "total_trades": 2,
            "max_drawdown": 2.0,
            "net_profit": 0.3,
        },
    ]

    trades = [
        {"symbol": "AAA", "pnl": 1.0, "planned_risk": 0.5},
        {"symbol": "AAA", "pnl": -0.4, "planned_risk": 0.5},
        {"symbol": "BBB", "pnl": 0.9, "planned_risk": 0.5},
        {"symbol": "BBB", "pnl": -0.6, "planned_risk": 0.5},
    ]

    return summaries, trades


def make_evaluation(candidate_id, metrics):

    return {
        **metrics,
        "candidate_id": candidate_id,
        "data_role": "TRAINING",
        "qualified": True,
        "trading_authorized": False,
    }


def test_candidate_count(selector):

    assert len(selector.generate_candidates()) == 12


def test_unique_candidate_ids(selector):

    candidates = selector.generate_candidates()

    ids = [candidate["candidate_id"] for candidate in candidates]

    assert len(ids) == len(set(ids))


def test_candidate_generation_deterministic(selector):

    assert selector.generate_candidates() == selector.generate_candidates()


def test_first_candidate(selector):

    candidate = selector.generate_candidates()[0]

    assert candidate["candidate_id"] == "C001"
    assert candidate["minimum_score"] == 55
    assert candidate["stop_atr_multiplier"] == 1.5
    assert candidate["risk_reward_ratio"] == 2.0


def test_last_candidate(selector):

    candidate = selector.generate_candidates()[-1]

    assert candidate["candidate_id"] == "C012"
    assert candidate["minimum_score"] == 75
    assert candidate["stop_atr_multiplier"] == 2.0
    assert candidate["risk_reward_ratio"] == 3.0


@pytest.mark.parametrize(
    "field,value",
    [
        ("live_trading_enabled", True),
        ("historical_simulation_only", False),
        ("automatic_strategy_activation", True),
        ("require_forward_validation", False),
        ("allow_test_data_for_selection", True),
    ],
)
def test_unsafe_configuration_rejected(config, field, value):

    config["safety"][field] = value

    with pytest.raises(RuntimeError):
        ParameterSelector(config)


def test_forced_selection_rejected(config):

    config["selection"]["allow_no_selection"] = False

    with pytest.raises(RuntimeError):
        ParameterSelector(config)


def test_duplicate_symbols_rejected(config):

    config["symbols"] = ["AAA", "AAA"]

    with pytest.raises(ValueError):
        ParameterSelector(config)


def test_empty_symbols_rejected(config):

    config["symbols"] = []

    with pytest.raises(ValueError):
        ParameterSelector(config)


@pytest.mark.parametrize(
    "values", [[], [55, 55], [-10], [0], [101], [float("nan")], [float("inf")]]
)
def test_invalid_parameter_candidates(config, values):

    config["parameter_candidates"]["minimum_score"] = values

    with pytest.raises(ValueError):
        ParameterSelector(config)


def test_valid_candidate_qualification(selector, valid_metrics):

    qualified, reason = selector.qualify_candidate(valid_metrics)

    assert qualified is True
    assert reason == "Passed training requirements"


@pytest.mark.parametrize(
    "field,value",
    [
        ("total_trades", 29),
        ("profit_factor", 1.24),
        ("expectancy", 0.0),
        ("average_return", 0.0),
        ("maximum_drawdown", 5.01),
        ("risk_breach_percent", 10.01),
    ],
)
def test_candidate_qualification_failure(selector, valid_metrics, field, value):

    metrics = deepcopy(valid_metrics)

    metrics[field] = value

    qualified, reason = selector.qualify_candidate(metrics)

    assert qualified is False
    assert reason


@pytest.mark.parametrize(
    "field,value",
    [
        ("expectancy", float("nan")),
        ("average_return", float("inf")),
        ("maximum_drawdown", float("nan")),
        ("risk_breach_percent", float("inf")),
        ("profit_factor", float("nan")),
    ],
)
def test_invalid_metrics_rejected(selector, valid_metrics, field, value):

    metrics = deepcopy(valid_metrics)

    metrics[field] = value

    qualified, reason = selector.qualify_candidate(metrics)

    assert qualified is False
    assert reason


def test_qualification_boundaries(selector, valid_metrics):

    valid_metrics["total_trades"] = 30
    valid_metrics["profit_factor"] = 1.25
    valid_metrics["maximum_drawdown"] = 5.0
    valid_metrics["risk_breach_percent"] = 10.0

    qualified, _ = selector.qualify_candidate(valid_metrics)

    assert qualified is True


def test_training_metric_calculation(selector, training_data):

    summaries, trades = training_data

    metrics = selector.calculate_training_metrics(summaries, trades)

    assert metrics["symbols_tested"] == 2
    assert metrics["total_trades"] == 4
    assert metrics["gross_profit"] == pytest.approx(1.9)
    assert metrics["gross_loss"] == pytest.approx(1.0)
    assert metrics["net_profit"] == pytest.approx(0.9)
    assert metrics["profit_factor"] == pytest.approx(1.9)
    assert metrics["expectancy"] == pytest.approx(0.225)
    assert metrics["win_rate"] == pytest.approx(50.0)
    assert metrics["average_return"] == pytest.approx(0.45)
    assert metrics["maximum_drawdown"] == pytest.approx(2.0)
    assert metrics["profitable_symbols"] == 2
    assert metrics["risk_breaches"] == 1
    assert metrics["risk_breach_percent"] == pytest.approx(25.0)


def test_empty_training_summary_rejected(selector):

    with pytest.raises(ValueError):
        selector.calculate_training_metrics([], [])


def test_incomplete_symbol_coverage(selector, training_data):

    summaries, trades = training_data

    with pytest.raises(ValueError):
        selector.calculate_training_metrics(summaries[:1], trades[:2])


def test_trade_count_mismatch(selector, training_data):

    summaries, trades = training_data

    with pytest.raises(ValueError):
        selector.calculate_training_metrics(summaries, trades[:-1])


def test_pnl_reconciliation_failure(selector, training_data):

    summaries, trades = training_data

    summaries[0]["net_profit"] = 999.0

    with pytest.raises(ValueError):
        selector.calculate_training_metrics(summaries, trades)


def test_invalid_planned_risk(selector, training_data):

    summaries, trades = training_data

    trades[0]["planned_risk"] = 0.0

    with pytest.raises(ValueError):
        selector.calculate_training_metrics(summaries, trades)


def test_no_trade_metrics(selector):

    summaries = [
        {
            "symbol": symbol,
            "total_return": 0.0,
            "total_trades": 0,
            "max_drawdown": 0.0,
            "net_profit": 0.0,
        }
        for symbol in ["AAA", "BBB"]
    ]

    metrics = selector.calculate_training_metrics(summaries, [])

    assert metrics["total_trades"] == 0
    assert metrics["profit_factor"] == 0.0
    assert metrics["expectancy"] == 0.0
    assert metrics["risk_breaches"] == 0


def test_testing_data_evaluation_blocked(selector, training_data):

    summaries, trades = training_data

    candidate = selector.generate_candidates()[0]

    with pytest.raises(RuntimeError):
        selector.evaluate_candidate(candidate, summaries, trades, data_role="TESTING")


def test_testing_data_selection_blocked(selector, valid_metrics):

    evaluation = make_evaluation("C001", valid_metrics)

    evaluation["data_role"] = "TESTING"

    with pytest.raises(RuntimeError):
        selector.select([evaluation])


def test_empty_selection(selector):

    result = selector.select([])

    assert result["status"] == "NO_SELECTION"
    assert result["selected_candidate"] is None
    assert result["trading_authorized"] is False


def test_no_qualified_candidates(selector, valid_metrics):

    evaluation = make_evaluation("C001", valid_metrics)

    evaluation["profit_factor"] = 0.8

    result = selector.select([evaluation])

    assert result["status"] == "NO_SELECTION"
    assert result["selected_candidate"] is None


def test_highest_return_candidate_selected(selector, valid_metrics):

    first = make_evaluation("C001", valid_metrics)

    second = make_evaluation("C002", valid_metrics)

    second["average_return"] = 3.0

    result = selector.select([first, second])

    assert result["selected_candidate"]["candidate_id"] == "C002"


def test_deterministic_tie_break(selector, valid_metrics):

    first = make_evaluation("C001", valid_metrics)

    second = make_evaluation("C002", valid_metrics)

    result = selector.select([second, first])

    assert result["selected_candidate"]["candidate_id"] == "C001"


def test_rejected_candidate_not_selected(selector, valid_metrics):

    evaluation = make_evaluation("C001", valid_metrics)

    evaluation["qualified"] = False

    result = selector.select([evaluation])

    assert result["status"] == "NO_SELECTION"


def test_selection_does_not_authorize_trading(selector, valid_metrics):

    evaluation = make_evaluation("C001", valid_metrics)

    result = selector.select([evaluation])

    assert result["status"] == "SELECTED_FOR_HISTORICAL_TEST"

    assert result["trading_authorized"] is False
    assert result["forward_validation_required"] is True
