from decimal import (
    Decimal as D,
    ROUND_DOWN,
    ROUND_HALF_EVEN,
    getcontext,
    localcontext,
)

import pytest

from portfolio.decimal_utils import (
    ZERO,
    PRICE_STEP,
    QUANTITY_STEP,
    MONEY_STEP,
    WORKING_PRECISION,
    as_decimal,
    quantize_value,
    normalize_price,
    normalize_quantity,
    normalize_money,
    normalize_rate,
    calculate_buy_price,
    calculate_sell_price,
    calculate_notional,
    calculate_fee,
    calculate_entry_cost,
    calculate_exit_proceeds,
    calculate_planned_risk,
    calculate_pnl,
)


SLIPPAGE = D("0.001")
FEE_RATE = D("0.0005")


# ============================================================
# ACCOUNTING CONSTANTS
# ============================================================


def test_price_precision():

    assert PRICE_STEP == D("0.00000001")


def test_quantity_precision():

    assert QUANTITY_STEP == D("0.000001")


def test_money_precision():

    assert MONEY_STEP == D("0.000000000001")


def test_working_precision():

    assert WORKING_PRECISION == 64


# ============================================================
# DECIMAL CONVERSION
# ============================================================


@pytest.mark.parametrize(
    "value, expected",
    [
        (100, D("100")),
        (100.25, D("100.25")),
        ("100.25", D("100.25")),
        (D("100.25"), D("100.25")),
        (0, ZERO),
        ("-10.25", D("-10.25")),
        ("1E-12", D("1E-12")),
        (" 100.25 ", D("100.25")),
    ],
)
def test_valid_decimal_conversion(value, expected):

    assert as_decimal(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        None,
        [],
        {},
        (),
        "",
        "invalid",
        "NaN",
        "sNaN",
        "Infinity",
        "-Infinity",
        float("nan"),
        float("inf"),
        float("-inf"),
        D("NaN"),
        D("Infinity"),
    ],
)
def test_invalid_decimal_conversion(value):

    with pytest.raises(ValueError):

        as_decimal(value)


# ============================================================
# QUANTIZATION
# ============================================================


@pytest.mark.parametrize(
    "value, expected",
    [
        ("1.123456789", "1.12345679"),
        ("1.123456784", "1.12345678"),
        ("1.123456785", "1.12345678"),
        ("1.123456795", "1.12345680"),
        ("100", "100.00000000"),
    ],
)
def test_half_even_quantization(value, expected):

    assert quantize_value(
        value,
        PRICE_STEP,
        ROUND_HALF_EVEN,
    ) == D(expected)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("0.123456789", "0.123456"),
        ("1.999999999", "1.999999"),
        ("0.000000999", "0.000000"),
        ("10", "10.000000"),
    ],
)
def test_round_down_quantization(value, expected):

    assert quantize_value(
        value,
        QUANTITY_STEP,
        ROUND_DOWN,
    ) == D(expected)


def test_unsupported_accounting_precision():

    with pytest.raises(ValueError):

        normalize_money("1E100")


# ============================================================
# PRICE NORMALIZATION
# ============================================================


@pytest.mark.parametrize(
    "value, expected",
    [
        ("100", "100.00000000"),
        ("100.123456789", "100.12345679"),
        ("25.123456789123", "25.12345679"),
        ("0.00000001", "0.00000001"),
        ("0.000000006", "0.00000001"),
        ("999.12345678912345", "999.12345679"),
    ],
)
def test_valid_price_normalization(value, expected):

    result = normalize_price(value)

    assert result == D(expected)

    assert result.as_tuple().exponent == -8


@pytest.mark.parametrize(
    "value",
    [
        "0",
        "-1",
        "-100",
        "0.000000001",
        "0.000000005",
        "NaN",
        "Infinity",
        True,
        None,
    ],
)
def test_invalid_price_normalization(value):

    with pytest.raises(ValueError):

        normalize_price(value)


# ============================================================
# QUANTITY NORMALIZATION
# ============================================================


@pytest.mark.parametrize(
    "value, expected",
    [
        ("1", "1.000000"),
        ("0.123456789", "0.123456"),
        ("0.999999999", "0.999999"),
        ("0.000001", "0.000001"),
        ("0.000000999", "0.000000"),
        ("0", "0.000000"),
        ("10.123456", "10.123456"),
    ],
)
def test_valid_quantity_normalization(value, expected):

    result = normalize_quantity(value)

    assert result == D(expected)

    assert result.as_tuple().exponent == -6


@pytest.mark.parametrize(
    "value",
    [
        "-1",
        "-0.000001",
        "-0.000000001",
        "NaN",
        "Infinity",
        True,
        None,
    ],
)
def test_invalid_quantity_normalization(value):

    with pytest.raises(ValueError):

        normalize_quantity(value)


# ============================================================
# MONEY NORMALIZATION
# ============================================================


@pytest.mark.parametrize(
    "value, expected",
    [
        ("100", "100.000000000000"),
        ("0", "0.000000000000"),
        ("-10.25", "-10.250000000000"),
        ("1.1234567891234", "1.123456789123"),
        ("1.1234567891236", "1.123456789124"),
        ("1.1234567891225", "1.123456789122"),
        ("1.1234567891235", "1.123456789124"),
        ("0.0000000000001", "0.000000000000"),
    ],
)
def test_money_normalization(value, expected):

    result = normalize_money(value)

    assert result == D(expected)

    assert result.as_tuple().exponent == -12


@pytest.mark.parametrize(
    "value",
    [
        "NaN",
        "Infinity",
        "-Infinity",
        True,
        None,
    ],
)
def test_invalid_money_normalization(value):

    with pytest.raises(ValueError):

        normalize_money(value)


# ============================================================
# RATE VALIDATION
# ============================================================


@pytest.mark.parametrize(
    "value",
    [
        "0",
        "0.0005",
        "0.001",
        "0.01",
        "0.50",
        "0.999999",
    ],
)
def test_valid_rates(value):

    assert normalize_rate(value) == D(value)


@pytest.mark.parametrize(
    "value",
    [
        "-0.001",
        "-1",
        "1",
        "1.01",
        "100",
        "NaN",
        "Infinity",
        True,
        None,
    ],
)
def test_invalid_rates(value):

    with pytest.raises(ValueError):

        normalize_rate(value)


# ============================================================
# EXECUTION PRICES
# ============================================================


@pytest.mark.parametrize(
    "market_price, rate, expected",
    [
        ("100", "0.001", "100.10000000"),
        ("100", "0", "100.00000000"),
        ("50", "0.01", "50.50000000"),
        ("25", "0.005", "25.12500000"),
    ],
)
def test_buy_price(market_price, rate, expected):

    assert calculate_buy_price(
        market_price,
        rate,
    ) == D(expected)


@pytest.mark.parametrize(
    "market_price, rate, expected",
    [
        ("100", "0.001", "99.90000000"),
        ("100", "0", "100.00000000"),
        ("50", "0.01", "49.50000000"),
        ("25", "0.005", "24.87500000"),
    ],
)
def test_sell_price(market_price, rate, expected):

    assert calculate_sell_price(
        market_price,
        rate,
    ) == D(expected)


@pytest.mark.parametrize(
    "function",
    [
        calculate_buy_price,
        calculate_sell_price,
    ],
)
@pytest.mark.parametrize(
    "price, rate",
    [
        ("0", "0.001"),
        ("-100", "0.001"),
        ("100", "-0.001"),
        ("100", "1"),
        ("NaN", "0.001"),
    ],
)
def test_invalid_execution_inputs(function, price, rate):

    with pytest.raises(ValueError):

        function(price, rate)


def test_execution_price_direction():

    market_price = D("100")

    buy_price = calculate_buy_price(
        market_price,
        SLIPPAGE,
    )

    sell_price = calculate_sell_price(
        market_price,
        SLIPPAGE,
    )

    assert buy_price > market_price

    assert sell_price < market_price


# ============================================================
# NOTIONAL AND FEES
# ============================================================


@pytest.mark.parametrize(
    "quantity, price, expected",
    [
        ("0.15", "100", "15"),
        ("1", "25", "25"),
        ("0.000001", "100", "0.0001"),
        ("0", "100", "0"),
        ("0.123456789", "100", "12.3456"),
    ],
)
def test_notional(quantity, price, expected):

    assert calculate_notional(
        quantity,
        price,
    ) == D(expected)


@pytest.mark.parametrize(
    "notional, rate, expected",
    [
        ("100", "0.0005", "0.05"),
        ("15", "0.0005", "0.0075"),
        ("100", "0", "0"),
        ("0", "0.0005", "0"),
        ("25", "0.01", "0.25"),
    ],
)
def test_fee_calculation(notional, rate, expected):

    assert calculate_fee(
        notional,
        rate,
    ) == D(expected)


@pytest.mark.parametrize(
    "notional, rate",
    [
        ("-100", "0.001"),
        ("100", "-0.001"),
        ("100", "1"),
        ("NaN", "0.001"),
    ],
)
def test_invalid_fee_inputs(notional, rate):

    with pytest.raises(ValueError):

        calculate_fee(notional, rate)


# ============================================================
# ENTRY AND EXIT ACCOUNTING
# ============================================================


def test_entry_cost():

    result = calculate_entry_cost(
        "0.15",
        "100",
        FEE_RATE,
    )

    assert result == D("15.0075")


def test_exit_proceeds():

    result = calculate_exit_proceeds(
        "0.15",
        "100",
        FEE_RATE,
    )

    assert result == D("14.9925")


def test_entry_cost_without_fees():

    assert calculate_entry_cost(
        "0.15",
        "100",
        ZERO,
    ) == D("15")


def test_exit_proceeds_without_fees():

    assert calculate_exit_proceeds(
        "0.15",
        "100",
        ZERO,
    ) == D("15")


def test_entry_cost_exceeds_notional():

    notional = calculate_notional(
        "0.15",
        "100",
    )

    entry_cost = calculate_entry_cost(
        "0.15",
        "100",
        FEE_RATE,
    )

    assert entry_cost > notional


def test_exit_proceeds_below_notional():

    notional = calculate_notional(
        "0.15",
        "100",
    )

    proceeds = calculate_exit_proceeds(
        "0.15",
        "100",
        FEE_RATE,
    )

    assert proceeds < notional


# ============================================================
# PROFIT AND LOSS
# ============================================================


@pytest.mark.parametrize(
    "entry_cost, exit_proceeds, expected",
    [
        ("15", "16", "1"),
        ("15", "14", "-1"),
        ("15", "15", "0"),
        ("15.0075", "14.9925", "-0.015"),
        ("0", "0", "0"),
    ],
)
def test_pnl(entry_cost, exit_proceeds, expected):

    assert calculate_pnl(
        entry_cost,
        exit_proceeds,
    ) == D(expected)


@pytest.mark.parametrize(
    "entry_cost, exit_proceeds",
    [
        ("-1", "10"),
        ("10", "-1"),
        ("NaN", "10"),
        ("10", "Infinity"),
    ],
)
def test_invalid_pnl_inputs(entry_cost, exit_proceeds):

    with pytest.raises(ValueError):

        calculate_pnl(entry_cost, exit_proceeds)


# ============================================================
# PLANNED RISK
# ============================================================


def test_planned_risk_baseline():

    risk = calculate_planned_risk(
        quantity="0.15",
        entry_price="100.10",
        stop_price="97.10",
        slippage_rate=SLIPPAGE,
        fee_rate=FEE_RATE,
    )

    assert risk == D("0.479347717500")


@pytest.mark.parametrize(
    "stop_price",
    [
        "100",
        "101",
        "150",
        "0",
        "-1",
    ],
)
def test_invalid_stop_prices(stop_price):

    with pytest.raises(ValueError):

        calculate_planned_risk(
            quantity="0.15",
            entry_price="100",
            stop_price=stop_price,
            slippage_rate=SLIPPAGE,
            fee_rate=FEE_RATE,
        )


def test_risk_without_execution_costs():

    risk = calculate_planned_risk(
        quantity="0.15",
        entry_price="100",
        stop_price="95",
        slippage_rate=ZERO,
        fee_rate=ZERO,
    )

    assert risk == D("0.75")


# ============================================================
# HIGH-PRECISION ACCOUNTING MATRIX
# ============================================================


@pytest.mark.parametrize(
    "price",
    [
        "5.123456789",
        "10.987654321",
        "25.123456789123",
        "50.987654321987",
        "100",
        "100.12345678912345",
        "150.98765432198765",
        "250.12345678912345",
        "500.98765432198765",
        "999.12345678912345",
    ],
)
@pytest.mark.parametrize(
    "atr_percent",
    [
        "0.5",
        "1",
        "2",
        "3",
        "5",
    ],
)
def test_high_precision_stop_reconciliation(
    price,
    atr_percent,
):

    market_price = normalize_price(price)

    quantity = normalize_quantity(
        D("15") / market_price
    )

    entry_price = calculate_buy_price(
        market_price,
        SLIPPAGE,
    )

    atr = (
        market_price
        * D(atr_percent)
        / D("100")
    )

    stop_price = normalize_price(
        entry_price - atr * D("1.5")
    )

    planned_risk = calculate_planned_risk(
        quantity,
        entry_price,
        stop_price,
        SLIPPAGE,
        FEE_RATE,
    )

    entry_cost = calculate_entry_cost(
        quantity,
        entry_price,
        FEE_RATE,
    )

    exit_price = calculate_sell_price(
        stop_price,
        SLIPPAGE,
    )

    exit_proceeds = calculate_exit_proceeds(
        quantity,
        exit_price,
        FEE_RATE,
    )

    pnl = calculate_pnl(
        entry_cost,
        exit_proceeds,
    )

    actual_loss = max(ZERO, -pnl)

    assert actual_loss == planned_risk

    cash = normalize_money(
        D("100") - entry_cost
    )

    cash = normalize_money(
        cash + exit_proceeds
    )

    assert cash == normalize_money(
        D("100") + pnl
    )


# ============================================================
# RISK BREACH CLASSIFICATION
# ============================================================


@pytest.mark.parametrize(
    "market_exit, expected_breach",
    [
        ("97.10", False),
        ("97.09", True),
        ("96", True),
        ("90", True),
        ("50", True),
        ("98", False),
        ("100", False),
        ("106", False),
    ],
)
def test_risk_breach_classification(
    market_exit,
    expected_breach,
):

    quantity = D("0.15")

    entry_price = calculate_buy_price(
        "100",
        SLIPPAGE,
    )

    stop_price = D("97.10")

    planned_risk = calculate_planned_risk(
        quantity,
        entry_price,
        stop_price,
        SLIPPAGE,
        FEE_RATE,
    )

    entry_cost = calculate_entry_cost(
        quantity,
        entry_price,
        FEE_RATE,
    )

    exit_price = calculate_sell_price(
        market_exit,
        SLIPPAGE,
    )

    proceeds = calculate_exit_proceeds(
        quantity,
        exit_price,
        FEE_RATE,
    )

    pnl = calculate_pnl(
        entry_cost,
        proceeds,
    )

    actual_loss = max(ZERO, -pnl)

    assert (
        actual_loss > planned_risk
    ) == expected_breach


# ============================================================
# REPEATED LEDGER TRANSACTIONS
# ============================================================


@pytest.mark.parametrize(
    "transaction_count",
    [
        10,
        100,
        500,
        1000,
    ],
)
def test_repeated_ledger_reconciliation(
    transaction_count,
):

    initial_capital = normalize_money("100")

    cash = initial_capital

    realized_pnl = normalize_money(ZERO)

    trade_pnls = []

    for i in range(transaction_count):

        market_price = normalize_price(
            D("100.12345678912345")
            + D(i) / D("7")
        )

        quantity = normalize_quantity(
            D("15") / market_price
        )

        entry_price = calculate_buy_price(
            market_price,
            SLIPPAGE,
        )

        entry_cost = calculate_entry_cost(
            quantity,
            entry_price,
            FEE_RATE,
        )

        cash = normalize_money(
            cash - entry_cost
        )

        market_value = calculate_notional(
            quantity,
            market_price,
        )

        unrealized_pnl = normalize_money(
            market_value - entry_cost
        )

        equity = normalize_money(
            cash + market_value
        )

        assert equity == (
            initial_capital
            + realized_pnl
            + unrealized_pnl
        )

        market_exit = normalize_price(
            market_price
            * (
                D("1.01")
                if i % 2 == 0
                else D("0.995")
            )
        )

        exit_price = calculate_sell_price(
            market_exit,
            SLIPPAGE,
        )

        proceeds = calculate_exit_proceeds(
            quantity,
            exit_price,
            FEE_RATE,
        )

        pnl = calculate_pnl(
            entry_cost,
            proceeds,
        )

        cash = normalize_money(
            cash + proceeds
        )

        realized_pnl = normalize_money(
            realized_pnl + pnl
        )

        trade_pnls.append(pnl)

        assert cash == (
            initial_capital + realized_pnl
        )

        assert cash >= ZERO

    assert realized_pnl == sum(
        trade_pnls,
        ZERO,
    )


# ============================================================
# DECIMAL CONTEXT ISOLATION
# ============================================================


def test_global_context_unchanged():

    original_precision = getcontext().prec

    original_rounding = getcontext().rounding

    calculate_planned_risk(
        quantity="0.15",
        entry_price="100.10",
        stop_price="97.10",
        slippage_rate=SLIPPAGE,
        fee_rate=FEE_RATE,
    )

    assert getcontext().prec == original_precision

    assert getcontext().rounding == original_rounding


@pytest.mark.parametrize(
    "precision",
    [
        10,
        16,
        28,
        50,
    ],
)
def test_local_precision_isolation(precision):

    with localcontext() as context:

        context.prec = precision

        result = calculate_planned_risk(
            quantity="0.15",
            entry_price="100.10",
            stop_price="97.10",
            slippage_rate=SLIPPAGE,
            fee_rate=FEE_RATE,
        )

        assert result == D("0.479347717500")

        assert context.prec == precision


def test_local_rounding_isolation():

    with localcontext() as context:

        context.rounding = ROUND_DOWN

        result = normalize_price(
            "1.123456789"
        )

        assert result == D("1.12345679")

        assert context.rounding == ROUND_DOWN