from decimal import (
    Decimal,
    InvalidOperation,
    ROUND_DOWN,
    ROUND_HALF_EVEN,
    localcontext,
)

from portfolio.accounting_context import (
    accounting_context,
    isolated_accounting,
)


D = Decimal

ZERO = D("0")
ONE = D("1")
HUNDRED = D("100")

PRICE_STEP = D("0.00000001")
QUANTITY_STEP = D("0.000001")
MONEY_STEP = D("0.000000000001")

WORKING_PRECISION = 64


@isolated_accounting
def as_decimal(value):

    if isinstance(value, bool):
        raise ValueError("Boolean values are not valid decimals.")

    if not isinstance(value, (Decimal, int, float, str)):
        raise ValueError("Unsupported decimal value.")

    try:

        result = D(str(value))

    except (InvalidOperation, ValueError) as error:

        raise ValueError("Invalid decimal value.") from error

    if not result.is_finite():

        raise ValueError("Decimal value must be finite.")

    return result


@isolated_accounting
def quantize_value(value, step, rounding=ROUND_HALF_EVEN):

    value = as_decimal(value)

    try:

        with accounting_context():

            return value.quantize(
                step,
                rounding=rounding,
            )

    except InvalidOperation as error:

        raise ValueError(
            "Value exceeds supported accounting precision."
        ) from error


@isolated_accounting
def normalize_price(value):

    value = as_decimal(value)

    if value <= ZERO:

        raise ValueError("Price must be positive.")

    result = quantize_value(
        value,
        PRICE_STEP,
    )

    if result <= ZERO:

        raise ValueError(
            "Price is below supported precision."
        )

    return result


@isolated_accounting
def normalize_quantity(value):

    value = as_decimal(value)

    if value < ZERO:

        raise ValueError("Quantity cannot be negative.")

    return quantize_value(
        value,
        QUANTITY_STEP,
        ROUND_DOWN,
    )


@isolated_accounting
def normalize_money(value):

    return quantize_value(
        value,
        MONEY_STEP,
    )


@isolated_accounting
def normalize_rate(value):

    value = as_decimal(value)

    if not ZERO <= value < ONE:

        raise ValueError(
            "Rate must be between zero and one."
        )

    return value


@isolated_accounting
def calculate_buy_price(market_price, slippage_rate):

    market_price = normalize_price(market_price)
    slippage_rate = normalize_rate(slippage_rate)

    with accounting_context():

        return normalize_price(
            market_price * (ONE + slippage_rate)
        )


@isolated_accounting
def calculate_sell_price(market_price, slippage_rate):

    market_price = normalize_price(market_price)
    slippage_rate = normalize_rate(slippage_rate)

    with accounting_context():

        return normalize_price(
            market_price * (ONE - slippage_rate)
        )


@isolated_accounting
def calculate_notional(quantity, price):

    quantity = normalize_quantity(quantity)
    price = normalize_price(price)

    with accounting_context():

        return normalize_money(
            quantity * price
        )


@isolated_accounting
def calculate_fee(notional, fee_rate):

    notional = normalize_money(notional)
    fee_rate = normalize_rate(fee_rate)

    if notional < ZERO:

        raise ValueError("Notional cannot be negative.")

    with accounting_context():

        return normalize_money(
            notional * fee_rate
        )


@isolated_accounting
def calculate_entry_cost(quantity, price, fee_rate):

    notional = calculate_notional(
        quantity,
        price,
    )

    fee = calculate_fee(
        notional,
        fee_rate,
    )

    with accounting_context():

        return normalize_money(
            notional + fee
        )


@isolated_accounting
def calculate_exit_proceeds(quantity, price, fee_rate):

    notional = calculate_notional(
        quantity,
        price,
    )

    fee = calculate_fee(
        notional,
        fee_rate,
    )

    with accounting_context():

        return normalize_money(
            notional - fee
        )


@isolated_accounting
def calculate_planned_risk(
    quantity,
    entry_price,
    stop_price,
    slippage_rate,
    fee_rate,
):

    entry_price = normalize_price(entry_price)
    stop_price = normalize_price(stop_price)

    if stop_price >= entry_price:

        raise ValueError(
            "Stop price must be below entry price."
        )

    entry_cost = calculate_entry_cost(
        quantity,
        entry_price,
        fee_rate,
    )

    stop_fill = calculate_sell_price(
        stop_price,
        slippage_rate,
    )

    stop_proceeds = calculate_exit_proceeds(
        quantity,
        stop_fill,
        fee_rate,
    )

    with accounting_context():

        return normalize_money(
            entry_cost - stop_proceeds
        )


@isolated_accounting
def calculate_pnl(entry_cost, exit_proceeds):

    entry_cost = normalize_money(entry_cost)
    exit_proceeds = normalize_money(exit_proceeds)

    if entry_cost < ZERO or exit_proceeds < ZERO:

        raise ValueError(
            "Entry cost and exit proceeds cannot be negative."
        )

    with accounting_context():

        return normalize_money(
            exit_proceeds - entry_cost
        )


def display_summary():

    print("=" * 70)
    print("JARVIS DECIMAL ACCOUNTING ENGINE")
    print("VERSION 0.8.6.7.2C.1")
    print("=" * 70)

    print("Price precision:", PRICE_STEP)
    print("Quantity precision:", QUANTITY_STEP)
    print("Monetary precision:", MONEY_STEP)
    print("Working precision:", WORKING_PRECISION)

    print("Price rounding: ROUND_HALF_EVEN")
    print("Quantity rounding: ROUND_DOWN")
    print("Money rounding: ROUND_HALF_EVEN")
    print("Global Decimal context: UNCHANGED")
    print("Historical simulation only")
    print("Broker orders: DISABLED")
    print("Live trading: DISABLED")


if __name__ == "__main__":

    display_summary()