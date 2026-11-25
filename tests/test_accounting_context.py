from decimal import (
    Decimal,
    getcontext,
    localcontext,
    ROUND_HALF_EVEN,
    ROUND_HALF_UP,
    ROUND_DOWN,
    ROUND_UP,
    ROUND_FLOOR,
    ROUND_CEILING,
    Inexact,
    Rounded,
    DivisionByZero,
    InvalidOperation,
    Overflow,
)

import pytest

from portfolio.accounting_context import (
    WORKING_PRECISION,
    create_accounting_context,
    accounting_context,
    isolated_accounting,
)


D = Decimal

PRECISIONS = [
    10,
    16,
    20,
    28,
    32,
    50,
    64,
]

ROUNDING_MODES = [
    ROUND_HALF_EVEN,
    ROUND_HALF_UP,
    ROUND_DOWN,
    ROUND_UP,
    ROUND_FLOOR,
    ROUND_CEILING,
]


def context_signature():

    context = getcontext()

    return (
        context.prec,
        context.rounding,
        context.Emin,
        context.Emax,
        context.capitals,
        context.clamp,
        dict(context.traps),
        dict(context.flags),
    )


@isolated_accounting
def calculate_reference():

    return D("1") / D("7")


def test_working_precision():

    assert WORKING_PRECISION == 64


def test_context_precision():

    context = create_accounting_context()

    assert context.prec == 64


def test_context_rounding():

    context = create_accounting_context()

    assert context.rounding == ROUND_HALF_EVEN


def test_context_exponent_limits():

    context = create_accounting_context()

    assert context.Emin == -999999

    assert context.Emax == 999999


def test_context_default_traps():

    context = create_accounting_context()

    assert context.traps[InvalidOperation]

    assert context.traps[DivisionByZero]

    assert context.traps[Overflow]

    assert not context.traps[Inexact]

    assert not context.traps[Rounded]


def test_context_flags_initially_clear():

    context = create_accounting_context()

    assert not any(context.flags.values())


def test_context_instances_independent():

    first = create_accounting_context()

    second = create_accounting_context()

    first.prec = 10

    first.rounding = ROUND_DOWN

    assert second.prec == 64

    assert second.rounding == ROUND_HALF_EVEN


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("rounding", ROUNDING_MODES)
def test_context_matrix(precision, rounding):

    with accounting_context():

        expected = D("1") / D("7")

    with localcontext() as caller:

        caller.prec = precision

        caller.rounding = rounding

        before = context_signature()

        result = calculate_reference()

        after = context_signature()

        assert result == expected

        assert before == after


@pytest.mark.parametrize("precision", PRECISIONS)
def test_context_manager_precision(precision):

    with localcontext() as caller:

        caller.prec = precision

        before = context_signature()

        with accounting_context():

            assert getcontext().prec == 64

            assert (
                getcontext().rounding
                == ROUND_HALF_EVEN
            )

        assert context_signature() == before


@pytest.mark.parametrize(
    "signal",
    [Inexact, Rounded],
)
def test_caller_traps_not_inherited(signal):

    with localcontext() as caller:

        caller.traps[signal] = True

        before = context_signature()

        result = calculate_reference()

        assert result.is_finite()

        assert context_signature() == before


@pytest.mark.parametrize(
    "signal",
    [Inexact, Rounded],
)
def test_caller_flags_preserved(signal):

    with localcontext() as caller:

        caller.flags[signal] = True

        before = context_signature()

        calculate_reference()

        assert context_signature() == before


def test_caller_exponent_limits_not_inherited():

    with localcontext() as caller:

        caller.Emin = -2

        caller.Emax = 2

        before = context_signature()

        with accounting_context():

            result = D("100000") * D("100000")

            assert result == D("10000000000")

        assert context_signature() == before


def test_context_restored_after_exception():

    with localcontext() as caller:

        caller.prec = 10

        caller.rounding = ROUND_DOWN

        before = context_signature()

        with pytest.raises(RuntimeError):

            with accounting_context():

                raise RuntimeError("Test exception")

        assert context_signature() == before


def test_decorator_restored_after_exception():

    @isolated_accounting
    def failing_function():

        raise ValueError("Test exception")

    before = context_signature()

    with pytest.raises(ValueError):

        failing_function()

    assert context_signature() == before


def test_nested_accounting_contexts():

    before = context_signature()

    with accounting_context():

        outer = context_signature()

        with accounting_context():

            assert getcontext().prec == 64

            result = D("1") / D("3")

            assert result.is_finite()

        assert context_signature() == outer

    assert context_signature() == before


def test_decorator_preserves_metadata():

    @isolated_accounting
    def example_function():

        """Example documentation."""

        return D("100")

    assert example_function.__name__ == (
        "example_function"
    )

    assert example_function.__doc__ == (
        "Example documentation."
    )

    assert example_function() == D("100")


def test_decorator_accepts_arguments():

    @isolated_accounting
    def calculate(a, b, multiplier=D("1")):

        return (a / b) * multiplier

    with accounting_context():

        expected = (
            D("100") / D("3")
        ) * D("2")

    result = calculate(
        D("100"),
        D("3"),
        multiplier=D("2"),
    )

    assert result == expected


def test_division_by_zero_trap():

    before = context_signature()

    with pytest.raises(DivisionByZero):

        with accounting_context():

            D("1") / D("0")

    assert context_signature() == before


def test_invalid_operation_trap():

    before = context_signature()

    with pytest.raises(InvalidOperation):

        with accounting_context():

            D("0") / D("0")

    assert context_signature() == before


def test_global_context_unchanged():

    before = context_signature()

    for _ in range(100):

        calculate_reference()

    assert context_signature() == before