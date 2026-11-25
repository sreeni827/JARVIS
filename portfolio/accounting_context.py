from decimal import (
    Context,
    DivisionByZero,
    InvalidOperation,
    Overflow,
    ROUND_HALF_EVEN,
    localcontext,
)

from functools import wraps


WORKING_PRECISION = 64

MINIMUM_EXPONENT = -999999

MAXIMUM_EXPONENT = 999999


def create_accounting_context():
    """
    Create an independent Decimal context for JARVIS.

    Caller precision, rounding, traps and flags
    are not inherited.
    """

    return Context(
        prec=WORKING_PRECISION,
        rounding=ROUND_HALF_EVEN,
        Emin=MINIMUM_EXPONENT,
        Emax=MAXIMUM_EXPONENT,
        capitals=1,
        clamp=0,
        flags=[],
        traps=[
            InvalidOperation,
            DivisionByZero,
            Overflow,
        ],
    )


def accounting_context():
    """
    Return an isolated accounting context manager.
    """

    return localcontext(
        create_accounting_context()
    )


def isolated_accounting(function):
    """
    Execute a function inside the JARVIS
    accounting context.

    The caller context is restored after
    successful execution or an exception.
    """

    @wraps(function)
    def wrapper(*args, **kwargs):

        with accounting_context():

            return function(*args, **kwargs)

    return wrapper


def display_summary():

    context = create_accounting_context()

    print("=" * 65)
    print("JARVIS ACCOUNTING CONTEXT")
    print("VERSION 0.8.6.7.2C.4D.1")
    print("=" * 65)

    print("Working precision:", context.prec)
    print("Rounding mode:", context.rounding)
    print("Minimum exponent:", context.Emin)
    print("Maximum exponent:", context.Emax)

    print("Caller precision inherited: NO")
    print("Caller rounding inherited: NO")
    print("Caller traps inherited: NO")
    print("Caller flags inherited: NO")

    print("Global context modification: NONE")
    print("ACCOUNTING CONTEXT: READY")


if __name__ == "__main__":

    display_summary()