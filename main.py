import os
from decimal import Decimal
from pathlib import Path

import requests
from alpaca.trading.client import TradingClient
from dotenv import load_dotenv

# Load configuration
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=True)

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK_URL")

STARTING_CAPITAL = Decimal(os.getenv("STARTING_CAPITAL", "100"))


# Discord notification
def send_discord(message):
    if not DISCORD_WEBHOOK:
        print("Discord: Not configured")
        return

    try:
        response = requests.post(
            DISCORD_WEBHOOK, json={"username": "JARVIS", "content": message}, timeout=10
        )

        response.raise_for_status()
        print("Discord: Notification sent")

    except requests.RequestException:
        print("Discord: Notification failed")


# Main application
def main():

    print("=" * 55)
    print("             JARVIS TRADING SYSTEM")
    print("                  VERSION 0.1")
    print("=" * 55)

    # Safety checks
    if os.getenv("TRADING_MODE") != "paper":
        print("ERROR: Only paper trading is permitted.")
        return

    if os.getenv("LIVE_TRADING_ENABLED") != "false":
        print("ERROR: Live trading must remain disabled.")
        return

    if not API_KEY or not SECRET_KEY:
        print("ERROR: Alpaca credentials are missing.")
        return

    try:
        # Connect to Alpaca paper trading
        client = TradingClient(api_key=API_KEY, secret_key=SECRET_KEY, paper=True)

        account = client.get_account()
        clock = client.get_clock()
        positions = client.get_all_positions()

        market_status = "OPEN" if clock.is_open else "CLOSED"

        print("\nSYSTEM STATUS")
        print("-" * 55)
        print("Agent Status       : ONLINE")
        print("Broker             : ALPACA")
        print("Trading Mode       : PAPER")
        print("Broker Connection  : CONNECTED")
        print("Live Trading       : DISABLED")
        print("Order Execution    : NOT IMPLEMENTED")

        print("\nACCOUNT INFORMATION")
        print("-" * 55)
        print(f"Account Status     : {account.status}")
        print(f"Cash Balance       : ${Decimal(account.cash):,.2f}")
        print(f"Portfolio Value    : ${Decimal(account.equity):,.2f}")
        print(f"Buying Power       : ${Decimal(account.buying_power):,.2f}")
        print(f"Open Positions     : {len(positions)}")

        print("\nJARVIS CONFIGURATION")
        print("-" * 55)
        print(f"Planned Capital    : ${STARTING_CAPITAL:,.2f}")
        print("Risk Guardian      : NOT IMPLEMENTED")
        print("Margin Strategy    : DISABLED")

        print("\nMARKET INFORMATION")
        print("-" * 55)
        print(f"US Market          : {market_status}")
        print(f"Next Market Open   : {clock.next_open}")
        print(f"Next Market Close  : {clock.next_close}")

        print("\nNOTIFICATIONS")
        print("-" * 55)

        send_discord(
            "**JARVIS V0.1 ONLINE**\n"
            "Broker: Alpaca\n"
            "Environment: Paper Trading\n"
            "Connection: Successful\n"
            f"Planned Capital: ${STARTING_CAPITAL:,.2f}\n"
            f"Market Status: {market_status}\n"
            f"Open Positions: {len(positions)}\n"
            "Live Trading: Disabled\n"
            "Order Execution: Not implemented"
        )

        print("\n" + "=" * 55)
        print("       JARVIS INITIALIZATION SUCCESSFUL")
        print("=" * 55)

    except Exception as error:
        print("\nJARVIS CONNECTION FAILED")
        print(f"Error Type: {type(error).__name__}")
        print("Check Alpaca paper credentials and internet connection.")


if __name__ == "__main__":
    main()
