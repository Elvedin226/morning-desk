"""Alpaca broker interface. Paper by default, and loudly so.

Two safety properties matter more than anything else here:

  1. PAPER IS THE DEFAULT. Live trading requires setting ALPACA_LIVE=1 *and*
     supplying live keys. There is no way to reach the live endpoint by
     forgetting a flag.
  2. NO KEYS = DRY RUN. With no credentials the broker still works, reports a
     simulated account, and logs every order it *would* have placed. The runner
     can be developed and tested end to end before an account exists.

Alpaca is used rather than Robinhood because Robinhood has no official API for
stocks or options - only crypto. The unofficial libraries violate their terms of
service and risk account suspension.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import requests

PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"

DRY_EQUITY = 421.0  # what DRY RUN pretends the account holds


@dataclass
class Order:
    symbol: str
    qty: float
    side: str          # "buy" | "sell"
    intended: float    # price when the decision was made
    filled: float | None = None
    order_id: str | None = None
    status: str = "dry_run"

    @property
    def slippage_pct(self) -> float | None:
        """Actual fill against intended price. THE number a backtest cannot give
        you, and the one that killed two strategies in this project."""
        if self.filled is None or self.intended <= 0:
            return None
        sign = 1 if self.side == "buy" else -1
        return sign * (self.filled - self.intended) / self.intended


class Broker:
    def __init__(self):
        self.key = os.environ.get("ALPACA_KEY_ID", "")
        self.secret = os.environ.get("ALPACA_SECRET_KEY", "")
        self.live = os.environ.get("ALPACA_LIVE", "") == "1"
        self.dry = not (self.key and self.secret)
        self.base = LIVE_URL if self.live else PAPER_URL

    @property
    def mode(self) -> str:
        if self.dry:
            return "DRY RUN (no credentials - nothing is sent anywhere)"
        return "LIVE - REAL MONEY" if self.live else "PAPER"

    def _headers(self) -> dict:
        return {"APCA-API-KEY-ID": self.key, "APCA-API-SECRET-KEY": self.secret}

    def _get(self, path: str):
        r = requests.get(f"{self.base}{path}", headers=self._headers(), timeout=20)
        r.raise_for_status()
        return r.json()

    def equity(self) -> float:
        if self.dry:
            return DRY_EQUITY
        return float(self._get("/v2/account")["equity"])

    def positions(self) -> list[dict]:
        """Open positions. Read at startup so a restart mid-position does not
        double up - the runner reconciles against this, never against memory."""
        if self.dry:
            return []
        return self._get("/v2/positions")

    def is_open(self) -> bool:
        """Market open right now? Orders placed while closed queue to the next
        session, which silently changes the fill price the decision assumed."""
        if self.dry:
            return True
        return bool(self._get("/v2/clock")["is_open"])

    def submit(self, symbol: str, qty: float, side: str, intended: float) -> Order:
        """Market order. Notional (dollar) orders would allow fractional shares
        on any name, but market-order fills are what the strategy assumes, so
        this stays a plain market order and the runner rounds the size."""
        order = Order(symbol=symbol, qty=qty, side=side, intended=intended)
        if self.dry:
            return order

        body = {"symbol": symbol, "qty": str(qty), "side": side,
                "type": "market", "time_in_force": "day"}
        r = requests.post(f"{self.base}/v2/orders", json=body,
                          headers=self._headers(), timeout=20)
        r.raise_for_status()
        data = r.json()
        order.order_id = data.get("id")
        order.status = data.get("status", "submitted")
        fill = data.get("filled_avg_price")
        order.filled = float(fill) if fill else None
        return order

    def order_status(self, order_id: str) -> dict:
        return self._get(f"/v2/orders/{order_id}")
