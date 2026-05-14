"""China convertible bond (CB / 可转债) backtest engine.

Market rules (vs A-share):
  - **T+0**: can sell same day as buy (different from A-share T+1)
  - No short selling for retail (same as A-share)
  - **No daily price limits** (CB can move freely — even when emergency
    halts trigger on 30% moves, intraday limits are not enforced like
    stocks; for backtest purposes we don't enforce a static limit)
  - **Minimum unit: 10 张** (1 张 = 100 RMB face value, so smallest
    tradeable notional is 1000 RMB nominal). Sizes are rounded down to
    multiples of 10.
  - **Commission: 万2 (0.0002) bilateral**, typically no minimum
    fee at modern brokers (legacy was ¥1 min — kept as config knob)
  - **No stamp tax** (印花税仅对股票, CB 免)
  - **No transfer fee** (CB 无过户费)
  - **Slippage**: similar to A-share, default 0.1%

This engine is a fork of ChinaAEngine with CB-specific rule overrides,
used by belk_classic strategy template for Chinese convertible bond
backtests (sh11xxxx / sz12xxxx codes).
"""

from __future__ import annotations

import pandas as pd

from backtest.engines.base import BaseEngine


class ChinaCBEngine(BaseEngine):
    """China convertible bond market engine.

    Config keys (all have CB-tuned defaults):
      - commission_rate: default 0.0002 (万2, bilateral)
      - commission_min: default 0.0 (modern brokers waive minimum for CB;
        set to 1.0 if you want legacy 1元 floor)
      - slippage: default 0.001
      - lot_size: default 10 (10 张 per minimum order)
    """

    def __init__(self, config: dict):
        config = {**config, "leverage": 1.0}  # CB: no leverage for retail
        super().__init__(config)
        self.commission_rate: float = config.get("commission_rate", 0.0002)
        self.commission_min: float = config.get("commission_min", 0.0)
        self.slippage_rate: float = config.get("slippage", 0.001)
        self.lot_size: int = config.get("lot_size", 10)

    def can_execute(self, symbol: str, direction: int, bar: pd.Series) -> bool:
        """CB execution rules.

        Args:
            symbol: CB code (e.g. 113061.SH).
            direction: 1 (buy), -1 (short — always blocked), 0 (sell/close).
            bar: Current bar OHLCV.

        Returns:
            True if the trade is allowed.

        Notes:
            - No short selling (retail constraint).
            - No T+1 — CB is T+0, can close same day.
            - No price limit check — CB moves freely.
        """
        if direction == -1:
            return False
        return True

    def round_size(self, raw_size: float, price: float) -> float:
        """Round down to lot_size (10 张) increments.

        CB sizes are in 张 (face-value units). Minimum tradeable unit is
        10 张 = 1000 RMB nominal. Raw size is in 张; we floor to nearest
        multiple of lot_size.
        """
        return max(int(raw_size / self.lot_size) * self.lot_size, 0)

    def calc_commission(self, size: float, price: float, direction: int, is_open: bool) -> float:
        """CB fee structure: commission only (no stamp tax, no transfer fee).

        Args:
            size: Number of 张 (face-value units).
            price: Trade price (RMB per 张).
            direction: 1 (buy) / 0 (sell) / -1 (short).
            is_open: True for opening order, False for closing.

        Returns:
            Total fee in RMB.
        """
        notional = size * price
        comm = notional * self.commission_rate
        if self.commission_min > 0:
            comm = max(comm, self.commission_min)
        return comm

    def apply_slippage(self, price: float, direction: int) -> float:
        """CB slippage (default 0.1%, similar to A-share).

        Direction encoding follows base contract: 1 means buy (price
        rounds up against us), -1/0 means sell (price rounds down).
        """
        return price * (1 + direction * self.slippage_rate)
