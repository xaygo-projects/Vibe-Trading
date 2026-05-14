"""wkserver data loader — reads OHLCV from wkserver instead of external APIs.

Architecture: in the quant-wk project, wkcrawler is the SINGLE entity that
talks to external data sources (akshare / tushare / okx / yfinance / ccxt /
futu). wkserver owns the PG `ohlcv_bars` table and exposes two endpoints:

  GET  /internal/data/ohlcv   — read current PG content
  POST /internal/data/backfill — trigger sync backfill (wkcrawler picks it
                                  up and pushes back via /internal/crawler/push)

This loader does exactly that: GET first, then if the returned bars don't
cover the requested window, POST backfill (waits up to 120s), then GET
again. End result: vt's backtest runs get data from PG, which is kept
fresh by wkcrawler's realtime push + on-demand backfill.

Env vars (set in vt's docker-compose.yaml):
  WKSERVER_URL          required, e.g. http://quantwk_server:8811
  WKSERVER_TOKEN        required, same INTERNAL_API_TOKEN wkserver uses
  WKSERVER_TIMEOUT_SECONDS   optional, default 30
  WKSERVER_BACKFILL_TIMEOUT  optional, default 120 (sent to /backfill)
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import pandas as pd
import requests

from backtest.loaders.base import validate_date_range
from backtest.loaders.registry import register


logger = logging.getLogger(__name__)


# Markets we route through wkserver. Mirrors the universe wkcrawler can
# fetch (all of vt's loaders' market lists combined).
_SUPPORTED_MARKETS = {
    "crypto", "cb", "a_share", "us_equity", "hk_equity",
    "forex", "futures", "fund", "macro",
    # Legacy aliases used by the Go wkcrawler config:
    "gold", "stock",
}


@register
class DataLoader:
    """Read OHLCV via wkserver's /internal/data/* endpoints.

    Auto-triggers backfill when PG doesn't fully cover the requested
    window. From the caller's perspective this is the same as any other
    loader — give it codes + date range, get back DataFrames.
    """

    name = "wkserver"
    markets = _SUPPORTED_MARKETS
    requires_auth = True  # needs WKSERVER_TOKEN

    def is_available(self) -> bool:
        """Available if both env vars are set."""
        return bool(os.getenv("WKSERVER_URL", "")) and bool(
            os.getenv("WKSERVER_TOKEN", "")
        )

    def __init__(self) -> None:
        self.base_url = os.getenv("WKSERVER_URL", "").rstrip("/")
        self.token = os.getenv("WKSERVER_TOKEN", "")
        self.timeout = float(os.getenv("WKSERVER_TIMEOUT_SECONDS", "30"))
        self.backfill_timeout = int(os.getenv("WKSERVER_BACKFILL_TIMEOUT", "120"))
        if not self.base_url or not self.token:
            raise RuntimeError(
                "wkserver loader requires WKSERVER_URL + WKSERVER_TOKEN env vars"
            )

    def fetch(
        self,
        codes: List[str],
        start_date: str,
        end_date: str,
        *,
        interval: str = "1D",
        fields: Optional[List[str]] = None,
        market: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV for each code, going through wkserver.

        Args:
            codes: Symbol list, e.g. ["113061.SH", "AAPL"].
            start_date: YYYY-MM-DD inclusive.
            end_date: YYYY-MM-DD inclusive.
            interval: vt-style ("1m" / "5m" / "1H" / "1D" / etc.)
            fields: Ignored.
            market: Optional override (defaults to auto-detect per code).

        Returns:
            Mapping symbol -> OHLCV DataFrame.
        """
        validate_date_range(start_date, end_date)

        out: Dict[str, pd.DataFrame] = {}
        for code in codes:
            market_for_code = market or _detect_market(code)
            tf = _normalize_interval(interval)
            try:
                df = self._fetch_one(code, market_for_code, tf, start_date, end_date)
                if df is not None and not df.empty:
                    out[code] = df
            except Exception as exc:
                logger.warning(
                    "wkserver fetch failed for %s (%s/%s/%s..%s): %s",
                    code, market_for_code, tf, start_date, end_date, exc,
                )
        return out

    # ─── per-code logic ───────────────────────────────────────────────

    def _fetch_one(
        self, code: str, market: str, timeframe: str,
        start_date: str, end_date: str,
    ) -> Optional[pd.DataFrame]:
        """1. GET → check coverage. 2. POST backfill if incomplete. 3. GET again."""
        df = self._read_ohlcv(code, market, timeframe, start_date, end_date)
        if _covers_range(df, start_date, end_date):
            return df

        # Gap detected — trigger backfill. wkcrawler will fetch the
        # whole [start, end] window from akshare/okx/etc and push.
        # Idempotent on the server side (upsert dedup), so overlap with
        # what we already have is harmless.
        logger.info(
            "wkserver: PG miss for %s/%s/%s — triggering backfill [%s..%s]",
            market, code, timeframe, start_date, end_date,
        )
        try:
            self._trigger_backfill(code, market, timeframe, start_date, end_date)
        except Exception as exc:
            logger.warning(
                "wkserver backfill trigger failed for %s/%s/%s: %s — returning "
                "whatever PG had", market, code, timeframe, exc,
            )
            return df

        # Re-read after backfill completes.
        return self._read_ohlcv(code, market, timeframe, start_date, end_date)

    def _read_ohlcv(
        self, code: str, market: str, timeframe: str,
        start_date: str, end_date: str,
    ) -> Optional[pd.DataFrame]:
        """GET /internal/data/ohlcv → DataFrame."""
        resp = requests.get(
            f"{self.base_url}/internal/data/ohlcv",
            params={
                "code": code,
                "market": market,
                "timeframe": timeframe,
                "start_date": start_date,
                "end_date": end_date,
            },
            headers={"X-Internal-Token": self.token},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        body = resp.json()
        # wkserver envelope: {code, data:{bars: [...]}}
        data = body.get("data") or {}
        bars = data.get("bars") or []
        if not bars:
            return None
        df = pd.DataFrame(bars)
        # ts is unix seconds (UTC). Build DatetimeIndex.
        df["trade_date"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert(None)
        df = df.set_index("trade_date").sort_index()
        df = df[["open", "high", "low", "close", "volume"]]
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.dropna(subset=["open", "high", "low", "close"])

    def _trigger_backfill(
        self, code: str, market: str, timeframe: str,
        start_date: str, end_date: str,
    ) -> None:
        """POST /internal/data/backfill — sync wait for completion."""
        resp = requests.post(
            f"{self.base_url}/internal/data/backfill",
            json={
                "code": code,
                "market": market,
                "timeframe": timeframe,
                "start_date": start_date,
                "end_date": end_date,
                "timeout_seconds": self.backfill_timeout,
            },
            headers={
                "X-Internal-Token": self.token,
                "Content-Type": "application/json",
            },
            timeout=self.backfill_timeout + 30,
        )
        if resp.status_code == 200:
            return  # done
        if resp.status_code == 504:
            # Backfill still running on wkcrawler side. The bars *might*
            # show up by the time we re-GET, so don't bail loudly.
            logger.warning(
                "wkserver backfill timed out (server still working): %s",
                resp.text[:200],
            )
            return
        # Real failure (502 / 400 etc.) — propagate so caller falls back.
        raise RuntimeError(f"backfill returned {resp.status_code}: {resp.text[:200]}")


# ─── helpers ──────────────────────────────────────────────────────────


def _normalize_interval(interval: str) -> str:
    """vt-style ('1H', '1D') → wkserver-side ('1h', '1d').

    wkserver normalizes timeframe to lowercase server-side, so either
    case works; we lowercase here for HTTP URL consistency.
    """
    return interval.strip().lower()


def _covers_range(df: Optional[pd.DataFrame], start_date: str, end_date: str) -> bool:
    """Best-effort 'do we already have data for this range?' check.

    Loose: returns True only when DataFrame exists AND its min/max
    DatetimeIndex span covers the requested window. Doesn't try to
    detect tiny gaps inside the range — that's wkserver-upsert's job
    to make harmless.
    """
    if df is None or df.empty:
        return False
    try:
        actual_min = df.index.min()
        actual_max = df.index.max()
        wanted_min = pd.Timestamp(start_date)
        wanted_max = pd.Timestamp(end_date)
    except Exception:
        return False
    return actual_min <= wanted_min and actual_max >= wanted_max


# Market detection — mirrors runner.py's _MARKET_PATTERNS so we can route
# without depending on it directly (avoids circular import at module load).
import re as _re

_MARKET_PATTERNS = [
    (_re.compile(r"^(11[03]\d{3}\.SH|12[378]\d{3}\.SZ)$", _re.I), "cb"),
    (_re.compile(r"^\d{6}\.(SZ|SH|BJ)$", _re.I), "a_share"),
    (_re.compile(r"^(51|15|56)\d{4}\.(SZ|SH)$", _re.I), "a_share"),
    (_re.compile(r"^[A-Z]+\.US$", _re.I), "us_equity"),
    (_re.compile(r"^\d{3,5}\.HK$", _re.I), "hk_equity"),
    (_re.compile(r"^[A-Z]+-USDT$", _re.I), "crypto"),
    (_re.compile(r"^[A-Z]+/USDT$", _re.I), "crypto"),
    (_re.compile(r"^[A-Za-z]{1,2}\d{3,4}\.(ZCE|DCE|SHFE|INE|CFFEX|GFEX)$", _re.I), "futures"),
    (_re.compile(r"^[A-Z]{3}/[A-Z]{3}$"), "forex"),
    (_re.compile(r"^[A-Z]{6}\.FX$"), "forex"),
]


def _detect_market(code: str) -> str:
    for pattern, market in _MARKET_PATTERNS:
        if pattern.match(code):
            return market
    return "a_share"
