"""AKShare loader: free, no-auth data for A-shares, US, HK, futures, forex, macro.

AKShare (https://github.com/akfamily/akshare) is a completely free financial
data aggregator covering Chinese and global markets.  No API token required.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd

from backtest.loaders.base import validate_date_range
from backtest.loaders.registry import register

logger = logging.getLogger(__name__)

_INTERVAL_MAP_DAILY = {
    "1D": "daily",
    "1W": "weekly",
    "1M": "monthly",
}


def _is_a_share(code: str) -> bool:
    return code.upper().endswith((".SZ", ".SH", ".BJ"))


# Chinese convertible bond (CB) code prefixes:
#   SH: 110xxx (公司可转债), 113xxx (新券可转债)
#   SZ: 123xxx (创业板可转债), 127xxx (中小板可转债), 128xxx (深市可转债)
# Belk's legacy strict_v2_signals table holds 125k signals against these
# codes — keeping the prefix detection narrow so we don't catch regular
# stocks that happen to start with 1.
_CB_PREFIXES_SH = frozenset({"110", "113"})
_CB_PREFIXES_SZ = frozenset({"123", "127", "128"})


def _is_cb(code: str) -> bool:
    """Detect Chinese convertible bonds (沪深可转债)."""
    upper = code.upper()
    if not upper.endswith((".SH", ".SZ")):
        return False
    digits, _, suffix = upper.partition(".")
    if len(digits) != 6 or not digits.isdigit():
        return False
    if suffix == "SH":
        return digits[:3] in _CB_PREFIXES_SH
    if suffix == "SZ":
        return digits[:3] in _CB_PREFIXES_SZ
    return False


def _is_hk(code: str) -> bool:
    return code.upper().endswith(".HK")


def _is_us(code: str) -> bool:
    return code.upper().endswith(".US")


def _is_crypto(code: str) -> bool:
    return "-USDT" in code.upper() or "/USDT" in code.upper()


# Exchange-listed ETF / LOF prefix codes:
#   SH: 50/51/52/56/58 (ETFs), SZ: 15/16 (ETFs + LOFs).
# Issue #50 — these symbols look like A-shares (.SH / .SZ) but stock_zh_a_hist
# can't price them; route through fund_etf_hist_sina instead.
_ETF_PREFIXES = frozenset({"15", "16", "50", "51", "52", "56", "58"})


def _is_etf_listed(code: str) -> bool:
    """Detect exchange-listed ETF / LOF symbols (e.g. 518880.SH, 159915.SZ)."""
    upper = code.upper()
    if not upper.endswith((".SH", ".SZ")):
        return False
    digits = upper.split(".")[0]
    if len(digits) != 6 or not digits.isdigit():
        return False
    return digits[:2] in _ETF_PREFIXES


def _is_forex(code: str) -> bool:
    """Detect forex pairs by matching against AKShare's symbol_market_map.

    Issue #54 — forex symbols (EURUSD, GBPUSD, etc.) have no exchange suffix
    and previously fell through to the A-share endpoint.
    """
    upper = code.upper().removesuffix(".FX")
    try:
        from akshare.forex.cons import symbol_market_map
    except Exception:
        return False
    return upper in symbol_market_map


@register
class DataLoader:
    """AKShare universal OHLCV loader (free, no auth)."""

    name = "akshare"
    markets = {"a_share", "us_equity", "hk_equity", "futures", "fund", "macro", "forex", "cb"}
    requires_auth = False

    def is_available(self) -> bool:
        """Available if akshare is installed."""
        try:
            import akshare  # noqa: F401
            return True
        except ImportError:
            return False

    def __init__(self) -> None:
        pass

    def fetch(
        self,
        codes: List[str],
        start_date: str,
        end_date: str,
        *,
        interval: str = "1D",
        fields: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV data via AKShare.

        Args:
            codes: Symbol list.
            start_date: YYYY-MM-DD.
            end_date: YYYY-MM-DD.
            interval: Bar size (only 1D supported currently).
            fields: Ignored.

        Returns:
            Mapping symbol -> OHLCV DataFrame.
        """
        validate_date_range(start_date, end_date)

        result: Dict[str, pd.DataFrame] = {}
        for code in codes:
            try:
                df = self._fetch_one(code, start_date, end_date, interval)
                if df is not None and not df.empty:
                    result[code] = df
            except Exception as exc:
                logger.warning("akshare failed for %s: %s", code, exc)
        return result

    def _fetch_one(
        self, code: str, start_date: str, end_date: str, interval: str,
    ) -> Optional[pd.DataFrame]:
        """Fetch a single symbol."""
        import akshare as ak

        # CB check must precede A-share — 113061.SH looks like an A-share
        # by suffix but needs the bond_zh_hs_cov_* endpoints. ETF check
        # similarly runs first because 51xxxx.SH / 15xxxx.SZ are ETFs
        # not stocks.
        if _is_cb(code):
            return self._fetch_cb(ak, code, start_date, end_date, interval)
        if _is_etf_listed(code):
            return self._fetch_etf(ak, code, start_date, end_date)
        if _is_a_share(code):
            return self._fetch_a_share(ak, code, start_date, end_date, interval)
        if _is_us(code):
            return self._fetch_us(ak, code, start_date, end_date)
        if _is_hk(code):
            return self._fetch_hk(ak, code, start_date, end_date)
        if _is_forex(code):
            return self._fetch_forex(ak, code, start_date, end_date)
        # Default: try A-share
        return self._fetch_a_share(ak, code, start_date, end_date, interval)

    def _fetch_cb(
        self, ak, code: str, start_date: str, end_date: str, interval: str,
    ) -> Optional[pd.DataFrame]:
        """Fetch Chinese convertible bond OHLCV via akshare.

        akshare's CB API splits by interval:
          - daily:   bond_zh_hs_cov_daily(symbol='sh113061')
                     → date / open / high / low / close / volume
          - intraday: bond_zh_hs_cov_min(symbol='sh113061', period='15')
                     → currently unreliable on this network (RemoteDisconnected);
                     v1 supports D1 only, minute support deferred.

        Code conversion: '113061.SH' → 'sh113061' (akshare prefix style).
        """
        digits, _, suffix = code.upper().partition(".")
        symbol = f"{suffix.lower()}{digits}"

        if interval in ("1D", "daily"):
            df = ak.bond_zh_hs_cov_daily(symbol=symbol)
            if df is None or df.empty:
                return None
            # bond_zh_hs_cov_daily columns: date / open / high / low / close / volume
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            for col in ("open", "high", "low", "close", "volume"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df[["open", "high", "low", "close", "volume"]].dropna(
                subset=["open", "high", "low", "close"]
            )
            return df.loc[start_date:end_date]

        # Minute support — currently flaky from akshare network path.
        # Belk's original system pulled minute data straight from
        # EastMoney (CBPriceUrl) via Go HTTP; if M5/M15/H1 demand
        # arises, write a direct EastMoney fetcher rather than retrying
        # akshare's bond_zh_hs_cov_min.
        logger.warning(
            "akshare CB intraday (%s) not supported in v1 — D1 only. "
            "Code %s skipped.", interval, code,
        )
        return None

    def _fetch_a_share(
        self, ak, code: str, start_date: str, end_date: str, interval: str,
    ) -> Optional[pd.DataFrame]:
        """Fetch A-share with East-Money + Sina dual-path.

        Tries `stock_zh_a_hist` (push2his.eastmoney.com) first — richer
        columns and intraday support. Some VPS/datacenter networks
        can't reach that host and trip RemoteDisconnected. In that
        case fall back to `stock_zh_a_daily` (Sina, hq.sinajs.cn),
        which uses a different CDN and tends to be reachable when EM
        isn't. Sina's response uses lowercase English column names so
        the date_col argument to _normalize switches accordingly.
        """
        symbol = code.split(".")[0]
        period = _INTERVAL_MAP_DAILY.get(interval, "daily")
        sd = start_date.replace("-", "")
        ed = end_date.replace("-", "")

        try:
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period=period,
                start_date=sd,
                end_date=ed,
                adjust="qfq",
            )
            if df is not None and not df.empty:
                return self._normalize(df, date_col="日期")
        except Exception as exc:
            logger.warning(
                "akshare stock_zh_a_hist failed for %s: %s; trying Sina fallback",
                code, exc,
            )

        # Sina expects an explicit market prefix; daily-only.
        suffix = code.rsplit(".", 1)[-1].upper() if "." in code else ""
        market_prefix = {"SH": "sh", "SZ": "sz"}.get(suffix)
        if market_prefix is None:
            return None
        try:
            df = ak.stock_zh_a_daily(
                symbol=f"{market_prefix}{symbol}",
                start_date=sd,
                end_date=ed,
                adjust="qfq",
            )
        except Exception as exc:
            logger.warning(
                "akshare stock_zh_a_daily fallback also failed for %s: %s",
                code, exc,
            )
            return None
        if df is None or df.empty:
            return None
        return self._normalize(df, date_col="date")

    def _fetch_us(self, ak, code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch US stock via stock_us_hist."""
        symbol = code.replace(".US", "")
        # akshare uses the format like "105.AAPL" for NASDAQ
        # Try common prefixes
        for prefix in ["105.", "106.", ""]:
            try:
                df = ak.stock_us_hist(
                    symbol=f"{prefix}{symbol}",
                    period="daily",
                    start_date=start_date.replace("-", ""),
                    end_date=end_date.replace("-", ""),
                    adjust="qfq",
                )
                if df is not None and not df.empty:
                    return self._normalize(df, date_col="日期")
            except Exception:
                continue
        return None

    def _fetch_etf(self, ak, code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch exchange-listed ETF / LOF via fund_etf_hist_sina.

        Sina symbol format is ``sh518880`` / ``sz159915``. The endpoint returns
        the full history; we filter to the requested window after fetching.
        """
        digits, _, suffix = code.upper().partition(".")
        symbol = f"{suffix.lower()}{digits}"
        df = ak.fund_etf_hist_sina(symbol=symbol)
        if df is None or df.empty:
            return None
        df = self._normalize(df, date_col="date")
        # fund_etf_hist_sina returns full history — clip to window.
        return df.loc[start_date:end_date]

    def _fetch_forex(self, ak, code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch forex pair via forex_hist_em.

        Columns returned are 日期 / 代码 / 名称 / 今开 / 最新价 / 最高 / 最低 / 振幅
        — note ``最新价`` (latest) plays the role of close. Volume isn't reported,
        so we synthesize a zero column to satisfy the OHLCV contract.
        """
        symbol = code.upper().removesuffix(".FX")
        df = ak.forex_hist_em(symbol=symbol)
        if df is None or df.empty:
            return None
        df = df.rename(columns={
            "日期": "trade_date",
            "今开": "open",
            "最新价": "close",
            "最高": "high",
            "最低": "low",
        })
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.set_index("trade_date").sort_index()
        df["volume"] = 0.0
        for col in ("open", "high", "low", "close"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[["open", "high", "low", "close", "volume"]].dropna(
            subset=["open", "high", "low", "close"]
        )
        return df.loc[start_date:end_date]

    def _fetch_hk(self, ak, code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch HK stock via stock_hk_hist."""
        symbol = code.replace(".HK", "").zfill(5)
        df = ak.stock_hk_hist(
            symbol=symbol,
            period="daily",
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
            adjust="qfq",
        )
        if df is None or df.empty:
            return None
        return self._normalize(df, date_col="日期")

    @staticmethod
    def _normalize(df: pd.DataFrame, date_col: str = "日期") -> pd.DataFrame:
        """Normalize AKShare DataFrame to standard OHLCV schema.

        AKShare Chinese column names: 日期, 开盘, 最高, 最低, 收盘, 成交量
        AKShare English column names: date, open, high, low, close, volume
        """
        col_map_cn = {"开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume"}
        col_map_en = {"date": "trade_date", "open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"}

        if date_col in df.columns:
            df = df.rename(columns={date_col: "trade_date"})
        elif "date" in df.columns:
            df = df.rename(columns={"date": "trade_date"})

        # Try Chinese column names first, then English
        if "开盘" in df.columns:
            df = df.rename(columns=col_map_cn)
        else:
            df = df.rename(columns=col_map_en)

        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.set_index("trade_date").sort_index()

        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        ohlcv_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[ohlcv_cols].dropna(subset=["open", "high", "low", "close"])
        if "volume" not in df.columns:
            df["volume"] = 0.0
        return df
