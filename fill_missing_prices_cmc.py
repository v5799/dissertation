"""
fill_missing_prices_cmc.py
==========================

This utility extends an existing merged daily price table by
filling in missing DAO price series using the CoinMarketCap
(CMC) API.  After manually downloading price data for as many
DAOs as possible, run this script to query CMC for any DAO
tokens that remain absent from the merged price file.  The
script relies on the ``top_daos.csv`` (produced by
``analysis_80daos_updated.py``) to map DAO titles to their
token symbols.  It then uses the CMC historical quotes
endpoint to download daily close prices in USD and adds
columns to the merged price table.  Missing values (e.g., for
dates before a token started trading) are left as ``NaN``.

Example usage from a command line:

    python fill_missing_prices_cmc.py \
        --merged-file merged_prices.csv \
        --top-daos top_120_daos.csv \
        --deepdao-key <your_deepdao_key> \
        --cmc-key <your_cmc_key> \
        --start-date 2018-01-01 \
        --end-date 2025-06-01 \
        --output merged_with_cmc.csv

Notes
-----
* The script makes one CMC API call per missing DAO.  Keep
  track of your monthly request quota (e.g., 10,000 calls in
  the demo tier) and adjust the number of DAOs accordingly.
* Only DAOs whose token symbols can be determined will be
  fetched.  If no symbol is available, the DAO is skipped and
  reported in the summary.
* The resulting file contains a ``date`` column and one
  column per DAO, covering the union of dates across the
  existing merged table and newly fetched series.  Prices are
  expressed in USD and aligned by calendar date.

"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import os
import pandas as pd
import requests
import time  # for sleep
import numpy as np

# We'll reuse DeepDAO helper functions from analysis scripts, but
# reimplement minimal versions here to avoid importing user code.

DEEPDAO_BASE_URL = "https://api.deepdao.io/v0.1"


def _dd_headers(key: Optional[str] = None) -> Dict[str, str]:
    """Construct HTTP headers for DeepDAO requests."""
    headers = {"accept": "application/json"}
    if key:
        headers["x-api-key"] = key
    return headers


def fetch_top_organizations(limit: int = 120, api_key: Optional[str] = None) -> pd.DataFrame:
    """Fetch the top DAOs by AUM from DeepDAO.

    Parameters
    ----------
    limit : int
        Number of DAOs to request.  Note that some may be filtered
        later depending on token availability.
    api_key : str or None
        DeepDAO API key.

    Returns
    -------
    pandas.DataFrame
        DataFrame of DAOs.  Columns of interest include ``title``
        and ``symbol`` (if present).
    """
    url = f"{DEEPDAO_BASE_URL}/organizations/top_aum_organizations?limit={limit}"
    resp = requests.get(url, headers=_dd_headers(api_key), timeout=30)
    resp.raise_for_status()
    js = resp.json()
    if isinstance(js, dict):
        data = js.get("data") or js.get("resources")
    else:
        data = js
    return pd.DataFrame(data)


def _cmc_headers(key: Optional[str] = None) -> Dict[str, str]:
    """Construct HTTP headers for CoinMarketCap requests."""
    headers = {"accepts": "application/json"}
    if key:
        headers["X-CMC_PRO_API_KEY"] = key
    return headers


def fetch_cmc_series(symbol: str, start_date: str, end_date: str,
                     cmc_key: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Retrieve daily historical closing prices from CoinMarketCap.

    Parameters
    ----------
    symbol : str
        Token symbol as recognised by CoinMarketCap.
    start_date, end_date : str
        Date bounds in YYYY-MM-DD format.
    cmc_key : str or None
        Your CMC API key.

    Returns
    -------
    pandas.DataFrame or None
        DataFrame with ``date`` and ``<symbol>`` columns, or
        ``None`` if retrieval failed.
    """
    url = "https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/historical"
    params = {
        "symbol": symbol,
        "time_start": start_date,
        "time_end": end_date,
        "interval": "daily",
        "convert": "USD",
    }
    headers = _cmc_headers(cmc_key)
    # Add Accept and Accept-Encoding headers for best practices
    headers.setdefault("Accept", "application/json")
    headers.setdefault("Accept-Encoding", "deflate, gzip")
    try:
        r = requests.get(url, headers=headers, params=params, timeout=30)
    except Exception:
        return None
    # Handle common HTTP errors gracefully
    if r.status_code == 429:
        # Rate limit exceeded; caller should back off and retry later
        return None
    if r.status_code != 200:
        return None
    js = None
    try:
        js = r.json()
    except Exception:
        return None
    # Check for status.error_code in the response
    status = js.get("status")
    if isinstance(status, dict) and status.get("error_code", 0) != 0:
        # Invalid key or plan restriction
        return None
    data = js.get("data", {}).get(symbol)
    if not data or not isinstance(data.get("quotes"), list):
        return None
    records = data["quotes"]
    dates: List[pd.Timestamp] = []
    prices: List[float] = []
    for rec in records:
        ts = rec.get("timestamp")
        quote = rec.get("quote", {}).get("USD", {})
        price = quote.get("close")
        if ts and price is not None:
            dates.append(pd.to_datetime(ts).normalize())
            prices.append(price)
    if not dates:
        return None
    return pd.DataFrame({"date": dates, symbol: prices})


def build_title_to_symbol_map(top_daos_df: pd.DataFrame, deepdao_key: str) -> Dict[str, str]:
    """Create a mapping from DAO title/name to token symbol.

    The top DAO CSV typically contains only the ``id`` and
    ``title`` columns.  This helper fetches the full top DAO
    metadata from DeepDAO and extracts any available ``symbol"
    field.  If no symbol is present for a DAO, that entry is
    omitted from the mapping.

    Parameters
    ----------
    top_daos_df : pandas.DataFrame
        DataFrame containing at least a ``title`` column.
    deepdao_key : str
        DeepDAO API key for fetching metadata.

    Returns
    -------
    dict
        Mapping from DAO title to token symbol.
    """
    # Determine how many DAOs to fetch.  We fetch at least as many
    # as in the top list; DeepDAO may return more but that's fine.
    n = len(top_daos_df)
    meta_df = fetch_top_organizations(limit=n, api_key=deepdao_key)
    if meta_df.empty:
        return {}
    if "title" not in meta_df.columns or "symbol" not in meta_df.columns:
        return {}
    # Some DAOs have duplicate titles (e.g., alias vs canonical name)
    # We build a simple mapping; if duplicates exist, later ones
    # overwrite earlier ones which is acceptable for our use case.
    mapping = meta_df.set_index("title")["symbol"].dropna().to_dict()
    return mapping


def fill_missing_prices(merged_df: pd.DataFrame,
                        top_daos_df: pd.DataFrame,
                        cmc_key: str,
                        deepdao_key: str,
                        start_date: str,
                        end_date: str,
                        *,
                        include_missing_columns: bool = True,
                        verbose: bool = True) -> pd.DataFrame:
    """Fill missing DAO price series using CMC.

    Parameters
    ----------
    merged_df : pandas.DataFrame
        DataFrame of existing prices, with a ``date`` column and
        one column per DAO.
    top_daos_df : pandas.DataFrame
        DataFrame listing DAOs (must include ``title``).
    cmc_key : str
        CoinMarketCap API key.
    deepdao_key : str
        DeepDAO API key.
    start_date, end_date : str
        Date boundaries for historical price retrieval.
    verbose : bool, default True
        Whether to print progress messages.

    Returns
    -------
    pandas.DataFrame
        Updated DataFrame with additional DAO columns filled from CMC.
    """
    if "date" not in merged_df.columns:
        raise ValueError("merged_df must contain a 'date' column")
    merged_df = merged_df.copy()
    merged_df["date"] = pd.to_datetime(merged_df["date"])
    merged_df = merged_df.set_index("date")
    # Determine which DAO columns are missing
    existing_daos: set[str] = set(merged_df.columns)
    if "title" not in top_daos_df.columns:
        raise ValueError("top_daos_df must contain a 'title' column")
    desired_daos = list(top_daos_df["title"].dropna().unique())
    missing_daos = [dao for dao in desired_daos if dao not in existing_daos]
    if verbose:
        print(f"Found {len(existing_daos)} price series; {len(missing_daos)} missing")
    # Build mapping of title to symbol using DeepDAO metadata
    title_to_sym = build_title_to_symbol_map(top_daos_df, deepdao_key)
    filled_count = 0
    skipped: List[str] = []
    for dao in missing_daos:
        symbol = title_to_sym.get(dao)
        series_df: Optional[pd.DataFrame] = None
        if symbol:
            if verbose:
                print(f"Fetching {dao} ({symbol}) from CMC...")
            series_df = fetch_cmc_series(symbol, start_date, end_date, cmc_key)
        if series_df is not None and not series_df.empty:
            series_df = series_df.set_index("date").rename(columns={symbol: dao})
            series_df = series_df.reindex(merged_df.index, copy=False)
            merged_df = merged_df.join(series_df, how="left")
            filled_count += 1
            # Sleep to respect API rate limits
            time.sleep(2)
        else:
            skipped.append(dao)
            # If include_missing_columns is True, add a column of NaNs for this DAO
            if include_missing_columns:
                if verbose:
                    print(f"  -> no data or symbol for {dao}, adding empty column")
                merged_df[dao] = np.nan
    if verbose:
        print(f"Filled {filled_count} DAO price series from CMC; skipped {len(skipped)}")
        if skipped:
            print("Skipped due to missing symbol or data:", ", ".join(skipped))
    merged_df = merged_df.reset_index()
    return merged_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill missing DAO prices using CoinMarketCap")
    parser.add_argument("--merged-file", required=True,
                        help="Path to merged price CSV (output of merge script)")
    parser.add_argument("--top-daos", required=True,
                        help="CSV file listing top DAOs (from analysis script)")
    parser.add_argument("--deepdao-key", required=True,
                        help="DeepDAO API key for symbol lookup")
    parser.add_argument("--cmc-key", required=True,
                        help="CoinMarketCap API key")
    parser.add_argument("--start-date", required=True,
                        help="Earliest date (YYYY-MM-DD) for price retrieval")
    parser.add_argument("--end-date", required=True,
                        help="Latest date (YYYY-MM-DD) for price retrieval")
    parser.add_argument("--output", required=True,
                        help="Path to write the updated merged price CSV")
    parser.add_argument("--verbose", action="store_true", help="Print progress messages")
    args = parser.parse_args()
    # Support both CSV and Excel formats for the merged price file.  Some
    # environments (e.g., Jupyter downloads) may save .xls by default.
    input_path = Path(args.merged_file)
    if input_path.suffix.lower() in {".xls", ".xlsx"}:
        merged = pd.read_excel(args.merged_file)
    else:
        merged = pd.read_csv(args.merged_file)
    top_daos = pd.read_csv(args.top_daos)
    updated = fill_missing_prices(
        merged_df=merged,
        top_daos_df=top_daos,
        cmc_key=args.cmc_key,
        deepdao_key=args.deepdao_key,
        start_date=args.start_date,
        end_date=args.end_date,
        verbose=args.verbose,
    )
    updated.to_csv(args.output, index=False)
    if args.verbose:
        print(f"Saved updated price table to {args.output} (shape {updated.shape})")


if __name__ == "__main__":
    main()
