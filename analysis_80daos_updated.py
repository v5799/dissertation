"""
analysis_80daos_updated.py
===========================

This script implements a self‑contained workflow for computing
governance participation **z‑scores** for a subset of decentralised
autonomous organisations (DAOs) over a three‑year sample window and
optionally merging those scores with token price data.  It has been
tailored to the requirements of the user's *mini project* and is meant
to be executed in a Jupyter notebook or directly via the command
line.  The key features include:

* **Three‑year window:** Monthly governance data are aggregated from
  one month before ``start_label`` through the month prior to
  ``end_label``.  For example, with the default labels of
  ``2021‑12‑01`` and ``2025‑01‑01``, the aggregation covers
  December 2021 through December 2024, labelled 2022‑01‑01 through
  2025‑01‑01.
* **Top 120 DAOs:** We request the top 120 organisations by assets
  under management (AUM) from DeepDAO.  Up to 80 of these (by
  default) are processed for governance metrics; the extended list
  functions as a buffer when price data are missing.
* **Progress reporting:** Each significant operation logs a clear
  message to the console.  This includes API connectivity tests,
  organisation processing, price loading and merging, and final
  output.
* **Missing price handling:** When merging price CSVs, any DAO whose
  title does not appear among the columns of the price table is
  recorded in ``missing_price_files.txt``.  This file allows the
  user to identify which tokens require manual price downloads.  The
  z‑score computation continues regardless of missing prices.
* **API keys:** DeepDAO and CoinMarketCap keys can be provided as
  arguments or via environment variables (``DEEPDAO_API_KEY`` and
  ``CMC_API_KEY``).  The code validates the DeepDAO key before
  proceeding.

Example usage::

    from analysis_80daos_updated import run_analysis

    # Replace with your actual keys
    deepdao_key = "YOUR_DEEPDAO_API_KEY"
    cmc_key     = "YOUR_CMC_API_KEY"  # optional

    run_analysis(
        api_key=deepdao_key,
        cmc_key=cmc_key,
        price_directory=r"C:\\Users\\cheet\\Downloads\\DAO priciing data",
        start_label="2021-12-01",
        end_label="2025-01-01",
        max_daos=80,
        top_limit=120,
        verbose=True,
    )

The function will generate the following files in the current working
directory:

* ``top_120_daos.csv`` – metadata for the top 120 DAOs (id, title,
  symbol, treasuryUsd).
* ``master_zscores.csv`` – a panel of monthly z‑scores (rows are
  dates; columns are DAO titles).
* ``master_zscores_with_price.csv`` – the z‑score panel merged with
  monthly closing prices.  Columns corresponding to DAOs with no
  available price data remain NaN.
* ``missing_price_files.txt`` – list of DAO titles with missing price
  CSVs (one per line).

This script can be run directly as a module from the command line
using ``python analysis_80daos_updated.py --help`` for argument
options.
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# DeepDAO API helpers
# ---------------------------------------------------------------------------

BASE_URL = "https://api.deepdao.io/v0.1"


def _get_headers(api_key: Optional[str] = None) -> Dict[str, str]:
    """Return HTTP headers for DeepDAO requests.

    Parameters
    ----------
    api_key : str, optional
        DeepDAO API key.  If not provided, the ``DEEPDAO_API_KEY``
        environment variable is used.

    Returns
    -------
    dict
        Dictionary of headers for the request.
    """
    key = api_key or os.getenv("DEEPDAO_API_KEY")
    headers = {"accept": "application/json"}
    if key:
        headers["x-api-key"] = key
    return headers


def test_connection(api_key: Optional[str] = None) -> bool:
    """Test connectivity to the DeepDAO API.

    Performs a simple request to the top AUM endpoint to verify
    authentication.  Logs the result to stdout.

    Returns
    -------
    bool
        True if the request succeeds (HTTP 200); False otherwise.
    """
    url = f"{BASE_URL}/organizations/top_aum_organizations?limit=1"
    try:
        resp = requests.get(url, headers=_get_headers(api_key), timeout=10)
        if resp.status_code == 200:
            print("DeepDAO API connection successful.")
            return True
        else:
            print(f"DeepDAO API returned status {resp.status_code}: {resp.text[:200]}")
            return False
    except Exception as e:
        print(f"Error testing DeepDAO API connection: {e}")
        return False


def fetch_top_organizations(limit: int = 120, api_key: Optional[str] = None) -> pd.DataFrame:
    """Fetch the top DAOs by AUM from DeepDAO.

    Parameters
    ----------
    limit : int
        Maximum number of DAOs to request (API limit is 2000).
    api_key : str, optional
        DeepDAO API key.

    Returns
    -------
    pandas.DataFrame
        Data frame with columns such as ``id``, ``title`` and ``symbol``.
    """
    url = f"{BASE_URL}/organizations/top_aum_organizations?limit={limit}"
    resp = requests.get(url, headers=_get_headers(api_key), timeout=30)
    resp.raise_for_status()
    js = resp.json()
    rows: List[dict] = []
    if isinstance(js, dict):
        data = js.get("data") or js.get("resources")
        if isinstance(data, list):
            rows = data
    elif isinstance(js, list):
        rows = js
    return pd.DataFrame(rows)


def fetch_all_organizations(api_key: Optional[str] = None) -> pd.DataFrame:
    """Fetch the full list of DAOs and their metadata.

    The `/organizations` endpoint returns a JSON object with a `resources`
    list.  Each resource contains fields such as `organizationId`, `name`,
    `description`, and `tokens`.  This function retrieves all entries and
    constructs a DataFrame.  If the request fails, an empty DataFrame is
    returned.

    Parameters
    ----------
    api_key : str, optional
        DeepDAO API key.

    Returns
    -------
    pandas.DataFrame
        DataFrame of organisation metadata, including a `tokens` column
        (list) when available.
    """
    url = f"{BASE_URL}/organizations"
    try:
        resp = requests.get(url, headers=_get_headers(api_key), timeout=60)
    except Exception:
        return pd.DataFrame()
    if resp.status_code != 200:
        return pd.DataFrame()
    js = resp.json()
    resources = None
    if isinstance(js, dict):
        data = js.get("data")
        if isinstance(data, dict):
            resources = data.get("resources")
    if not isinstance(resources, list):
        return pd.DataFrame()
    return pd.DataFrame(resources)


def fetch_timeseries(org_id: str, metric: str, start_date: str, end_date: str,
                     api_key: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Download a daily governance timeseries from DeepDAO.

    Parameters
    ----------
    org_id : str
        Organisation identifier.
    metric : str
        One of ``"proposals"`` or ``"votes"``.  The API endpoint for
        ``votes`` is ``daily_dao_votes/{org_id}``; for ``proposals`` it is
        ``daily_dao_proposals/{org_id}``.
    start_date, end_date : str
        Inclusive date range in ``YYYY-MM-DD`` format.
    api_key : str, optional
        DeepDAO API key.

    Returns
    -------
    pandas.DataFrame or None
        DataFrame indexed by date with a single column named after
        ``metric``, or ``None`` if the request fails.
    """
    # Build endpoint; note that we do not support the "voters" metric
    if metric == "voters":
        endpoint = f"daily_dao_voters/{org_id}"
    else:
        endpoint = f"daily_dao_{metric}/{org_id}"
    url = f"{BASE_URL}/timeseries/{endpoint}?startDate={start_date}&endDate={end_date}"
    try:
        r = requests.get(url, headers=_get_headers(api_key), timeout=30)
    except Exception:
        return None
    if r.status_code != 200:
        return None
    js = r.json()
    records = None
    if isinstance(js, dict):
        records = js.get("data") or js.get("resources")
    elif isinstance(js, list):
        records = js
    if not records:
        return None
    df = pd.DataFrame(records)
    if df.empty or 'date' not in df.columns or 'counter' not in df.columns:
        return None
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df = df.rename(columns={'counter': metric})
    return df[[metric]]


def _aggregate_to_monthly(series: pd.Series, start_label: str, end_label: str) -> pd.Series:
    """Aggregate a daily series into monthly sums labelled by the next month.

    The input series must have a ``DateTimeIndex``.  The function
    performs the following steps:

    1. Resample by calendar month (month end) and sum values.
    2. Shift the resulting index to the first day of the following month.
    3. Reindex to include **every** monthly label between ``start_label``
       and ``end_label``, filling missing months with zeros.

    Parameters
    ----------
    series : pandas.Series
        Daily values to aggregate.
    start_label, end_label : str
        First and last labels (inclusive) for the monthly index.

    Returns
    -------
    pandas.Series
        Monthly summed series indexed by month starts.
    """
    # Sum by month end
    # Use 'ME' (month end) rather than 'M' to avoid FutureWarning
    monthly_end = series.resample("ME").sum()
    # Shift to next month start
    monthly_start = monthly_end.copy()
    monthly_start.index = monthly_start.index + pd.offsets.MonthBegin(1)
    # Build full monthly index
    full_index = pd.date_range(start=start_label, end=end_label, freq="MS")
    monthly_start = monthly_start.reindex(full_index, fill_value=0)
    monthly_start.index.name = "date"
    return monthly_start


def process_organization(org_id: str, start_label: str, end_label: str,
                         api_key: str, verbose: bool = True) -> Optional[pd.Series]:
    """Compute the participation z‑score series for a single DAO.

    The function downloads daily proposal and vote counts, aggregates
    each into monthly sums, computes the votes‑per‑proposal ratio,
    and converts it into a z‑score based on the entire sample.  If
    missing data or zero variance are encountered, ``None`` is
    returned.

    Parameters
    ----------
    org_id : str
        DAO identifier.
    start_label, end_label : str
        Monthly index boundaries (inclusive).  The aggregation period
        effectively spans the calendar months prior to these labels.
    api_key : str
        DeepDAO API key.
    verbose : bool
        Whether to print progress messages.

    Returns
    -------
    pandas.Series or None
        Z‑score series indexed by monthly labels, or ``None`` if data
        are unavailable.
    """
    proposals = fetch_timeseries(org_id, "proposals", start_label, end_label, api_key)
    votes     = fetch_timeseries(org_id, "votes",     start_label, end_label, api_key)
    if proposals is None or votes is None:
        return None
    # Aggregate daily data into monthly sums
    prop_month  = _aggregate_to_monthly(proposals.iloc[:, 0], start_label, end_label)
    votes_month = _aggregate_to_monthly(votes.iloc[:, 0],     start_label, end_label)
    # Compute participation ratio; treat zero proposals as NaN to avoid division by zero
    ratio = votes_month.divide(prop_month.replace(0, np.nan))
    ratio = ratio.fillna(0.0)
    # Standardise to z‑scores
    mu    = ratio.mean()
    sigma = ratio.std(ddof=0)
    if sigma == 0:
        return None
    z_scores = (ratio - mu) / sigma
    return z_scores


def merge_price_csvs(directory: str | Path, *, verbose: bool = True) -> pd.DataFrame:
    """Merge multiple price CSV files into a single DataFrame.

    Each CSV file must contain a date column and a price column.  The
    DAO title is derived from the filename (text before ``" Price"``).
    The resulting DataFrame has one row per date and one column per
    DAO, with dates sorted ascending.  Missing prices remain NaN.

    Parameters
    ----------
    directory : str or Path
        Folder containing price CSV files.
    verbose : bool
        Whether to print status messages.

    Returns
    -------
    pandas.DataFrame
        Merged price table with a ``date`` column and one column per DAO.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Price directory not found: {directory}")
    csv_files = [f for f in directory.iterdir() if f.suffix.lower() == ".csv"]
    merged_df: Optional[pd.DataFrame] = None
    for fpath in sorted(csv_files):
        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            if verbose:
                print(f"Skipping {fpath.name}: failed to read ({e})")
            continue
        if df.shape[1] < 2:
            if verbose:
                print(f"Skipping {fpath.name}: not enough columns")
            continue
        sub = df.iloc[:, :2].copy()
        sub.columns = ["Date", "Price"]
        sub["date"] = pd.to_datetime(sub["Date"], errors="coerce")
        sub = sub.drop(columns=["Date"])
        dao_name = fpath.stem.split(" Price", 1)[0]
        sub = sub.rename(columns={"Price": dao_name})
        if verbose:
            print(f"Loaded {dao_name} from {fpath.name} ({len(sub)} rows)")
        if merged_df is None:
            merged_df = sub
        else:
            merged_df = pd.merge(merged_df, sub, on="date", how="outer")
    if merged_df is None:
        raise ValueError(f"No valid price files in {directory}")
    merged_df = merged_df.sort_values(by="date").reset_index(drop=True)
    return merged_df


def _cmc_headers(cmc_key: Optional[str] = None) -> Dict[str, str]:
    """Build headers for the CoinMarketCap API."""
    key = cmc_key or os.getenv("CMC_API_KEY")
    headers = {"accepts": "application/json"}
    if key:
        headers["X-CMC_PRO_API_KEY"] = key
    return headers


def fetch_cmc_historical(symbol: str, start_date: str, end_date: str,
                          cmc_key: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Fetch daily closing prices from CoinMarketCap.

    Parameters
    ----------
    symbol : str
        Token symbol recognised by CMC (e.g. ``UNI``).
    start_date, end_date : str
        Inclusive date range in ``YYYY-MM-DD`` format.
    cmc_key : str, optional
        CoinMarketCap API key.

    Returns
    -------
    pandas.DataFrame or None
        DataFrame with columns ``date`` and ``<symbol>`` or ``None`` if
        the request fails.
    """
    url = "https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/historical"
    params = {
        "symbol": symbol,
        "time_start": start_date,
        "time_end": end_date,
        "interval": "daily",
        "convert": "USD",
    }
    try:
        r = requests.get(url, headers=_cmc_headers(cmc_key), params=params, timeout=30)
    except Exception:
        return None
    if r.status_code != 200:
        return None
    js = r.json()
    data = js.get("data", {}).get(symbol)
    if not data or not isinstance(data.get("quotes"), list):
        return None
    records = data["quotes"]
    dates, prices = [], []
    for rec in records:
        timestamp = rec.get("timestamp")
        quote = rec.get("quote", {}).get("USD", {})
        price = quote.get("close")
        if timestamp and price is not None:
            dates.append(pd.to_datetime(timestamp).date())
            prices.append(price)
    if not dates:
        return None
    df = pd.DataFrame({"date": dates, symbol: prices})
    return df


def run_analysis(api_key: str,
                 cmc_key: Optional[str] = None,
                 price_directory: Optional[str] = None,
                 start_label: str = "2021-12-01",
                 end_label: str = "2025-01-01",
                 max_daos: int = 80,
                 top_limit: int = 120,
                 verbose: bool = True) -> None:
    """Execute the governance factor pipeline for the project sample.

    Parameters
    ----------
    api_key : str
        DeepDAO API key (mandatory).
    cmc_key : str, optional
        CoinMarketCap API key for fetching missing price data.
    price_directory : str, optional
        Directory containing manually downloaded price CSV files.  If
        provided, price histories are loaded from this directory.
    start_label, end_label : str
        Start and end labels for monthly aggregation (YYYY‑MM‑DD).  The
        aggregation covers the period from one month prior to
        ``start_label`` through the month preceding ``end_label``.
    max_daos : int
        Maximum number of DAOs to process (stopping early once this
        number of valid DAOs is reached).
    top_limit : int
        Number of top DAOs by AUM to request from DeepDAO.  The first
        ``max_daos`` DAOs with valid data are processed.
    verbose : bool
        Whether to print progress messages.

    Returns
    -------
    None
        The function saves CSV files to disk; it does not return a
        value.
    """
    # Step 1: verify API connectivity
    if verbose:
        print("Testing DeepDAO API connection...")
    if not test_connection(api_key):
        raise RuntimeError("DeepDAO API connection failed; check your API key")
    # Step 2: fetch top DAOs
    if verbose:
        print(f"Fetching top {top_limit} DAOs by AUM...")
    top_df = fetch_top_organizations(limit=top_limit, api_key=api_key)
    # Identify column names for id and title, as API fields may vary
    id_col = next((c for c in ["id", "organizationId", "organization_id"] if c in top_df.columns), None)
    title_col = next((c for c in ["title", "name"] if c in top_df.columns), None)
    symbol_col = next((c for c in ["symbol", "tokenSymbol", "token_symbol"] if c in top_df.columns), None)
    treasury_col = next((c for c in ["treasuryUsd", "treasuryUSD", "treasury_usd"] if c in top_df.columns), None)
    # Filter out DAOs without a tradable token.  Fetch the global list
    if verbose:
        print("Filtering DAOs without a traded token...")
    orgs_df = fetch_all_organizations(api_key)
    token_ids: set[str] = set()
    if not orgs_df.empty and "tokens" in orgs_df.columns:
        valid_mask = orgs_df["tokens"].apply(lambda x: isinstance(x, list) and len(x) > 0)
        token_ids = set(orgs_df.loc[valid_mask, "organizationId"].astype(str))
    if token_ids and id_col:
        top_df = top_df[top_df[id_col].astype(str).isin(token_ids)].reset_index(drop=True)
        if verbose:
            print(f"After token filter: {len(top_df)} DAOs remain")
    # Save top list for manual downloading.  We deliberately omit symbol
    # and treasury columns because they are often missing and cause
    # KeyError exceptions.  Only id and title/name are kept when
    # available; otherwise the full DataFrame is saved.
    meta_candidates = [id_col, title_col]
    meta_cols = [c for c in meta_candidates if c and c in top_df.columns]
    if not meta_cols:
        top_df.to_csv("top_120_daos.csv", index=False)
    else:
        top_df[meta_cols].to_csv("top_120_daos.csv", index=False)
    if verbose:
        print(f"Saved top_120_daos.csv with {len(top_df)} entries")
    # Step 3: process each DAO
    z_frames: Dict[str, pd.Series] = {}
    processed = 0
    for idx, row in top_df.iterrows():
        if processed >= max_daos:
            break
        # Determine organisation ID and title using detected columns
        raw_id = None
        if id_col and id_col in row:
            raw_id = row[id_col]
        else:
            # Fallback to common names
            raw_id = row.get("id") or row.get("organizationId") or row.get("organization_id")
        org_id = str(raw_id) if raw_id is not None else ""
        title_val = None
        if title_col and title_col in row:
            title_val = row[title_col]
        else:
            title_val = row.get("title") or row.get("name")
        title = title_val if title_val is not None else ""
        if verbose:
            print(f"[{processed+1}/{max_daos}] Processing {title} (ID {org_id})...")
        series = process_organization(org_id, start_label, end_label, api_key, verbose=verbose)
        if series is None:
            if verbose:
                print("  -> skipped: missing data or zero variance")
            continue
        z_frames[title] = series
        processed += 1
    if not z_frames:
        raise RuntimeError("No DAOs returned valid governance data")
    # Step 4: build master z‑score table
    master = pd.DataFrame(z_frames)
    master.index.name = "date"
    master.to_csv("master_zscores.csv")
    if verbose:
        print(f"Saved master_zscores.csv with shape {master.shape}")
    # Step 5: price merging is **not performed** in this script.  To
    # combine daily price files into a single table, use the separate
    # merge_csv_prices.py utility after you have downloaded the price
    # CSVs for the DAOs in `top_120_daos.csv`.  This script only
    # computes and saves the monthly z‑score panel.
    if verbose:
        print("Analysis complete. Z‑scores computed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute governance z‑scores for top DAOs (mini project)")
    parser.add_argument("--api-key", required=False, help="DeepDAO API key (can also be set via DEEPDAO_API_KEY)")
    parser.add_argument("--cmc-key", required=False, help="CoinMarketCap API key (optional)")
    parser.add_argument("--price-dir", required=False, help="Directory containing price CSV files")
    parser.add_argument("--start", default="2021-12-01", help="Start label for monthly aggregation (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-01-01", help="End label for monthly aggregation (YYYY-MM-DD)")
    parser.add_argument("--max-daos", type=int, default=80, help="Maximum number of DAOs to process")
    parser.add_argument("--top-limit", type=int, default=120, help="Number of top DAOs to request from DeepDAO")
    parser.add_argument("--no-verbose", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()
    run_analysis(api_key=args.api_key or os.getenv("DEEPDAO_API_KEY"),
                 cmc_key=args.cmc_key or os.getenv("CMC_API_KEY"),
                 price_directory=args.price_dir,
                 start_label=args.start,
                 end_label=args.end,
                 max_daos=args.max_daos,
                 top_limit=args.top_limit,
                 verbose=not args.no_verbose)


if __name__ == "__main__":
    main()