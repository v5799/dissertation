"""
analysis_80daos_cmc_updated.py
==============================

This variant of the mini project workflow uses the CoinMarketCap
(CMC) API exclusively for price data.  It computes governance
participation z‑scores for the top decentralised autonomous
organisations (DAOs) and then retrieves daily token prices from CMC
to build a monthly price panel.  The structure of the script
parallels that of ``analysis_80daos_updated.py`` but omits any
dependency on local CSV files.  Use this version when you do not
have local price data and are willing to consume your CMC API
quota.

* **Governance data:** Identical to the mini project version; uses
  the DeepDAO API to fetch daily proposals and votes, aggregates to
  monthly, and computes z‑scores.
* **CMC integration:** For each DAO processed, the script maps its
  title to its token symbol (as returned by the top AUM endpoint)
  and queries the CMC historical quotes endpoint.  It then
  resamples daily prices to month end, labels them by the first day
  of the next month, and aligns them with the z‑score index.
* **Outputs:** ``master_zscores.csv`` contains the governance panel,
  and ``master_zscores_with_price.csv`` contains the merged
  governance–price panel.  A list of DAOs for which CMC data could
  not be retrieved is written to ``missing_price_files.txt``.

Note: CMC imposes a quota (10 000 calls per month in the user's
demo tier).  Downloading prices for many DAOs may exhaust this
quota; consider throttling requests or selecting a smaller subset of
DAOs.
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

# DeepDAO helpers (duplicated for independence)
BASE_URL = "https://api.deepdao.io/v0.1"


def _get_headers(api_key: Optional[str] = None) -> Dict[str, str]:
    key = api_key or os.getenv("DEEPDAO_API_KEY")
    headers = {"accept": "application/json"}
    if key:
        headers["x-api-key"] = key
    return headers


def test_connection(api_key: Optional[str] = None) -> bool:
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
    """Fetch all DAOs and their token information.

    The `/organizations` endpoint provides a list of DAOs, including
    a `tokens` field.  This helper retrieves all organisations and
    returns a DataFrame.  If the request fails, an empty DataFrame
    is returned.
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
    monthly_end = series.resample("ME").sum()
    monthly_start = monthly_end.copy()
    monthly_start.index = monthly_start.index + pd.offsets.MonthBegin(1)
    full_index = pd.date_range(start=start_label, end=end_label, freq="MS")
    monthly_start = monthly_start.reindex(full_index, fill_value=0)
    monthly_start.index.name = "date"
    return monthly_start


def process_organization(org_id: str, start_label: str, end_label: str,
                         api_key: str, verbose: bool = True) -> Optional[pd.Series]:
    proposals = fetch_timeseries(org_id, "proposals", start_label, end_label, api_key)
    votes     = fetch_timeseries(org_id, "votes",     start_label, end_label, api_key)
    if proposals is None or votes is None:
        return None
    prop_month  = _aggregate_to_monthly(proposals.iloc[:, 0], start_label, end_label)
    votes_month = _aggregate_to_monthly(votes.iloc[:, 0],     start_label, end_label)
    ratio = votes_month.divide(prop_month.replace(0, np.nan))
    ratio = ratio.fillna(0.0)
    mu    = ratio.mean()
    sigma = ratio.std(ddof=0)
    if sigma == 0:
        return None
    z_scores = (ratio - mu) / sigma
    return z_scores


# CMC helper
def _cmc_headers(cmc_key: Optional[str] = None) -> Dict[str, str]:
    key = cmc_key or os.getenv("CMC_API_KEY")
    headers = {"accepts": "application/json"}
    if key:
        headers["X-CMC_PRO_API_KEY"] = key
    return headers


def fetch_cmc_historical(symbol: str, start_date: str, end_date: str,
                          cmc_key: Optional[str] = None) -> Optional[pd.DataFrame]:
    url = "https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/historical"
    params = {
        "symbol": symbol,
        "time_start": start_date,
        "time_end": end_date,
        "interval": "daily",
        "convert": "USD",
    }
    headers = _cmc_headers(cmc_key)
    headers.setdefault("Accept", "application/json")
    headers.setdefault("Accept-Encoding", "deflate, gzip")
    try:
        r = requests.get(url, headers=headers, params=params, timeout=30)
    except Exception:
        return None
    if r.status_code == 429:
        # Rate limit hit; skip gracefully
        return None
    if r.status_code != 200:
        return None
    try:
        js = r.json()
    except Exception:
        return None
    status = js.get("status")
    if isinstance(status, dict) and status.get("error_code", 0) != 0:
        return None
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
            dates.append(pd.to_datetime(timestamp).normalize())
            prices.append(price)
    if not dates:
        return None
    return pd.DataFrame({"date": dates, symbol: prices})


def run_analysis(api_key: str,
                 cmc_key: str,
                 start_label: str = "2021-12-01",
                 end_label: str = "2025-01-01",
                 max_daos: int = 80,
                 top_limit: int = 120,
                 verbose: bool = True) -> None:
    if verbose:
        print("Testing DeepDAO API connection...")
    if not test_connection(api_key):
        raise RuntimeError("DeepDAO API connection failed; check your API key")
    if verbose:
        print(f"Fetching top {top_limit} DAOs by AUM...")
    top_df = fetch_top_organizations(limit=top_limit, api_key=api_key)
    # Filter out DAOs without a tradable token
    if verbose:
        print("Filtering DAOs without a traded token...")
    orgs_df = fetch_all_organizations(api_key)
    token_ids: set[str] = set()
    if not orgs_df.empty and "tokens" in orgs_df.columns:
        valid_mask = orgs_df["tokens"].apply(lambda x: isinstance(x, list) and len(x) > 0)
        token_ids = set(orgs_df.loc[valid_mask, "organizationId"].astype(str))
    if token_ids:
        top_df = top_df[top_df["id"].astype(str).isin(token_ids)].reset_index(drop=True)
        if verbose:
            print(f"After token filter: {len(top_df)} DAOs remain")
    # Save metadata for user reference.  Only include id and title/name
    # to avoid errors from missing symbol or treasury fields.
    meta_candidates = [id_col, title_col]
    existing_cols = [c for c in meta_candidates if c and c in top_df.columns]
    if not existing_cols:
        top_df.to_csv("top_120_daos.csv", index=False)
    else:
        top_df[existing_cols].to_csv("top_120_daos.csv", index=False)
    if verbose:
        print(f"Saved top_120_daos.csv with {len(top_df)} entries")
    z_frames: Dict[str, pd.Series] = {}
    processed = 0
    for idx, row in top_df.iterrows():
        if processed >= max_daos:
            break
        raw_id = None
        if id_col and id_col in row:
            raw_id = row[id_col]
        else:
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
    master = pd.DataFrame(z_frames)
    master.index.name = "date"
    master.to_csv("master_zscores.csv")
    if verbose:
        print(f"Saved master_zscores.csv with shape {master.shape}")
    # Fetch prices from CMC
    missing_titles: List[str] = []
    # Build mapping from title/name to symbol for CMC lookups
    name_field = title_col if (title_col and title_col in top_df.columns) else ("title" if "title" in top_df.columns else "name")
    sym_field  = symbol_col if (symbol_col and symbol_col in top_df.columns) else ("symbol" if "symbol" in top_df.columns else None)
    if sym_field:
        id_to_symbol = top_df.set_index(name_field)[sym_field].to_dict()
    else:
        id_to_symbol = {}
    price_df: Optional[pd.DataFrame] = None
    for dao in master.columns:
        symbol = id_to_symbol.get(dao)
        if not symbol:
            missing_titles.append(dao)
            continue
        if verbose:
            print(f"Fetching CMC prices for {dao} ({symbol})...")
        cmc_df = fetch_cmc_historical(symbol, start_label, end_label, cmc_key)
        if cmc_df is None:
            missing_titles.append(dao)
            if verbose:
                print(f"  -> failed to fetch CMC data for {dao}")
            continue
        cmc_df = cmc_df.set_index("date")
        monthly = cmc_df.resample("M").last()
        monthly.index = monthly.index + pd.offsets.MonthBegin(1)
        monthly = monthly.reindex(master.index, method="pad")
        if price_df is None:
            price_df = monthly
        else:
            price_df = price_df.join(monthly, how="outer")
        time.sleep(0.5)
    # Ensure price_df exists; if None, create empty frame
    if price_df is None:
        price_df = pd.DataFrame(index=master.index)
    # Add placeholder columns for DAOs with missing prices
    for missing in missing_titles:
        price_df[missing] = np.nan
    # Merge z‑scores and prices
    merged = master.join(price_df, how="left")
    merged.to_csv("master_zscores_with_price.csv")
    if verbose:
        print(f"Saved master_zscores_with_price.csv with shape {merged.shape}")
    # Write missing list
    if missing_titles:
        with open("missing_price_files.txt", "w", encoding="utf-8") as fout:
            for name in missing_titles:
                fout.write(name + "\n")
        if verbose:
            print(f"Missing price files recorded in missing_price_files.txt ({len(missing_titles)} entries)")
    if verbose:
        print("Analysis complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute z‑scores and fetch prices via CMC (mini project)")
    parser.add_argument("--api-key", required=False, help="DeepDAO API key (can also be set via DEEPDAO_API_KEY)")
    parser.add_argument("--cmc-key", required=False, help="CoinMarketCap API key (required)")
    parser.add_argument("--start", default="2021-12-01", help="Start label for monthly aggregation")
    parser.add_argument("--end", default="2025-01-01", help="End label for monthly aggregation")
    parser.add_argument("--max-daos", type=int, default=80, help="Maximum number of DAOs to process")
    parser.add_argument("--top-limit", type=int, default=120, help="Number of top DAOs to request from DeepDAO")
    parser.add_argument("--no-verbose", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()
    run_analysis(api_key=args.api_key or os.getenv("DEEPDAO_API_KEY"),
                 cmc_key=args.cmc_key or os.getenv("CMC_API_KEY"),
                 start_label=args.start,
                 end_label=args.end,
                 max_daos=args.max_daos,
                 top_limit=args.top_limit,
                 verbose=not args.no_verbose)


if __name__ == "__main__":
    main()