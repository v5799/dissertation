"""
analysis_max_daos_updated.py
============================

This module extends the mini‑project workflow to a broader
sample of decentralised autonomous organisations (DAOs) over a
longer historical window.  It retrieves governance activity for a
large number of DAOs (up to the API limit of 2000) from the
DeepDAO API, computes monthly participation z‑scores, and
optionally merges these with token price data.  The code mirrors
``analysis_80daos_updated.py`` but adjusts the default date
range and the number of organisations processed.

* **Date range:** By default, monthly data are aggregated from
  December 2017 through May 2025 (labelled 2018‑01‑01 through
  2025‑06‑01).  This extended horizon is intended for the user's
  larger thesis project.
* **DAO universe:** Up to 2000 top DAOs by AUM are requested.  All
  available organisations are processed unless ``max_daos`` is
  specified to a lower value.  This gives the user flexibility to
  experiment with different universe sizes.
* **Price merging and fallback:** As with the mini‑project script,
  local price CSVs are merged first.  Missing series can be filled
  using the CoinMarketCap API if an API key is supplied.  A
  ``missing_price_files.txt`` file is written for convenience.

Usage example::

    from analysis_max_daos_updated import run_analysis

    deepdao_key = "YOUR_DEEPDAO_API_KEY"
    cmc_key     = "YOUR_CMC_API_KEY"  # optional

    run_analysis(
        api_key=deepdao_key,
        cmc_key=cmc_key,
        price_directory=r"C:\\Users\\cheet\\Downloads\\DAO priciing data",
        start_label="2018-01-01",
        end_label="2025-06-01",
        max_daos=2000,
        top_limit=2000,
        verbose=True,
    )

When executed, the script will generate the same set of files as the
mini‑project version but for the full universe and date range.
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

# Reuse the helper functions from the mini project.  We copy them
# explicitly to maintain independence between modules and ensure that
# modifications in one file do not inadvertently affect the other.

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


def fetch_top_organizations(limit: int = 2000, api_key: Optional[str] = None) -> pd.DataFrame:
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
    """Fetch all organisations and their metadata, including token lists.

    This utility calls the `/organizations` endpoint, which returns a
    `resources` list of DAO metadata.  The returned DataFrame may include
    fields such as `organizationId`, `name`, `description`, and `tokens`.
    If the request fails, an empty DataFrame is returned.

    Parameters
    ----------
    api_key : str, optional
        DeepDAO API key.

    Returns
    -------
    pandas.DataFrame
        DataFrame of organisations.  Use the `tokens` column to
        identify whether a DAO has a traded token.
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
    monthly_end = series.resample("M").sum()
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


def merge_price_csvs(directory: str | Path, *, verbose: bool = True) -> pd.DataFrame:
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
                 start_label: str = "2018-01-01",
                 end_label: str = "2025-06-01",
                 max_daos: int = 2000,
                 top_limit: int = 2000,
                 verbose: bool = True) -> None:
    # Test API connection
    if verbose:
        print("Testing DeepDAO API connection...")
    if not test_connection(api_key):
        raise RuntimeError("DeepDAO API connection failed; check your API key")
    # Fetch top DAOs
    if verbose:
        print(f"Fetching top {top_limit} DAOs by AUM...")
    top_df = fetch_top_organizations(limit=top_limit, api_key=api_key)
    # Identify column names for id, title, symbol, and treasury
    id_col = next((c for c in ["id", "organizationId", "organization_id"] if c in top_df.columns), None)
    title_col = next((c for c in ["title", "name"] if c in top_df.columns), None)
    symbol_col = next((c for c in ["symbol", "tokenSymbol", "token_symbol"] if c in top_df.columns), None)
    treasury_col = next((c for c in ["treasuryUsd", "treasuryUSD", "treasury_usd"] if c in top_df.columns), None)
    # Filter by availability of a tradable token
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
    # Save metadata for reference.  Only include id and title/name to avoid
    # errors when symbol or treasury fields are missing.  If these
    # columns are absent, save the entire DataFrame.
    meta_candidates = [id_col, title_col]
    meta_cols = [c for c in meta_candidates if c and c in top_df.columns]
    if not meta_cols:
        top_df.to_csv("top_{}_daos.csv".format(top_limit), index=False)
    else:
        top_df[meta_cols].to_csv("top_{}_daos.csv".format(top_limit), index=False)
    if verbose:
        print(f"Saved top_{top_limit}_daos.csv with {len(top_df)} entries")
    # Process DAOs
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
    missing_titles: List[str] = []
    if price_directory:
        if verbose:
            print("Merging price CSV files...")
        price_df = merge_price_csvs(price_directory, verbose=verbose)
        price_df = price_df.set_index("date")
        price_df = price_df.reindex(master.index, method=None)
        missing_cols = [dao for dao in master.columns if dao not in price_df.columns]
        missing_titles = list(missing_cols)
        if missing_cols and cmc_key:
            id_to_symbol = top_df.set_index("title")["symbol"].to_dict()
            for dao in missing_cols:
                symbol = id_to_symbol.get(dao)
                if symbol:
                    if verbose:
                        print(f"Fetching CMC prices for {dao} ({symbol})...")
                    cmc_df = fetch_cmc_historical(symbol, start_label, end_label, cmc_key)
                    if cmc_df is not None:
                        cmc_df = cmc_df.set_index("date")
                        monthly = cmc_df.resample("M").last()
                        monthly.index = monthly.index + pd.offsets.MonthBegin(1)
                        monthly = monthly.reindex(master.index, method="pad")
                        price_df = price_df.join(monthly, how="left")
                    time.sleep(0.5)
        merged = master.join(price_df, how="left")
        merged.to_csv("master_zscores_with_price.csv")
        if verbose:
            print(f"Saved master_zscores_with_price.csv with shape {merged.shape}")
    if missing_titles:
        with open("missing_price_files.txt", "w", encoding="utf-8") as fout:
            for name in missing_titles:
                fout.write(name + "\n")
        if verbose:
            print(f"Missing price files recorded in missing_price_files.txt ({len(missing_titles)} entries)")
    if verbose:
        print("Analysis complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute governance z‑scores for a large set of DAOs")
    parser.add_argument("--api-key", required=False, help="DeepDAO API key (can also be set via DEEPDAO_API_KEY)")
    parser.add_argument("--cmc-key", required=False, help="CoinMarketCap API key (optional)")
    parser.add_argument("--price-dir", required=False, help="Directory containing price CSV files")
    parser.add_argument("--start", default="2018-01-01", help="Start label for monthly aggregation (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-06-01", help="End label for monthly aggregation (YYYY-MM-DD)")
    parser.add_argument("--max-daos", type=int, default=2000, help="Maximum number of DAOs to process")
    parser.add_argument("--top-limit", type=int, default=2000, help="Number of top DAOs to request from DeepDAO")
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