"""
analyse_portfolio_sample.py
==========================

This module automates the process of selecting a subset of DAOs
with sufficient pricing data over a specified range, computing new
governance z‑scores for that range, constructing factor-sorted
portfolios, and running simple regressions to test whether
governance participation predicts future returns.

The script is intended for exploratory analysis when only a
fraction of DAO tokens have adequate historical price coverage.
For example, if only 40 or 60 DAOs have daily prices from
2023-01-01 through 2025-06-01, the script will select those, rank
them monthly by governance z‑score, form long/short portfolios, and
estimate the relationship between governance participation and
portfolio returns.

Usage within a Jupyter notebook::

    from analyse_portfolio_sample import run_portfolio_analysis
    run_portfolio_analysis(
        zscore_file="master_zscores.csv",
        price_file="merged_prices.csv",
        start_date="2023-01-01",
        end_date="2025-06-01",
        sample_sizes=[20, 40, 60],
        deepdao_key=YOUR_KEY,
    )

Parameters
----------
zscore_file : str
    Path to the CSV containing monthly governance z‑scores.  The
    file must be in the format ``date`` | DAO1 | DAO2 | ... as
    produced by ``analysis_80daos_updated.py`` when run over the
    desired sample window.
price_file : str
    Path to the merged daily price CSV.  This should be the
    output of ``merge_csv_prices.py`` or ``fill_missing_prices_cmc.py``.
start_date, end_date : str
    Date boundaries (YYYY-MM-DD) defining the analysis window.
sample_sizes : list of int
    List of desired sample sizes.  The script attempts to form
    portfolios from the largest size down; if insufficient DAOs
    have complete data, it falls back to smaller sizes.
deepdao_key : str
    DeepDAO API key to recompute z‑scores if ``zscore_file`` does
    not cover the specified range.

Outputs
-------
For each sample size achieved, the script prints:
* The number of DAOs with complete data.
* The regression summary for a simple predictive model
  ``portfolio_return_t ~ z_score_mean_{t}``.
* A DataFrame of monthly portfolio returns.

It also returns a dictionary mapping sample size to the
corresponding results for further inspection in notebooks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

# We'll reuse the governance processing functions from analysis
# if available.  Import conditionally to avoid mandatory
# dependency.
try:
    from analysis_80daos_updated import process_organization, fetch_top_organizations, fetch_all_organizations
except Exception:
    process_organization = None  # type: ignore
    fetch_top_organizations = None  # type: ignore
    fetch_all_organizations = None  # type: ignore


def _recompute_zscores(top_df: pd.DataFrame, start_date: str, end_date: str,
                       deepdao_key: str, max_daos: int) -> pd.DataFrame:
    """Recompute z‑scores for a given date range.

    This helper calls the same logic used in the analysis script to
    aggregate governance metrics and compute z‑scores over the
    specified period.  Only the first ``max_daos`` DAOs from
    ``top_df`` are processed.
    """
    if process_organization is None:
        raise RuntimeError("analysis_80daos_updated.process_organization is not available")
    z_frames: Dict[str, pd.Series] = {}
    processed = 0
    for _, row in top_df.iterrows():
        if processed >= max_daos:
            break
        org_id = row.get("id") or row.get("organizationId") or row.get("organization_id")
        title = row.get("title") or row.get("name") or ""
        series = process_organization(org_id, start_date, end_date, deepdao_key, verbose=False)
        if series is None:
            continue
        z_frames[title] = series
        processed += 1
    if not z_frames:
        raise RuntimeError("No DAOs returned valid z‑scores for the specified range")
    master = pd.DataFrame(z_frames)
    master.index.name = "date"
    return master.reset_index()


def _ensure_zscore_range(zscore_file: str, start_date: str, end_date: str,
                         deepdao_key: str, max_daos: int) -> pd.DataFrame:
    """Load or recompute z‑scores to cover a desired date range.

    If the existing z‑score file does not span the required range,
    this function recomputes z‑scores over the new period using
    DeepDAO.  Otherwise, it simply loads the file.  The caller
    should ensure that the z‑score file contains monthly data for
    the same DAOs as those in the price table; missing columns
    will be handled later.
    """
    z_df = pd.read_csv(zscore_file, parse_dates=["date"])
    # Check coverage
    if z_df["date"].min() > pd.to_datetime(start_date) or z_df["date"].max() < pd.to_datetime(end_date):
        # Need to recompute
        if fetch_top_organizations is None or fetch_all_organizations is None:
            raise RuntimeError(
                "Cannot recompute z‑scores: DeepDAO helpers unavailable. "
                "Ensure you run this script in the same environment as analysis_80daos_updated.py"
            )
        # Use top DAOs from DeepDAO, filtered by token availability.
        top_df = fetch_top_organizations(limit=max_daos, api_key=deepdao_key)
        # Filter out DAOs without tokens
        orgs_df = fetch_all_organizations(api_key=deepdao_key)
        if "tokens" in orgs_df.columns:
            valid_mask = orgs_df["tokens"].apply(lambda x: isinstance(x, list) and len(x) > 0)
            valid_ids = set(orgs_df.loc[valid_mask, "organizationId"].astype(str))
            top_df = top_df[top_df["id"].astype(str).isin(valid_ids)].reset_index(drop=True)
        z_df = _recompute_zscores(top_df, start_date, end_date, deepdao_key, max_daos)
    return z_df


def run_portfolio_analysis(zscore_file: str,
                           price_file: str,
                           start_date: str,
                           end_date: str,
                           sample_sizes: Iterable[int],
                           deepdao_key: str,
                           *,
                           verbose: bool = True) -> Dict[int, Dict[str, pd.DataFrame]]:
    """Perform portfolio analysis for multiple sample sizes.

    Parameters
    ----------
    zscore_file : str
        Path to CSV with monthly z‑scores.
    price_file : str
        Path to CSV with daily prices.
    start_date, end_date : str
        Analysis date window.
    sample_sizes : iterable of int
        Desired numbers of DAOs to include in the portfolio.  The
        script will attempt to form the largest sample first and
        fall back to smaller sizes if necessary.
    deepdao_key : str
        DeepDAO API key for recomputing z‑scores if the given file
        does not cover the desired range.
    verbose : bool, default True
        Whether to print progress messages.

    Returns
    -------
    dict
        Mapping from sample size to a dictionary containing:
        ``portfolio_returns`` (DataFrame) and ``regression`` (result).
    """
    # Load or recompute z‑scores
    z_df = _ensure_zscore_range(zscore_file, start_date, end_date, deepdao_key, max(sample_sizes))
    z_df = z_df[(z_df["date"] >= pd.to_datetime(start_date)) & (z_df["date"] <= pd.to_datetime(end_date))].copy()
    z_df = z_df.set_index("date")
    # Load daily price data
    p_df = pd.read_csv(price_file)
    # Explicitly convert the 'date' column to datetime; coerce errors
    if "date" not in p_df.columns:
        raise ValueError("The price file must contain a 'date' column")
    p_df["date"] = pd.to_datetime(p_df["date"], errors="coerce")
    # Drop rows where date could not be parsed
    p_df = p_df.dropna(subset=["date"])
    # Filter by date range
    mask = (p_df["date"] >= pd.to_datetime(start_date)) & (p_df["date"] <= pd.to_datetime(end_date))
    p_df = p_df.loc[mask].copy()
    p_df = p_df.set_index("date")
    # Compute daily log returns
    daily_returns = np.log(p_df / p_df.shift(1))
    daily_returns = daily_returns.dropna(how="all")
    # Resample to month-end and label by the first day of next month
    monthly_returns = daily_returns.resample("M").sum()
    monthly_returns.index = monthly_returns.index + pd.offsets.MonthBegin(1)
    # Align indices
    common_dates = z_df.index.intersection(monthly_returns.index)
    z_df = z_df.reindex(common_dates)
    monthly_returns = monthly_returns.reindex(common_dates)
    results: Dict[int, Dict[str, pd.DataFrame]] = {}
    # Determine which DAOs have complete price data across the window
    complete_daos = [c for c in monthly_returns.columns if not monthly_returns[c].isna().any()]
    if verbose:
        print(f"Total DAOs with complete monthly price data: {len(complete_daos)}")
    # Use DAOs present in z-score file as well
    complete_daos = [dao for dao in complete_daos if dao in z_df.columns]
    complete_daos.sort()
    for size in sorted(sample_sizes, reverse=True):
        if len(complete_daos) < size:
            if verbose:
                print(f"Not enough DAOs ({len(complete_daos)}) for sample size {size}")
            continue
        sample_daos = complete_daos[:size]
        if verbose:
            print(f"Constructing portfolio for sample size {size} with {len(sample_daos)} DAOs")
        port_returns_list: List[Dict[str, float]] = []
        for dt in common_dates:
            z_scores = z_df.loc[dt, sample_daos]
            rets     = monthly_returns.loc[dt, sample_daos]
            # Skip if any z-score NaN
            if z_scores.isna().any():
                continue
            # Rank by z-score descending (higher participation = better)
            ranks = z_scores.rank(ascending=False, method="first")
            n = len(sample_daos)
            quint = n // 5
            # Determine top and bottom quintiles
            top_mask    = ranks <= quint
            bottom_mask = ranks > n - quint
            # Compute weights
            if top_mask.sum() == 0 or bottom_mask.sum() == 0:
                continue
            w_top    = 1.0 / top_mask.sum()
            w_bottom = -1.0 / bottom_mask.sum()
            portfolio_ret = (rets[top_mask] * w_top).sum() + (rets[bottom_mask] * w_bottom).sum()
            port_returns_list.append({
                "date": dt,
                "return": portfolio_ret,
                "z_score_mean": z_scores.mean()
            })
        if not port_returns_list:
            if verbose:
                print(f"No portfolio returns computed for sample size {size}")
            continue
        port_df = pd.DataFrame(port_returns_list)
        port_df = port_df.set_index("date").sort_index()
        # Predictive regression: portfolio return_t = alpha + beta * z_mean_{t-1}
        port_df["z_lag"] = port_df["z_score_mean"].shift(1)
        port_df = port_df.dropna()
        if len(port_df) < 3:
            if verbose:
                print(f"Not enough observations for regression (n={len(port_df)})")
            continue
        X = sm.add_constant(port_df["z_lag"])
        y = port_df["return"]
        model = sm.OLS(y, X).fit()
        if verbose:
            print(model.summary())
        results[size] = {
            "portfolio_returns": port_df,
            "regression": model,
        }
    return results
