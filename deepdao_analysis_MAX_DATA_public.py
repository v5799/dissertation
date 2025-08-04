# This is a student comment:
# I am running this version to test z-scores using only data between 2018-01-01 and 2025-01-01 for the metric calculations.
# The main analysis period is from 2018-01-01 to 2025-01-01, but for the full-history z-scores I’ll use the full data starting from 2018.
# This block will help me understand how sensitive results are to the choice of normalisation window.

"""
DeepDAO Governance Analysis
===========================

This module provides a collection of functions for downloading and analysing
governance activity of decentralised autonomous organisations (DAOs) via
the DeepDAO API.  It has been refined from an earlier draft to better
reflect the research design described by the user.  In particular:

* The aggregation window spans 25 months, covering activity from
  **December 2022 through December 2024**, inclusive.  Activity in a given
  calendar month is rolled up into a single point labelled by the
  **first day of the following month**.  For example, the data point
  labelled ``2023-01-01`` represents all proposals and voter activity
  occurring in **December 2022**.  This ensures that every monthly data
  point corresponds to a complete prior month.
* “Dead” months (periods with zero proposals and/or zero voters) are
  preserved in the final dataset.  Removing them would constitute
  survivorship bias; instead we reindex the series to include every
  monthly label between the start and end of the study window and
  fill missing values with zeros.
* The module exposes helper functions for loading historical token
  pricing data from CSV files and merging those prices onto the master
  z‑score table.  Pricing files should follow a consistent naming
  convention (for example ``Uniswap Price 2025-08-03.csv``) and
  contain two columns: ``Date`` and a price column.  See the function
  :func:`load_price_files` for details.

The main workflow comprises the following steps:

1. Use :func:`fetch_top_organizations` to retrieve a ranked list of DAOs
   by assets under management (AUM).  The API key must be supplied via
   the environment variable ``DEEPDAO_API_KEY`` or directly passed when
   constructing the request headers.
2. For each of the top N organisations, call :func:`process_organization`
   to download daily proposal and voter counts, aggregate them into
   monthly ratios, and compute a z‑score series.  The function
   automatically reindexes the series to include dead months.
3. Combine the individual DAO data frames with :func:`save_master_zscores`.
   This function will stop once it has processed the first 80 DAOs
   (after filtering out any that return invalid or incomplete series).
4. (Optional) Load token price histories from a folder using
   :func:`load_price_files` and merge them with the z‑score table via
   :func:`merge_price_data`.  This allows you to examine relationships
   between governance activity and market performance.
5. (Optional) Export the master table or the list of top DAO titles
   for manual token lookup using :func:`get_top_n_daos`.

The functions defined herein are designed to be called from a Jupyter
notebook or another Python script.  See the ``if __name__ == '__main__'``
block at the bottom of the file for an example workflow.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# Base URL for DeepDAO's v0.1 API.  If a newer API becomes available
# after the knowledge cut-off (2024‑06), update this constant accordingly.
BASE_URL = "https://api.deepdao.io/v0.1"

def _get_headers(api_key: Optional[str] = None) -> Dict[str, str]:
    """Build request headers for DeepDAO API.

    The API key can either be provided explicitly or via the
    ``DEEPDAO_API_KEY`` environment variable.  If no key is available,
    requests will likely fail with an HTTP 401 error.

    Parameters
    ----------
    api_key: optional, str
        The API key to use.  If ``None``, the ``DEEPDAO_API_KEY``
        environment variable will be read.

    Returns
    -------
    dict
        Headers dictionary suitable for use with ``requests.get``.
    """
    key = api_key or os.getenv("DEEPDAO_API_KEY")
    headers = {"accept": "application/json"}
    # DeepDAO uses an "x-api-key" header.  Using any other key name
    # (e.g., "api-key") will result in authentication failure.
    if key:
        headers["x-api-key"] = key
    return headers


def fetch_top_organizations(limit: int = 2000,
                            api_key: Optional[str] = None) -> pd.DataFrame:
    """Retrieve a ranked list of DAOs by assets under management (AUM).

    This helper calls the ``/organizations/top_aum_organizations`` endpoint
    and gracefully parses the returned JSON.  Earlier versions of this
    function naively passed the entire dictionary returned by the API
    into ``pd.DataFrame``, which raised a ``ValueError`` about mixing
    dicts with non‑Series.  The DeepDAO endpoint always returns a
    dictionary with a ``data`` key containing a list of organisation
    records.  This implementation extracts that list prior to
    constructing the DataFrame.

    Parameters
    ----------
    limit: int, default 2000
        Maximum number of organisations to fetch.  The DeepDAO API
        currently supports up to 2000 organisations.
    api_key: optional, str
        Your DeepDAO API key.  If omitted, the ``DEEPDAO_API_KEY``
        environment variable will be used.

    Returns
    -------
    pandas.DataFrame
        A data frame containing the raw organisation records returned
        by the API.  Typical fields include ``organizationId``, ``title``,
        ``aumUsd``, ``symbol``, etc.
    """
    url = f"{BASE_URL}/organizations/top_aum_organizations?limit={limit}"
    response = requests.get(url, headers=_get_headers(api_key))
    response.raise_for_status()
    json_data = response.json()
    # Extract the list of records from the 'data' field.  If 'data' is not
    # present or is not a list, fall back to an empty list to avoid
    # DataFrame construction errors.
    rows = []
    if isinstance(json_data, dict):
        maybe = json_data.get("data")
        if isinstance(maybe, list):
            rows = maybe
        else:
            # Some endpoints return a nested structure; attempt to pull
            # from 'resources' within 'data'
            if isinstance(maybe, dict) and isinstance(maybe.get("resources"), list):
                rows = maybe.get("resources")
    elif isinstance(json_data, list):
        rows = json_data
    # Construct DataFrame from the list of organisations
    return pd.DataFrame(rows)


def test_connection(api_key: Optional[str] = None) -> bool:
    """Test whether the provided API key can authenticate with DeepDAO.

    This function performs a minimal GET request to the DeepDAO API and
    reports whether it succeeds.  It is useful for verifying that the
    ``x-api-key`` header is set correctly and that the key is valid
    before proceeding with more expensive calls.  A message will be
    printed indicating the result of the test.

    Parameters
    ----------
    api_key: optional, str
        Your DeepDAO API key.  If omitted, the ``DEEPDAO_API_KEY``
        environment variable will be used.

    Returns
    -------
    bool
        ``True`` if the connection test succeeds (HTTP 200), ``False``
        otherwise.
    """
    test_url = f"{BASE_URL}/organizations/top_aum_organizations?limit=1"
    try:
        resp = requests.get(test_url, headers=_get_headers(api_key), timeout=10)
        if resp.status_code == 200:
            print("DeepDAO API connection successful.")
            return True
        print(f"DeepDAO API returned status {resp.status_code}: {resp.text[:200]}")
        return False
    except Exception as e:
        print(f"Error while testing DeepDAO API connection: {e}")
        return False


def fetch_all_organizations(limit: Optional[int] = None,
                            api_key: Optional[str] = None) -> pd.DataFrame:
    """Retrieve the full list of organisations from DeepDAO.

    The generic ``/organizations`` endpoint returns a paginated list of
    DAOs with many descriptive fields, including the ``tokens`` list used
    to determine whether an organisation has a tradable token.  The
    response JSON has the structure ``{"data": {"totalResources": ..., "resources": [...]}}``.
    This helper pulls the ``resources`` list and converts it to a
    DataFrame.  If a ``limit`` is supplied, it will be passed as a
    query parameter; otherwise, the API returns a default number of
    records.

    Parameters
    ----------
    limit: optional, int
        Maximum number of organisations to fetch.  If ``None``, the
        request will omit the limit parameter and rely on the API's
        default.
    api_key: optional, str
        Your DeepDAO API key.  If omitted, the ``DEEPDAO_API_KEY``
        environment variable will be used.

    Returns
    -------
    pandas.DataFrame
        DataFrame of organisations with columns such as
        ``organizationId``, ``name``, ``tokens``, ``proposals``,
        ``members``, etc.
    """
    params = {}
    if limit is not None:
        params["limit"] = limit
    url = f"{BASE_URL}/organizations"
    resp = requests.get(url, headers=_get_headers(api_key), params=params)
    resp.raise_for_status()
    js = resp.json()
    resources = []
    if isinstance(js, dict):
        data = js.get("data")
        if isinstance(data, dict):
            resources = data.get("resources", []) or data.get("daos", [])
    return pd.DataFrame(resources)


def filter_daos_with_tokens(orgs_df: pd.DataFrame) -> pd.DataFrame:
    """Filter organisations that have at least one tradable token.

    This helper replicates the logic used in the historical notebook where
    a DAO was considered investable if its ``tokens`` field was a non
    empty list.  Organisations lacking a ``tokens`` column will be
    returned unchanged.

    Parameters
    ----------
    orgs_df: pandas.DataFrame
        DataFrame containing a ``tokens`` column, possibly among others.

    Returns
    -------
    pandas.DataFrame
        Subset of ``orgs_df`` for which ``tokens`` is a non‑empty list.
    """
    if orgs_df is None or orgs_df.empty or 'tokens' not in orgs_df.columns:
        return orgs_df
    mask = orgs_df['tokens'].apply(lambda x: isinstance(x, list) and len(x) > 0)
    return orgs_df.loc[mask].copy().reset_index(drop=True)


def merge_top_and_tokens(top_df: pd.DataFrame,
                         tokens_df: pd.DataFrame) -> pd.DataFrame:
    """Intersect the top AUM list with the organisations that have tokens.

    Parameters
    ----------
    top_df: pandas.DataFrame
        Ranked organisations returned by :func:`fetch_top_organizations`.
        Must contain columns ``organizationId`` (or ``id``) and ``title``.
    tokens_df: pandas.DataFrame
        Organisations with non‑empty ``tokens`` lists as returned by
        :func:`filter_daos_with_tokens`.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the intersection of the two inputs,
        preserving the order of ``top_df``.  Rows will have the union
        of columns from both inputs.
    """
    if top_df is None or top_df.empty:
        return pd.DataFrame()
    if tokens_df is None or tokens_df.empty:
        return pd.DataFrame()
    # Ensure we have consistent key names
    top_key = 'organizationId' if 'organizationId' in top_df.columns else 'id'
    # Filter tokens_df to only those present in top_df
    common_ids = set(top_df[top_key].astype(str)) & set(tokens_df['organizationId'].astype(str))
    subset = top_df[top_df[top_key].astype(str).isin(common_ids)].copy()
    # Set index for ordering and merge
    merged = tokens_df.merge(subset, left_on='organizationId', right_on=top_key, how='inner')
    # Preserve ranking order
    valid_order = [oid for oid in subset[top_key].astype(str) if oid in merged['organizationId'].astype(str).values]
    merged = merged.set_index('organizationId').loc[valid_order].reset_index()
    return merged


def fetch_timeseries(org_id: str,
                     metric: str,
                     start_date: str,
                     end_date: str,
                     api_key: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Download a daily governance timeseries from DeepDAO.

    Parameters
    ----------
    org_id: str
        The identifier of the organisation (as returned by DeepDAO).
    metric: str
        One of ``"proposals"``, ``"voters"`` or ``"votes"``.  These
        correspond to the number of proposals created, the number of
        unique voters on any proposal, and the number of individual
        votes cast, respectively, on each day.  The DeepDAO API
        endpoints follow the pattern ``daily_dao_{metric}`` for
        ``proposals`` and ``votes``; for ``voters`` the endpoint is
        ``daily_dao_voters``.
    start_date: str or date-like
        The inclusive start date for the timeseries in YYYY-MM-DD format.
    end_date: str or date-like
        The inclusive end date for the timeseries in YYYY-MM-DD format.
    api_key: optional, str
        Your DeepDAO API key.  If omitted, the ``DEEPDAO_API_KEY``
        environment variable will be used.

    Returns
    -------
    pandas.DataFrame or None
        A data frame indexed by date with a single column named after
        ``metric``.  If the API call fails or returns no data, ``None``
        is returned instead of raising an exception.
    """
    # Construct the endpoint based on the metric.  Note that the
    # ``voters`` metric uses a plural endpoint name (``daily_dao_voters``)
    # while ``proposals`` and ``votes`` follow the generic
    # ``daily_dao_{metric}`` pattern.
    endpoint_metric = metric
    if metric == "voters":
        endpoint = f"daily_dao_voters/{org_id}"
    else:
        endpoint = f"daily_dao_{metric}/{org_id}"
    url = (
        f"{BASE_URL}/timeseries/{endpoint}"
        f"?startDate={start_date}&endDate={end_date}"
    )
    response = requests.get(url, headers=_get_headers(api_key))
    if response.status_code != 200:
        return None
    js = response.json()
    if js is None:
        return None
    # Extract the list of daily records.  The API sometimes returns
    # {"data": [...], ...} and sometimes a bare list.  Always look for
    # the 'data' key first; if not present and the JSON is a list,
    # assume that list contains the daily records.
    records = None
    if isinstance(js, dict):
        if isinstance(js.get("data"), list):
            records = js.get("data")
        elif isinstance(js.get("resources"), list):
            records = js.get("resources")
    elif isinstance(js, list):
        records = js
    if not records:
        return None
    df = pd.DataFrame(records)
    if df.empty or 'date' not in df.columns or 'counter' not in df.columns:
        return None
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    # Rename the 'counter' column to the metric name for clarity
    df = df.rename(columns={'counter': metric})
    return df[[metric]]


def _aggregate_previous_month(series: pd.Series,
                             start_label: str,
                             end_label: str) -> pd.Series:
    """Aggregate a daily series into monthly sums labelled by the start of the next month.

    Given a daily time‑indexed series, this helper performs the following:

    1. Resample the series by calendar month (using month‑end labels) and sum
       all values within each month.  Missing days within a month are treated
       as zeros, consistent with the assumption that no activity occurred.
    2. Shift the resulting index forward by one month so that each sum is
       labelled by the first day of the following month.  For example,
       activity in December 2022 will be summed at 2022‑12‑31 and then
       relabelled as 2023‑01‑01.
    3. Reindex the series to include **every** monthly label between
       ``start_label`` and ``end_label`` (inclusive).  Missing values are
       filled with zeros to preserve dead months.

    Parameters
    ----------
    series: pandas.Series
        Daily time‑indexed values to aggregate.
    start_label: str
        First monthly label to include in the output (YYYY‑MM‑DD).  This
        should correspond to the first data point of interest (e.g.,
        "2023-01-01").
    end_label: str
        Last monthly label to include in the output (YYYY‑MM‑DD).  This
        should correspond to the label following the final activity month
        (e.g., "2025-01-01").

    Returns
    -------
    pandas.Series
        A series indexed by monthly start dates between ``start_label`` and
        ``end_label`` inclusive, containing the aggregated sums.  Missing
        months are represented as zeros.
    """
    # Step 1: resample by month end ("M") and sum activity within each calendar month.
    monthly_end = series.resample("M").sum()
    # Step 2: shift index forward to the first day of next month.
    monthly_start = monthly_end.copy()
    monthly_start.index = monthly_start.index + pd.offsets.MonthBegin(1)
    # Step 3: build complete index of month-start labels and reindex.
    start_dt = pd.to_datetime(start_label)
    end_dt = pd.to_datetime(end_label)
    full_index = pd.date_range(start_dt, end_dt, freq="MS")
    monthly_complete = monthly_start.reindex(full_index, fill_value=0)
    return monthly_complete.squeeze()


def compute_z_scores(series: pd.Series) -> pd.Series:
    """Compute the z‑score of a numeric series.

    The z‑score measures how many standard deviations each value is from
    the mean.  If the standard deviation is zero (all values identical),
    the function returns a series of zeros.

    Parameters
    ----------
    series: pandas.Series
        The input series, typically containing voter/proposal ratios.

    Returns
    -------
    pandas.Series
        A series of z‑scores aligned to the input index.
    """
    if series.empty:
        return series
    mean = series.mean()
    std = series.std(ddof=0)  # population std dev
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    return (series - mean) / std


def process_organization(org_id: str,
                         title: str,
                         start_date: str,
                         end_date: str,
                         api_key: Optional[str] = None,
                         metric: str = "voters",
                         zscore_scope: str = "window",
                         full_history_start_date: str = "2018-01-01") -> Optional[pd.DataFrame]:
    """Compute a z‑score series for a single organisation.

    This function downloads the daily proposal and voter counts for the
    specified organisation, aggregates them into monthly sums for the
    window defined by ``start_date`` and ``end_date`` (which correspond to
    the first and last monthly **labels**), computes the voter-to-proposal
    ratio for each month, normalises that ratio via z‑score, and returns
    the result as a tidy data frame.  Months with no activity are
    retained with zero values.

    Parameters
    ----------
    org_id: str
        The organisation identifier.
    title: str
        The human‑readable name of the organisation.  Used for joining
        price data and for reporting purposes.
    start_date: str
        The first monthly label (e.g., ``"2023-01-01"``).  Activity in
        the month preceding this label is included in the first data
        point.
    end_date: str
        The last monthly label (e.g., ``"2025-01-01"``).  Activity in
        the month preceding this label is included in the final data
        point.
    api_key: optional, str
        Your DeepDAO API key.
    metric: str, default "voters"
        Which governance activity to use in the numerator of the ratio.
        Acceptable values are ``"voters"`` (unique addresses voting),
        ``"votes"`` (total number of votes cast) and ``"proposals"``.
        The denominator is always the number of proposals.  You can
        experiment with both ``voters`` and ``votes`` depending on
        research questions.
    zscore_scope: str, default ``"window"``
        Determines how the z‑scores are normalised.  If ``"window"``,
        the mean and standard deviation are computed solely from the
        selected date window (``start_date`` through ``end_date``).  If
        ``"full"``, the mean and standard deviation are computed from
        the entire history available for the DAO (beginning at
        ``full_history_start_date``) and the resulting mean and
        standard deviation are applied to the selected window.
    full_history_start_date: str, default ``"2018-01-01"``
        The inclusive start date to use when ``zscore_scope`` is
        ``"full"``.  Ignored when ``zscore_scope`` is ``"window"``.

    Returns
    -------
    pandas.DataFrame or None
        A data frame with columns ``organizationId``, ``title``, ``date``,
        ``z_score``, ``ratio``, ``voters`` and ``proposals``.  If the
        underlying data cannot be retrieved or aggregated, ``None`` is
        returned.
    """
    # Determine the raw daily start/end dates.  We need to fetch one full
    # month earlier than the first label because the first data point
    # aggregates the prior month.  For example, start_date="2018-01-01"
    # implies we need activity from 2022-12-01 onwards.
    first_label = pd.to_datetime(start_date)
    raw_start = (first_label - pd.offsets.MonthBegin(1)).date().isoformat()
    # The last label indicates the month after the final activity month, so
    # we fetch up to the last day of the month preceding end_date.
    last_label = pd.to_datetime(end_date)
    raw_end = (last_label - pd.offsets.Day(1)).date().isoformat()

    # Fetch daily proposals and the selected numerator metric from DeepDAO.
    # The denominator remains ``proposals``.  If either call fails,
    # return None so that the organisation can be skipped.
    proposals_df = fetch_timeseries(org_id, "proposals", raw_start, raw_end, api_key=api_key)
    num_df = fetch_timeseries(org_id, metric, raw_start, raw_end, api_key=api_key)
    if proposals_df is None or num_df is None:
        return None

    # Align on common index of dates; fill missing dates with zeros to
    # account for days with no activity.  This prevents misaligned sums.
    idx = pd.date_range(raw_start, raw_end, freq="D")
    proposals = proposals_df.reindex(idx, fill_value=0).iloc[:, 0]
    numerator = num_df.reindex(idx, fill_value=0).iloc[:, 0]

    # Aggregate daily counts into monthly sums labelled by the first of the
    # following month.  Use a complete index to preserve dead months.
    proposals_monthly = _aggregate_previous_month(
        proposals, start_label=start_date, end_label=end_date
    )
    numerator_monthly = _aggregate_previous_month(
        numerator, start_label=start_date, end_label=end_date
    )

    # Compute the ratio; avoid division by zero by setting the ratio to zero
    # whenever the proposal count is zero.  An alternative would be to
    # leave NaN values in place, but zeros reflect the lack of governance
    # activity (no proposals and no voters/votes) more intuitively.
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = numerator_monthly / proposals_monthly
        ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Determine mean and standard deviation.  If zscore_scope is 'full',
    # compute them over the entire available history rather than just
    # the selected window.
    if zscore_scope.lower() == "full":
        # We need to recompute the ratio over the full history.  If the
        # caller did not request a start date earlier than raw_start,
        # fetch the extra daily data and aggregate accordingly.
        full_start_date = full_history_start_date or raw_start
        # Compute raw_end for full history as before
        full_proposals_df = fetch_timeseries(org_id, "proposals", full_start_date, raw_end, api_key=api_key)
        full_num_df = fetch_timeseries(org_id, metric, full_start_date, raw_end, api_key=api_key)
        if full_proposals_df is None or full_num_df is None:
            return None
        full_idx = pd.date_range(full_start_date, raw_end, freq="D")
        full_proposals = full_proposals_df.reindex(full_idx, fill_value=0).iloc[:, 0]
        full_numerator = full_num_df.reindex(full_idx, fill_value=0).iloc[:, 0]
        full_props_monthly = _aggregate_previous_month(full_proposals,
                                                      start_label=start_date,
                                                      end_label=end_date)
        full_num_monthly = _aggregate_previous_month(full_numerator,
                                                     start_label=start_date,
                                                     end_label=end_date)
        with np.errstate(divide='ignore', invalid='ignore'):
            full_ratio = full_num_monthly / full_props_monthly
            full_ratio = full_ratio.replace([np.inf, -np.inf], np.nan).fillna(0)
        # Use mean and std of full history
        mean = full_ratio.mean()
        std = full_ratio.std(ddof=0)
        if std == 0 or np.isnan(std):
            z_scores = pd.Series(0.0, index=ratio.index)
        else:
            z_scores = (ratio - mean) / std
    else:
        # Use window mean/std
        z_scores = compute_z_scores(ratio)

    # Assemble tidy DataFrame.
    out = pd.DataFrame({
        "organizationId": org_id,
        "title": title,
        "date": z_scores.index,
        "z_score": z_scores.values,
        "ratio": ratio.values,
        metric: numerator_monthly.values,
        "proposals": proposals_monthly.values,
    })
    return out


def save_master_zscores(dao_list: pd.DataFrame,
                        start_date: str = "2023-01-01",
                        end_date: str = "2025-01-01",
                        max_daos: int = 80,
                        api_key: Optional[str] = None,
                        verbose: bool = True,
                        metric: str = "voters",
                        zscore_scope: str = "window",
                        full_history_start_date: str = "2018-01-01") -> pd.DataFrame:
    """Process a list of organisations and assemble the master z‑score table.

    Parameters
    ----------
    dao_list: pandas.DataFrame
        DataFrame returned by :func:`fetch_top_organizations` containing at
        least ``organizationId`` and ``title`` columns.
    start_date: str, default ``"2023-01-01"``
        First monthly label to include in the aggregation.  Note that
        activity from the preceding month (December 2022) will be needed
        to compute the first point.
    end_date: str, default ``"2025-01-01"``
        Last monthly label to include in the aggregation.  Activity from
        December 2024 will be aggregated into this label.
    max_daos: int, default 80
        Maximum number of organisations to process.  Processing stops
        early if ``max_daos`` valid organisation series have been
        collected.
    api_key: optional, str
        Your DeepDAO API key.
    verbose: bool, default True
        If True, progress messages will be printed to stdout.

    Returns
    -------
    pandas.DataFrame
        Concatenated z‑score table for all successfully processed DAOs.
        Contains columns ``organizationId``, ``title``, ``date``,
        ``z_score``, ``ratio``, ``voters`` and ``proposals``.
    """
    dfs: List[pd.DataFrame] = []
    count = 0
    if dao_list is None or dao_list.empty:
        return pd.DataFrame(columns=["organizationId", "title", "date", "z_score", "ratio", "voters", "proposals"])
    for _, row in dao_list.iterrows():
        org_id = row.get("organizationId") or row.get("org_id")
        title = row.get("title") or row.get("name")
        if not org_id or not title:
            continue
        if verbose:
            print(f"Processing {title} (ID {org_id})…")
        df = process_organization(org_id,
                                  title,
                                  start_date,
                                  end_date,
                                  api_key=api_key,
                                  metric=metric,
                                  zscore_scope=zscore_scope,
                                  full_history_start_date=full_history_start_date)
        if df is not None:
            dfs.append(df)
            count += 1
        if count >= max_daos:
            break
    if not dfs:
        return pd.DataFrame(columns=["organizationId", "title", "date", "z_score", "ratio", "voters", "proposals"])
    return pd.concat(dfs, ignore_index=True)


def get_top_n_daos(df_top: pd.DataFrame, n: int = 80) -> List[Tuple[str, str]]:
    """Extract a list of (organizationId, title) tuples for the top N DAOs.

    This utility makes it easy to inspect or export the set of DAOs that
    will be included in the analysis.  One common use-case is to pass
    this list to colleagues who will manually source token price data.

    Parameters
    ----------
    df_top: pandas.DataFrame
        The DataFrame returned by :func:`fetch_top_organizations`.
    n: int, default 80
        Number of entries to extract.

    Returns
    -------
    list of tuples
        Each tuple contains ``(organizationId, title)``.  Only the first
        ``n`` entries (after dropping rows with missing identifiers or
        titles) are returned.
    """
    if df_top is None or df_top.empty:
        return []
    subset = df_top.dropna(subset=["organizationId", "title"]).head(n)
    return list(zip(subset["organizationId"].astype(str), subset["title"].astype(str)))


def load_price_files(directory: str) -> Dict[str, pd.DataFrame]:
    """Load historical token price CSV files from a directory.

    Each CSV file is expected to contain monthly price history for a single
    token.  The file name should contain the token or DAO name and end
    with ``.csv``, for example ``Uniswap Price 2025-08-03.csv``.  The
    first column must be named ``Date`` and contain dates in YYYY-MM-DD
    format.  The second column contains the price and will be renamed
    automatically to ``price``.  Additional columns will be ignored.

    Parameters
    ----------
    directory: str
        Path to the folder containing the price CSV files.

    Returns
    -------
    dict
        Mapping from a simplified token key (derived from the file name)
        to a data frame with columns ``date`` and ``price``.
    """
    price_data: Dict[str, pd.DataFrame] = {}
    if not os.path.isdir(directory):
        raise ValueError(f"Directory not found: {directory}")
    for fname in os.listdir(directory):
        if not fname.lower().endswith(".csv"):
            continue
        # Derive key by stripping suffix after " Price" and removing file
        # extension.  For example, "Uniswap Price 2025-08-03.csv" -> "Uniswap".
        base = fname[:-4] if fname.lower().endswith(".csv") else fname
        if " Price" in base:
            token_key = base.split(" Price", 1)[0].strip()
        else:
            token_key = base.strip()
        # Load file
        path = os.path.join(directory, fname)
        df = pd.read_csv(path)
        if df.shape[1] < 2:
            continue
        df = df.iloc[:, :2].copy()
        df.columns = ["Date", "price_raw"]
        df["date"] = pd.to_datetime(df["Date"])
        df = df.drop(columns=["Date"])
        df = df.rename(columns={"price_raw": "price"})
        price_data[token_key] = df
    return price_data


def merge_price_data(z_df: pd.DataFrame,
                     price_data: Dict[str, pd.DataFrame],
                     key_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Merge token price series onto the master z‑score table.

    The master table produced by :func:`save_master_zscores` contains
    multiple rows per DAO, each identified by the ``title`` column and
    indexed by the monthly label ``date``.  This function merges the
    corresponding price series (if available) onto those rows.  Because
    naming conventions may differ between DeepDAO titles and token names
    used in the pricing files, you can supply a custom mapping via the
    ``key_map`` argument.  If a DAO title is not found in ``price_data``
    (either directly or via ``key_map``), its price field will be
    ``NaN``.

    Parameters
    ----------
    z_df: pandas.DataFrame
        The master z‑score table, must contain at least ``title`` and
        ``date`` columns.
    price_data: dict
        Mapping from token keys (strings) to price DataFrames as returned
        by :func:`load_price_files`.
    key_map: optional, dict
        Dictionary mapping DAO titles (keys in ``z_df['title']``) to
        token keys (keys in ``price_data``).  Use this to account for
        cases where the DAO name differs from the token file name.  If
        ``None``, titles are used as keys directly.

    Returns
    -------
    pandas.DataFrame
        The input ``z_df`` with an additional ``price`` column.  Rows for
        which no price data is available will have ``NaN`` in the price
        column.
    """
    if z_df is None or z_df.empty:
        return z_df
    merged_frames: List[pd.DataFrame] = []
    # Precompute case-insensitive mapping of price_data keys
    lowered = {k.lower(): k for k in price_data}
    for title, group in z_df.groupby("title", sort=False):
        # Determine key: use explicit key_map if provided, else use title directly
        token_key = None
        if key_map and title in key_map:
            token_key = key_map[title]
        else:
            if title in price_data:
                token_key = title
            else:
                # case-insensitive lookup
                if title.lower() in lowered:
                    token_key = lowered[title.lower()]
        price_df = price_data.get(token_key)
        if price_df is not None:
            # merge on date
            merged = pd.merge(group, price_df, how='left', on='date')
        else:
            merged = group.copy()
            merged["price"] = np.nan
        merged_frames.append(merged)
    return pd.concat(merged_frames, ignore_index=True)


if __name__ == "__main__":
    # Example usage when running this module directly.  These statements
    # demonstrate how to pull the top organisations, process governance
    # metrics, and (optionally) join token prices.  They are wrapped in
    # a main guard so they don't execute on import.
    import json
    api_key = os.getenv("DEEPDAO_API_KEY")
    if not api_key:
        print("Warning: DEEPDAO_API_KEY environment variable not set. Requests may fail.")

    # Example workflow broken into discrete steps for easier debugging.
    # 1. Test API key and connection.
    print("Testing DeepDAO API connection…")
    if not test_connection(api_key):
        print("Connection test failed. Please verify your API key and network connectivity.")
    else:
        # 2. Pull the top organisations by AUM (e.g. first 500)
        print("Fetching top organisations by AUM…")
        try:
            top_df = fetch_top_organizations(limit=500, api_key=api_key)
            print(f"Retrieved {len(top_df)} organisations ranked by AUM.")
        except Exception as e:
            print(f"Error fetching top organisations: {e}")
            top_df = pd.DataFrame()

        # 3. Fetch all organisations and filter those with a tradable token
        print("Fetching full organisation list for token filtering…")
        try:
            all_orgs = fetch_all_organizations(api_key=api_key)
            token_orgs = filter_daos_with_tokens(all_orgs)
            print(f"Identified {len(token_orgs)} organisations with at least one token.")
        except Exception as e:
            print(f"Error fetching organisations: {e}")
            token_orgs = pd.DataFrame()

        # 4. Intersect the top list with the token list to get investable DAOs
        if not top_df.empty and not token_orgs.empty:
            investable = merge_top_and_tokens(top_df, token_orgs)
            print(f"Intersected top list with tokens: {len(investable)} organisations remain.")
            # Save list for manual price sourcing if desired
            out_csv = os.getenv("TOP_TOKENS_CSV", "tokens_only_top_aum.csv")
            investable.to_csv(out_csv, index=False)
            print(f"Investable DAOs saved to {out_csv}.")
        else:
            investable = pd.DataFrame()

        # 5. Compute z‑scores for the first N investable DAOs (default 80)
        # Derive the list of DAOs to process for z-score calculations.  If
        # there is a non‑empty investable set, use its organisationId and
        # title (falling back to id/name columns if necessary).  Otherwise
        # use the raw top list.
        if not investable.empty:
            cols = []
            if 'organizationId' in investable.columns:
                cols.append('organizationId')
            elif 'id' in investable.columns:
                investable = investable.rename(columns={'id': 'organizationId'})
                cols.append('organizationId')
            if 'title' in investable.columns:
                cols.append('title')
            elif 'name' in investable.columns:
                investable = investable.rename(columns={'name': 'title'})
                cols.append('title')
            z_input = investable[cols].copy()
        else:
            # Fallback to raw top_df if investable is empty
            base = top_df.copy()
            if 'organizationId' not in base.columns and 'id' in base.columns:
                base = base.rename(columns={'id': 'organizationId'})
            if 'title' not in base.columns and 'name' in base.columns:
                base = base.rename(columns={'name': 'title'})
            z_input = base[['organizationId', 'title']] if not base.empty else pd.DataFrame()

        if not z_input.empty:
            print("Computing z-scores for selected DAOs…")
            master = save_master_zscores(z_input,
                                         start_date="2018-01-01",
                                         end_date="2025-01-01",
                                         max_daos=80,
                                         api_key=api_key)
            if master is not None and not master.empty:
                print(f"Processed {master['title'].nunique()} DAOs. Master table shape: {master.shape}")
                # Example: merge price data if available
                price_dir = os.getenv("PRICE_DATA_DIR", "./price_data")
                if os.path.isdir(price_dir):
                    try:
                        prices = load_price_files(price_dir)
                        master_with_prices = merge_price_data(master, prices)
                        print("Merged price data. Sample:")
                        print(master_with_prices.head())
                    except ValueError as e:
                        print(e)
                # Save master table
                out_path = os.getenv("OUTPUT_CSV", "z_scores_master.csv")
                master.to_csv(out_path, index=False)
                print(f"Master z‑score table saved to {out_path}")