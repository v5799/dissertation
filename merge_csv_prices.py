"""
merge_csv_prices.py
====================

This standalone utility script merges daily price CSV files for DAO
tokens into a single table suitable for downstream analysis.  It
expects each file to contain at least two columns: a date column
(named ``Date`` or similar) and a price column.  The DAO title is
inferred from the filename up to the substring ``" Price"``.  The
merged output retains all distinct dates and aligns each DAO's
prices by date, leaving missing values as ``NaN``.

Usage from the command line::

    python merge_csv_prices.py --input-dir "C:/path/to/csvs" --output merged_prices.csv

Within a Jupyter notebook you can import and call
``merge_prices`` directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


def merge_prices(directory: str | Path, verbose: bool = True) -> pd.DataFrame:
    """Merge all CSV price files in the given directory.

    Parameters
    ----------
    directory : str or Path
        Folder containing price CSV files.  Only files with suffix
        ``.csv`` are considered.
    verbose : bool, default True
        Whether to print status messages while reading files.

    Returns
    -------
    pandas.DataFrame
        DataFrame with a ``date`` column and one column per DAO.  The
        rows are sorted chronologically.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Price directory not found: {directory}")
    csv_files = [f for f in directory.iterdir() if f.suffix.lower() == ".csv"]
    merged: Optional[pd.DataFrame] = None
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
        if merged is None:
            merged = sub
        else:
            merged = pd.merge(merged, sub, on="date", how="outer")
    if merged is None:
        raise ValueError(f"No valid price files in {directory}")
    merged = merged.sort_values(by="date").reset_index(drop=True)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge DAO price CSV files")
    parser.add_argument("--input-dir", required=True, help="Path to directory containing price CSV files")
    parser.add_argument("--output", required=True, help="Path to write merged CSV")
    args = parser.parse_args()
    merged = merge_prices(args.input_dir, verbose=True)
    merged.to_csv(args.output, index=False)
    print(f"Merged price table saved to {args.output} (shape {merged.shape})")


if __name__ == "__main__":
    main()