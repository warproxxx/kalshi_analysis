"""Shared helpers for the Kalshi analysis notebooks.

Import this from the numbered notebooks so 0/1/2/... all reuse the same
download, load, and categorize logic:

    import kalshi
    kalshi.download_kalshi_data()          # 0  - fetch raw trades (parquet/day)
    df = kalshi.load_trades()              # 1+ - load all raw trades
    df = kalshi.build_categorized()        # 1  - add 'category', cache to disk
    df = kalshi.load_categorized()         # 2+ - load the categorized dataset
"""

import os
import glob
import json
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

import requests
import pandas as pd
import polars as pl
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Paths / constants -- the single source of truth for every notebook.
# --------------------------------------------------------------------------- #
DATA_DIR = 'data'
PARQUET_DIR = os.path.join(DATA_DIR, 'parquet')          # one file per day
TRADES_CSV = os.path.join(DATA_DIR, 'kalshi_trade_data.csv')   # legacy raw dump
CATEGORIZED_PARQUET = os.path.join(DATA_DIR, 'df_with_category.parquet')
CATEGORIZED_CSV = os.path.join(DATA_DIR, 'df_with_category.csv')   # legacy fallback
CATEGORY_WISE_CSV = os.path.join(DATA_DIR, 'category_wise.csv')
CATEGORIES_JSON = 'categories.json'

COLUMNS = ['create_ts', 'ticker_name', 'contracts_traded', 'price']
BASE_URL = 'https://kalshi-public-docs.s3.amazonaws.com/reporting/trade_data_{}.json'
SERIES_URL = 'https://api.elections.kalshi.com/trade-api/v2/series'
# Friendly renames applied to the API's category names.
CATEGORY_ALIASES = {'Exotics': 'Parlay'}

# One shared session -> connection pooling / keep-alive across all requests.
_session = requests.Session()


# --------------------------------------------------------------------------- #
# 0 - Download
# --------------------------------------------------------------------------- #
def _path_for(out_dir, date_str):
    return os.path.join(out_dir, f'{date_str}.parquet')


def _fetch_and_save(args):
    """Download one day and write its own parquet file.

    Returns (date_str, status): 'ok' | 'empty' | 'failed'.
    Each worker writes a distinct file, so parallelism is fully safe.
    """
    out_dir, date_str = args
    try:
        r = _session.get(BASE_URL.format(date_str), timeout=30)
        if r.status_code != 200:
            return date_str, 'failed'
        data = r.json()
        if not data:
            return date_str, 'empty'
        df = pd.DataFrame(data)
        cols = [c for c in COLUMNS if c in df.columns]
        if not cols:
            return date_str, 'empty'
        df = df[cols]
        if 'create_ts' in df.columns:
            df['create_ts'] = pd.to_datetime(df['create_ts'], utc=True)
        # Normalize numeric dtypes so every day-file has the same schema
        # (raw JSON gives ints some days, strings others).
        for c in ('contracts_traded', 'price'):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').astype('Int64')
        # temp-then-rename -> never leave a half-written file that the resume
        # logic would mistake for a completed day.
        path = _path_for(out_dir, date_str)
        tmp = path + '.tmp'
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
        return date_str, 'ok'
    except Exception:
        return date_str, 'failed'


def download_kalshi_data(out_dir=PARQUET_DIR, months_back=3, start=None,
                         max_workers=8, overwrite=False):
    """Download Kalshi trades as one parquet file per day.

    Date window: pass start='YYYY-MM-DD' for a fixed start date, otherwise the
    last `months_back` months. End is always yesterday.

    Resume is automatic: any date whose parquet already exists is skipped
    (pass overwrite=True to force re-download). Days are fetched in parallel;
    each writes its own file, so there is no shared-file append/ordering risk.
    """
    os.makedirs(out_dir, exist_ok=True)

    if start is not None:
        start_date = datetime.strptime(start, '%Y-%m-%d').date()
    else:
        start_date = (datetime.now() - timedelta(days=30 * months_back)).date()
    end_date = (datetime.now() - timedelta(days=1)).date()

    dates, d = [], start_date
    while d <= end_date:
        dates.append(d.strftime('%Y-%m-%d'))
        d += timedelta(days=1)

    todo = [ds for ds in dates
            if overwrite or not os.path.exists(_path_for(out_dir, ds))]
    print(f"Window {start_date} -> {end_date}: {len(dates)} days, "
          f"{len(dates) - len(todo)} already present, downloading {len(todo)}.")

    counts = {'ok': 0, 'empty': 0, 'failed': 0}
    failed = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for date_str, status in tqdm(
                ex.map(_fetch_and_save, [(out_dir, ds) for ds in todo]),
                total=len(todo), desc="Downloading"):
            counts[status] += 1
            if status == 'failed':
                failed.append(date_str)

    print(f"\nDone. ok={counts['ok']}  empty={counts['empty']}  "
          f"failed={counts['failed']} -> {out_dir}")
    if failed:
        print(f"Failed (re-run to retry): {failed[:10]}"
              + (" ..." if len(failed) > 10 else ""))
    return out_dir


# --------------------------------------------------------------------------- #
# 1+ - Load raw trades
# --------------------------------------------------------------------------- #
def _coerce_numeric(df):
    """contracts_traded / price arrive as strings from the JSON dumps; make
    them numeric so groupby-sum etc. behave (contracts as Int, price as Int)."""
    casts = []
    if 'contracts_traded' in df.columns and df['contracts_traded'].dtype == pl.Utf8:
        casts.append(pl.col('contracts_traded').cast(pl.Float64, strict=False)
                     .cast(pl.Int64, strict=False))
    if 'price' in df.columns and df['price'].dtype == pl.Utf8:
        casts.append(pl.col('price').cast(pl.Float64, strict=False)
                     .cast(pl.Int64, strict=False))
    return df.with_columns(casts) if casts else df


def load_trades(parquet_dir=PARQUET_DIR, columns=None, polars=True,
                csv_fallback=TRADES_CSV):
    """Load all raw trades into one DataFrame.

    Uses polars' native multithreaded parquet reader (reads every day-file
    across all cores in one shot -- ~7x faster than pandas here, and its
    *.parquet glob safely skips in-progress *.tmp files). Returns a polars
    DataFrame by default; pass polars=False for pandas. `columns` prunes
    columns at read time. Falls back to the legacy CSV dump if no parquet yet.
    """
    glob_path = os.path.join(parquet_dir, '*.parquet')
    files = sorted(glob.glob(glob_path))
    if files:
        try:
            df = pl.read_parquet(glob_path, columns=columns)
        except pl.exceptions.SchemaError:
            # Day-files can disagree on dtype (contracts_traded/price written as
            # String some days, Int64 others). Read each with the numeric cols
            # cast to a common type, then concat.
            lfs = [pl.scan_parquet(f).with_columns(
                       pl.col('contracts_traded').cast(pl.Utf8),
                       pl.col('price').cast(pl.Utf8))
                   for f in files]
            df = pl.concat(lfs, how='vertical').collect()
            if columns:
                df = df.select(columns)
        df = _coerce_numeric(df)
        print(f"Loaded {df.height:,} rows from {len(files)} parquet files")
        return df if polars else df.to_pandas()
    if csv_fallback and os.path.exists(csv_fallback):
        print(f"Loading {csv_fallback} (CSV fallback)")
        df = pl.read_csv(csv_fallback, columns=columns)
        return df if polars else df.to_pandas()
    raise FileNotFoundError(
        f"No parquet in {parquet_dir} and no CSV at {csv_fallback}. "
        f"Run download_kalshi_data() first.")


# --------------------------------------------------------------------------- #
# 1 - Categorize
# --------------------------------------------------------------------------- #
def update_categories(path=CATEGORIES_JSON):
    """Fetch every Kalshi series + its category from the API and write
    {series_ticker: category} to categories.json. Re-run anytime to refresh.

    This is the single source of truth for categories: the Kalshi `/series`
    endpoint returns all ~11k series with authoritative category labels in one
    call, so no manual curation is needed.
    """
    series = _session.get(SERIES_URL, timeout=30).json()['series']
    cats = {s['ticker']: CATEGORY_ALIASES.get(s['category'], s['category'])
            for s in series if s.get('ticker') and s.get('category')}
    with open(path, 'w') as f:
        json.dump(cats, f, indent=4, ensure_ascii=False, sort_keys=True)
    from collections import Counter
    print(f"Wrote {len(cats)} series categories -> {path}")
    print(f"categories: {dict(Counter(cats.values()))}")
    return cats


def load_categories(path=CATEGORIES_JSON):
    """Load the series-ticker -> category mapping (see update_categories)."""
    with open(path) as f:
        return json.load(f)


def categorize(df, categories=None):
    """Return `df` with a 'category' column.

    Looks up each ticker's series prefix (the part before the first '-') in the
    API-derived category map -- an exact match, so no substring collisions.
    Works on a polars or pandas DataFrame and returns the same type.
    """
    if categories is None:
        categories = load_categories()
    if isinstance(df, pl.DataFrame):
        return df.with_columns(
            pl.col('ticker_name').str.extract(r'^([^-]+)-', 1)
              .replace_strict(categories, default='Uncategorized')
              .fill_null('Uncategorized')
              .alias('category'))
    out = df.copy()
    prefix = out['ticker_name'].str.extract(r'^([^-]+)-', expand=False)
    out['category'] = prefix.map(categories).fillna('Uncategorized')
    return out


def build_categorized(save=True, out_parquet=CATEGORIZED_PARQUET):
    """Load raw trades, add 'category', cache to parquet, return polars df."""
    df = categorize(load_trades(polars=True))
    if save:
        os.makedirs(os.path.dirname(out_parquet) or '.', exist_ok=True)
        df.write_parquet(out_parquet)
        print(f"Saved {df.height:,} rows -> {out_parquet}")
    return df


# --------------------------------------------------------------------------- #
# 2+ - Load the categorized dataset
# --------------------------------------------------------------------------- #
def with_parsed_ts(df, col='create_ts', alias='parsed_ts'):
    """Add a datetime `alias` column from `col` (polars).

    Parquet already stores create_ts as a datetime, so this is a no-op rename;
    the legacy CSV stores it as a string, which this parses. Robust to both.
    """
    expr = pl.col(col)
    if df.schema[col] == pl.Utf8:
        expr = expr.str.to_datetime()
    return df.with_columns(expr.alias(alias))


def load_categorized(parquet_path=CATEGORIZED_PARQUET, polars=True,
                     csv_fallback=CATEGORIZED_CSV):
    """Load the categorized dataset built by build_categorized().

    Reads the parquet cache with polars (multithreaded); falls back to the
    legacy CSV if the parquet isn't there yet. Returns a polars DataFrame by
    default; pass polars=False for pandas.
    """
    if os.path.exists(parquet_path):
        df = pl.read_parquet(parquet_path)
    elif csv_fallback and os.path.exists(csv_fallback):
        print(f"Loading {csv_fallback} (CSV fallback)")
        df = pl.read_csv(csv_fallback)
    else:
        raise FileNotFoundError(
            f"Neither {parquet_path} nor {csv_fallback} found. "
            f"Run build_categorized() in notebook 1 first.")
    return df if polars else df.to_pandas()
