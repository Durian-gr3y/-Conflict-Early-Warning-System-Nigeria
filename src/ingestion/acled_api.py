"""
src/ingestion/acled_api.py
──────────────────────────
ACLED REST API client — event-level data for Nigeria.

STATUS: Disabled by default until an API key is granted.
        Set ACLED_API_KEY in your environment (or a .env file) and flip
        `enabled: true` in config/sources.yaml to activate.

WHAT THIS DOES vs THE WEEKLY AGG CSV
    The weekly aggregated CSV you have now is state-level only:
    one row = one week × one state × one event_type.
    The API returns individual events with:
        - exact latitude / longitude  (instead of state centroid)
        - exact date                  (instead of week)
        - actor names, disorder type, sub-event detail
        - geo_precision code (1=exact GPS, 8=country)

    This unlocks sub-state analysis (LGA level) and temporal precision
    needed for the spatio-temporal model in Phase 4.

LICENSING
    ACLED data is CC BY-NC-SA 4.0:
    - Attribution required.
    - Non-commercial use only (unless you have a paid licence).
    - You may not redistribute the raw events.
    - API terms: https://acleddata.com/terms-of-use/

FAILURE MODES (documented per sources.yaml)
    - Key not granted: raises ACLEDAuthError with next steps
    - Rate limiting: 1000 rows per call; paginated automatically
    - Schema drift: ACLED occasionally renames columns; check_schema() alerts you
    - Terms restrict redistribution: never expose raw rows in public dashboards
"""

from __future__ import annotations

import logging
import os
import pathlib
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()   # reads .env file if present

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ─── Constants ────────────────────────────────────────────────────────────────

ACLED_BASE_URL   = "https://api.acleddata.com/acled/read"
ACLED_PAGE_LIMIT = 5000     # rows per API call (max allowed)
ACLED_RATE_SLEEP = 1.0      # seconds between paginated calls (be a good citizen)

# Expected columns in the event-level API response
# Source: https://acleddata.com/resources/general-guides/#1603120929786-a5f39e00-2b19
EXPECTED_API_COLS = [
    "data_id", "iso", "event_id_cnty", "event_id_no_cnty",
    "event_date", "year", "time_precision", "disorder_type",
    "event_type", "sub_event_type", "actor1", "assoc_actor_1",
    "inter1", "actor2", "assoc_actor_2", "inter2", "interaction",
    "civilian_targeting", "iso3", "country", "admin1", "admin2",
    "admin3", "location", "latitude", "longitude", "geo_precision",
    "source", "source_scale", "notes", "fatalities", "tags", "timestamp",
]


# ─── Custom exceptions ────────────────────────────────────────────────────────

class ACLEDAuthError(Exception):
    """Raised when ACLED API key is missing or rejected."""

class ACLEDRateLimitError(Exception):
    """Raised when ACLED API returns a rate-limit response."""

class ACLEDSchemaError(Exception):
    """Raised when the API response schema doesn't match expectations."""


# ─── Client ───────────────────────────────────────────────────────────────────

class ACLEDClient:
    """
    Thin REST client for the ACLED events API.

    Parameters
    ----------
    api_key  : str   Your ACLED API key. If None, reads ACLED_API_KEY env var.
    email    : str   Your registered email. If None, reads ACLED_EMAIL env var.
    base_url : str   API endpoint (override for testing).
    dry_run  : bool  If True, print the request URL but do not call the API.
                     Useful to verify query parameters before your key arrives.
    """

    def __init__(
        self,
        api_key: str | None = None,
        email: str | None = None,
        base_url: str = ACLED_BASE_URL,
        dry_run: bool = False,
    ):
        self.api_key = api_key or os.environ.get("ACLED_API_KEY", "")
        self.email   = email   or os.environ.get("ACLED_EMAIL", "")
        self.base_url = base_url
        self.dry_run = dry_run

        if not self.api_key:
            logger.warning(
                "[WARN] ACLED_API_KEY not set. "
                "Set it in your environment or .env file.\n"
                "Request a key at: https://developer.acleddata.com/\n"
                "Running in DRY RUN mode — no API calls will be made."
            )
            self.dry_run = True

    def _get(self, params: dict) -> list[dict]:
        """Execute a single API call and return the list of event dicts."""
        params = {
            **params,
            "key":   self.api_key,
            "email": self.email,
            "limit": ACLED_PAGE_LIMIT,
        }

        if self.dry_run:
            import urllib.parse
            safe_params = {k: v for k, v in params.items() if k not in ("key", "email")}
            query = urllib.parse.urlencode(safe_params)
            print(f"[DRY RUN] Would call: {self.base_url}?{query}&key=***&email=***")
            return []

        resp = requests.get(self.base_url, params=params, timeout=60)

        if resp.status_code == 401:
            raise ACLEDAuthError(
                "ACLED API returned 401 Unauthorised.\n"
                "Check that ACLED_API_KEY and ACLED_EMAIL are correct.\n"
                "Request access: https://developer.acleddata.com/"
            )
        if resp.status_code == 429:
            raise ACLEDRateLimitError(
                "ACLED API rate limit hit. Wait 60 seconds and retry."
            )
        resp.raise_for_status()

        body = resp.json()
        if body.get("status") != 200:
            raise RuntimeError(
                f"ACLED API error: {body.get('status')} — {body.get('error', 'unknown')}"
            )
        return body.get("data", [])

    def fetch_nigeria_events(
        self,
        start_date: str,
        end_date: str,
        event_types: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Fetch all Nigeria events between start_date and end_date.

        Parameters
        ----------
        start_date  : str   'YYYY-MM-DD'
        end_date    : str   'YYYY-MM-DD'
        event_types : list  Optional filter, e.g. ['Battles', 'Riots'].
                            None = all event types.

        Returns
        -------
        pd.DataFrame  Event-level data with all ACLED columns.

        Note: For date ranges > 6 months, use fetch_nigeria_incremental()
        to avoid timeouts and make paginated calls easier to resume.
        """
        base_params: dict = {
            "country": "Nigeria",
            "event_date": f"{start_date}|{end_date}",
            "event_date_where": "BETWEEN",
        }
        if event_types:
            base_params["event_type"] = "|".join(event_types)

        all_rows: list[dict] = []
        page = 1
        while True:
            params = {**base_params, "page": page}
            logger.info(
                f"Fetching page {page}  "
                f"[{start_date} → {end_date}]  "
                f"rows so far: {len(all_rows):,}"
            )
            rows = self._get(params)
            if not rows:
                break
            all_rows.extend(rows)
            if len(rows) < ACLED_PAGE_LIMIT:
                break    # last page
            page += 1
            time.sleep(ACLED_RATE_SLEEP)

        if not all_rows:
            logger.info("No events returned.")
            return pd.DataFrame(columns=EXPECTED_API_COLS)

        df = pd.DataFrame(all_rows)
        df = self._coerce_types(df)
        logger.info(
            f"Fetched {len(df):,} events  "
            f"[{start_date} → {end_date}]  "
            f"Columns: {list(df.columns)}"
        )
        return df

    def fetch_nigeria_incremental(
        self,
        raw_dir: str | pathlib.Path,
        lookback_days: int = 35,
    ) -> pd.DataFrame:
        """
        Fetch the most recent events not yet in the local cache.

        Designed for the weekly automated update (Phase 5).
        Reads the latest cached file to find the last event date,
        then fetches from that date forward.

        Parameters
        ----------
        raw_dir      : path  Where cached ACLED CSVs are stored.
        lookback_days: int   How many days back to re-fetch for overlap
                             (ACLED may revise recent events). Default 35.
        """
        raw_dir = pathlib.Path(raw_dir)
        cached  = sorted(raw_dir.glob("acled_nigeria_events_*.parquet"))

        if cached:
            last_file = cached[-1]
            last_df   = pd.read_parquet(last_file)
            last_date = pd.to_datetime(last_df["event_date"]).max()
            start     = (last_date - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
            logger.info(f"Incremental fetch from {start} (last cache: {last_file.name})")
        else:
            # First run — fetch 2 years to bootstrap
            start = (datetime.today() - timedelta(days=730)).strftime("%Y-%m-%d")
            logger.info(f"No cache found. Bootstrapping from {start}.")

        end = datetime.today().strftime("%Y-%m-%d")
        new_df = self.fetch_nigeria_events(start_date=start, end_date=end)

        # Save to cache
        if len(new_df) > 0:
            fname = f"acled_nigeria_events_{datetime.today().strftime('%Y%m%d')}.parquet"
            out_path = raw_dir / fname
            new_df.to_parquet(out_path, index=False)
            logger.info(f"Cached: {out_path}")

        return new_df

    @staticmethod
    def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
        """Apply type coercions and check schema."""
        # Schema drift check
        actual = set(df.columns)
        expected = set(EXPECTED_API_COLS)
        missing = expected - actual
        extra   = actual - expected
        if missing:
            logger.warning(f"[SCHEMA DRIFT] Missing columns: {missing}")
        if extra:
            logger.info(f"[SCHEMA DRIFT] Extra columns (new in API): {extra}")

        df = df.copy()
        if "event_date" in df.columns:
            df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
        if "fatalities" in df.columns:
            df["fatalities"] = pd.to_numeric(df["fatalities"], errors="coerce").fillna(0).astype(int)
        if "latitude" in df.columns:
            df["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")
        if "longitude" in df.columns:
            df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        if "geo_precision" in df.columns:
            df["geo_precision"] = pd.to_numeric(df["geo_precision"], errors="coerce")
        return df


# ─── Aggregate event-level to monthly admin1 panel ────────────────────────────

def aggregate_acled_events_to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse event-level ACLED data to the monthly admin1 panel schema.

    Mirrors the structure produced by Phase 1 from the weekly aggregated CSV,
    but with richer quality metadata (geo_precision, number of events per type).

    Parameters
    ----------
    df : pd.DataFrame  Event-level ACLED data (output of ACLEDClient.fetch_*).

    Returns
    -------
    pd.DataFrame  Monthly admin1 panel with columns:
        month, admin1, events, fatalities, fatal_event,
        mean_geo_precision, pct_precise_events,
        weeks_observed,   (approximate — number of distinct weeks in the month)
        source
    """
    if len(df) == 0:
        return pd.DataFrame()

    df = df.copy()
    df["month"] = pd.to_datetime(df["event_date"]).dt.to_period("M")
    df["week"]  = pd.to_datetime(df["event_date"]).dt.to_period("W")

    agg = (
        df.groupby(["month", "admin1"])
        .agg(
            events=("data_id", "count"),
            fatalities=("fatalities", "sum"),
            mean_geo_precision=("geo_precision", "mean"),
            pct_precise_events=("geo_precision", lambda x: (x <= 2).mean()),
            weeks_observed=("week", "nunique"),
        )
        .reset_index()
    )
    agg["fatal_event"] = (agg["fatalities"] > 0).astype(int)
    agg["source"] = "ACLED_event"

    logger.info(
        f"Event-level panel: {len(agg):,} rows  |  "
        f"States: {agg['admin1'].nunique()}  |  "
        f"Months: {agg['month'].min()} → {agg['month'].max()}"
    )
    return agg


# ─── Dry-run demo ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("ACLED API client — dry run demo")
    print("(No API key needed — shows what request would be sent)\n")

    client = ACLEDClient(dry_run=True)
    _ = client.fetch_nigeria_events(
        start_date="2024-01-01",
        end_date="2024-12-31",
        event_types=["Battles", "Violence against civilians"],
    )
    print("\nTo enable real calls:")
    print("  1. Get a key: https://developer.acleddata.com/")
    print("  2. Add to your shell or .env file:")
    print("       ACLED_API_KEY=your_key_here")
    print("       ACLED_EMAIL=your@email.com")
    print("  3. Set enabled: true in config/sources.yaml under 'acled_api'")
