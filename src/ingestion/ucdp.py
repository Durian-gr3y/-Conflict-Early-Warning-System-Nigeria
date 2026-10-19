"""
src/ingestion/ucdp.py
─────────────────────
UCDP Georeferenced Event Dataset (GED) — download and parse module.

Covers:
  - Annual bulk release  (version 24.1, released ~June 2024, events through 2023)
  - Candidate dataset    (rolling monthly updates, provisional)

Usage:
    from src.ingestion.ucdp import download_ucdp_ged, load_ucdp_nigeria

    # 1. Download (only needed once per version)
    download_ucdp_ged(version="24.1", dest_dir="data/raw")
    download_ucdp_candidate(dest_dir="data/raw")

    # 2. Load + filter to Nigeria + align to monthly admin1 schema
    ged = load_ucdp_nigeria("data/raw")

What is UCDP GED?
    The Uppsala Conflict Data Program's Georeferenced Event Dataset records
    individual violent events with at least one organised armed actor.
    Inclusion threshold: any reported deaths (vs ACLED's lower threshold).
    Coverage: global 1989–present.

Key column differences vs ACLED:
    ACLED 'fatalities' ↔ UCDP 'best'   (best estimate; also has 'low' and 'high')
    ACLED 'event_type' ↔ UCDP 'type_of_violence'
        1 = state-based conflict
        2 = non-state conflict
        3 = one-sided violence (attacks on civilians)
    ACLED 'admin1'     ↔ UCDP 'adm_1'   (needs name normalisation)

Failure modes documented in config/sources.yaml:
    - ~12-month lag for annual release
    - Stricter inclusion → undercounts vs ACLED
    - Candidate dataset may be revised or retracted
"""

from __future__ import annotations

import io
import logging
import pathlib
import re
import time
import zipfile
from datetime import datetime
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ─── Constants ────────────────────────────────────────────────────────────────

NIGERIA_COUNTRY_ID = 475          # UCDP internal ID for Nigeria

# Version manifest — add new versions here as UCDP releases them
UCDP_VERSIONS: dict[str, str] = {
    "24.1": "https://ucdp.uu.se/downloads/ged/ged241-csv.zip",
    "23.1": "https://ucdp.uu.se/downloads/ged/ged231-csv.zip",
    "22.1": "https://ucdp.uu.se/downloads/ged/ged221-csv.zip",
}

# Candidate dataset URL (rolling updates, ~monthly)
UCDP_CANDIDATE_URL = (
    "https://ucdp.uu.se/downloads/candidateged/GEDEvent_v24_0_1candidate.csv"
)

# Geo-precision codes → readable labels (lower = more precise)
# UCDP uses 1–7; 1 = exact GPS, 7 = country centroid
GEO_PRECISION_LABEL = {
    1: "exact",
    2: "within_25km",
    3: "within_25_100km",
    4: "capital",
    5: "centroid_adm1",
    6: "centroid_adm2",
    7: "country",
}

# event_clarity codes
EVENT_CLARITY_LABEL = {
    1: "clear",
    2: "unclear",
}

# Nigerian state name harmonisation table
# Maps UCDP 'adm_1' values → canonical state names used in our panel
UCDP_TO_CANONICAL: dict[str, str] = {
    "Zamfara": "Zamfara",
    "Katsina": "Katsina",
    "Sokoto": "Sokoto",
    "Kaduna": "Kaduna",
    "Borno": "Borno",
    "Yobe": "Yobe",
    "Adamawa": "Adamawa",
    "Lagos": "Lagos",
    "Ogun": "Ogun",
    "Oyo": "Oyo",
    "Rivers": "Rivers",
    "Delta": "Delta",
    "Bayelsa": "Bayelsa",
    "Plateau": "Plateau",
    "Benue": "Benue",
    "Nasarawa": "Nasarawa",
    "Kano": "Kano",
    "Jigawa": "Jigawa",
    "Kebbi": "Kebbi",
    "Niger": "Niger",
    "Kwara": "Kwara",
    # Common variants
    "Abuja Federal Capital Territory": "FCT",
    "Federal Capital Territory": "FCT",
    "FCT": "FCT",
    "Akwa Ibom": "Akwa Ibom",
    "Anambra": "Anambra",
    "Imo": "Imo",
    "Enugu": "Enugu",
    "Ebonyi": "Ebonyi",
    "Cross River": "Cross River",
    "Abia": "Abia",
    "Osun": "Osun",
    "Ondo": "Ondo",
    "Ekiti": "Ekiti",
    "Edo": "Edo",
    "Gombe": "Gombe",
    "Taraba": "Taraba",
    "Bauchi": "Bauchi",
    "Kogi": "Kogi",
    "Ondo State": "Ondo",
    "Niger State": "Niger",
    "Zamfara State": "Zamfara",
}

# ─── Download helpers ─────────────────────────────────────────────────────────

def _download_with_retry(
    url: str,
    dest_path: pathlib.Path,
    max_retries: int = 3,
    chunk_size: int = 1024 * 1024,  # 1 MB
    timeout: int = 120,
) -> pathlib.Path:
    """
    Stream-download url → dest_path with retry logic.
    Shows a simple progress indicator without tqdm dependency.
    """
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"[attempt {attempt}/{max_retries}] Downloading: {url}")
            resp = requests.get(url, stream=True, timeout=timeout)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(dest_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(f"\r  {pct:5.1f}%  ({downloaded/1e6:.1f} MB / "
                              f"{total/1e6:.1f} MB)", end="", flush=True)
            print()  # newline after progress
            logger.info(f"Saved: {dest_path}  ({dest_path.stat().st_size/1e6:.1f} MB)")
            return dest_path
        except (requests.RequestException, IOError) as exc:
            logger.warning(f"Attempt {attempt} failed: {exc}")
            if attempt < max_retries:
                wait = 2 ** attempt
                logger.info(f"Retrying in {wait}s …")
                time.sleep(wait)
    raise RuntimeError(f"Failed to download {url} after {max_retries} attempts.")


def download_ucdp_ged(
    version: str = "24.1",
    dest_dir: str | pathlib.Path = "data/raw",
    force: bool = False,
) -> pathlib.Path:
    """
    Download the annual UCDP GED bulk release as a zip, extract the CSV.

    Parameters
    ----------
    version   : str   Version string, e.g. "24.1". Must be in UCDP_VERSIONS.
    dest_dir  : path  Directory to save files.
    force     : bool  Re-download even if file already exists.

    Returns
    -------
    pathlib.Path  Path to the extracted CSV file.

    Failure modes
    -------------
    - URL returns 404 if UCDP restructures their download page.
      Check https://ucdp.uu.se/downloads/ for current URLs and update
      UCDP_VERSIONS above.
    - File may be corrupted if download is interrupted; use force=True to retry.

    Licensing
    ---------
    UCDP GED is released under CC BY 4.0.
    Attribution required: 'Sundberg & Melander (2013) + UCDP/PRIO'
    """
    if version not in UCDP_VERSIONS:
        raise ValueError(
            f"Unknown version '{version}'. "
            f"Available: {list(UCDP_VERSIONS.keys())}"
        )

    dest_dir = pathlib.Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    url = UCDP_VERSIONS[version]
    ver_slug = version.replace(".", "")            # "24.1" → "241"
    zip_path = dest_dir / f"ged{ver_slug}-csv.zip"
    csv_name = f"GEDEvent_v{version}.csv"
    csv_path = dest_dir / csv_name

    if csv_path.exists() and not force:
        logger.info(f"Already exists (use force=True to re-download): {csv_path}")
        return csv_path

    # Download zip
    _download_with_retry(url, zip_path)

    # Extract the main CSV from the zip
    logger.info(f"Extracting {zip_path} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        logger.info(f"  Zip contains: {members}")
        # Find the main CSV (the largest .csv file in the archive)
        csv_members = [m for m in members if m.endswith(".csv")]
        if not csv_members:
            raise RuntimeError(f"No CSV found in zip. Contents: {members}")
        # Pick the largest
        target = max(csv_members,
                     key=lambda m: zf.getinfo(m).file_size)
        logger.info(f"  Extracting: {target}")
        with zf.open(target) as src, open(csv_path, "wb") as dst:
            dst.write(src.read())

    logger.info(f"Extracted: {csv_path}  ({csv_path.stat().st_size/1e6:.1f} MB)")

    # Stamp download metadata
    meta_path = dest_dir / f"ged{ver_slug}_download_meta.json"
    import json
    with open(meta_path, "w") as f:
        json.dump({
            "source": "UCDP GED",
            "version": version,
            "url": url,
            "downloaded_at": datetime.now().isoformat(),
            "file_size_bytes": csv_path.stat().st_size,
            "license": "CC BY 4.0",
            "citation": "Sundberg, R & Melander, E (2013). UCDP/PRIO.",
        }, f, indent=2)

    return csv_path


def download_ucdp_candidate(
    dest_dir: str | pathlib.Path = "data/raw",
    force: bool = False,
    url: str = UCDP_CANDIDATE_URL,
) -> pathlib.Path:
    """
    Download the UCDP GED candidate (provisional) dataset.

    IMPORTANT: Candidate data is provisional and may be revised or retracted
    in subsequent releases. Never use candidate data as the sole source for
    any analytical conclusion. Always flag it as provisional in outputs.

    The candidate dataset URL changes with each monthly release.
    Update UCDP_CANDIDATE_URL at the top of this file when needed.
    Check: https://ucdp.uu.se/downloads/
    """
    dest_dir = pathlib.Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Extract version slug from URL to create a stable filename
    slug_match = re.search(r"GEDEvent_(v[\d_]+candidate)\.csv", url)
    slug = slug_match.group(1) if slug_match else "candidate_latest"
    csv_path = dest_dir / f"ucdp_ged_{slug}.csv"

    if csv_path.exists() and not force:
        logger.info(f"Candidate already exists (use force=True): {csv_path}")
        return csv_path

    logger.info("[WARN] Candidate data is PROVISIONAL — flag all outputs accordingly")
    _download_with_retry(url, csv_path)

    import json
    meta_path = dest_dir / f"ucdp_ged_{slug}_meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            "source": "UCDP GED Candidate",
            "url": url,
            "downloaded_at": datetime.now().isoformat(),
            "provisional": True,
            "warning": "Subject to revision. Do not use as sole source.",
        }, f, indent=2)

    return csv_path


# ─── Parsing ──────────────────────────────────────────────────────────────────

def parse_ucdp_raw(csv_path: str | pathlib.Path) -> pd.DataFrame:
    """
    Load UCDP GED CSV and apply minimal, documented coercions.

    Returns a DataFrame with original UCDP columns plus derived helpers:
        date_start_dt  : datetime parsed from date_start
        date_end_dt    : datetime parsed from date_end
        geo_prec_label : human-readable geo_precision
        event_clarity_label : human-readable event_clarity
        duration_days  : date_end_dt - date_start_dt (in days)
        uncertainty_fatalities : high - low  (fatality uncertainty range)
    """
    logger.info(f"Loading UCDP GED: {csv_path}")
    df = pd.read_csv(
        csv_path,
        low_memory=False,
        dtype={
            # Keep UCDP IDs as strings to avoid int overflow
            "id": str,
            "conflict_new_id": str,
            "dyad_new_id": str,
            "country_id": "Int64",
        },
    )
    logger.info(f"  Rows loaded: {len(df):,}  |  Columns: {len(df.columns)}")

    # Date parsing — UCDP dates can be "YYYY-MM-DD" or just "YYYY"
    for col in ["date_start", "date_end"]:
        df[f"{col}_dt"] = pd.to_datetime(df[col], errors="coerce")

    # Derived fields
    df["geo_prec_label"] = df["where_prec"].map(GEO_PRECISION_LABEL)
    df["event_clarity_label"] = df["event_clarity"].map(EVENT_CLARITY_LABEL)
    df["duration_days"] = (df["date_end_dt"] - df["date_start_dt"]).dt.days.fillna(0)
    df["uncertainty_fatalities"] = (
        pd.to_numeric(df["high"], errors="coerce") -
        pd.to_numeric(df["low"],  errors="coerce")
    )

    return df


def filter_nigeria(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to Nigeria events only.

    UCDP uses country_id = 475 for Nigeria.
    We also keep a filter on the country name as a belt-and-suspenders check.
    """
    mask_id   = df["country_id"] == NIGERIA_COUNTRY_ID
    mask_name = df["country"].str.strip().str.lower() == "nigeria"
    combined  = mask_id | mask_name

    nigeria = df[combined].copy()
    logger.info(f"  Nigeria events: {len(nigeria):,} / {len(df):,} total")

    # Sanity check: are there rows matching name but not ID (data integrity)?
    mismatch = df[mask_name & ~mask_id]
    if len(mismatch) > 0:
        logger.warning(
            f"  [WARN] {len(mismatch)} rows match 'Nigeria' by name "
            f"but not by country_id=475. Check UCDP_TO_CANONICAL."
        )
    return nigeria


def normalise_admin1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Harmonise UCDP 'adm_1' values to canonical state names.

    Nigerian place-name ambiguity notes:
    - UCDP sometimes writes "Niger State" vs "Niger" (risk: confused with Niger Republic)
    - "Federal Capital Territory" / "Abuja FCT" / "FCT" — all same
    - Accents and diacritics stripped (UCDP uses ASCII; ACLED may not)
    - Unmatched values are kept as-is and logged as warnings

    Returns the DataFrame with an 'admin1' column added.
    """
    raw = df["adm_1"].str.strip().str.title()
    mapped = raw.map(UCDP_TO_CANONICAL)

    # Log anything we couldn't map
    unmapped = raw[mapped.isna()].unique()
    if len(unmapped) > 0:
        logger.warning(
            f"  [WARN] admin1 values not in UCDP_TO_CANONICAL "
            f"(kept as-is): {sorted(unmapped)}"
        )
        # Fall back to the raw title-cased value
        mapped = mapped.fillna(raw)

    df = df.copy()
    df["admin1"] = mapped
    return df


def map_event_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map UCDP 'type_of_violence' codes to a comparable event_type label.

    UCDP codes:
        1 = state-based conflict      → 'Battles'
        2 = non-state conflict        → 'Non-state Battles'
        3 = one-sided violence        → 'Violence against civilians'

    This is an approximation — ACLED and UCDP use different ontologies.
    The 'source' column lets downstream code know which system the label came from.
    """
    tov_map = {
        1: "Battles",
        2: "Non-state Battles",
        3: "Violence against civilians",
    }
    df = df.copy()
    df["event_type_ucdp"] = df["type_of_violence"].map(tov_map).fillna("Unknown")
    return df


def aggregate_to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate UCDP Nigeria events to the monthly admin1 panel schema.

    Output columns mirror the ACLED monthly panel so both can be merged:
        month              : pd.Period('M')
        admin1             : str
        events_ucdp        : int   number of UCDP events
        fatalities_ucdp    : int   sum of 'best' estimate
        fatalities_low     : int   sum of 'low' estimate
        fatalities_high    : int   sum of 'high' estimate
        fatal_event_ucdp   : int   1 if fatalities_ucdp > 0
        mean_geo_prec      : float mean where_prec (1=best, 7=worst)
        pct_clear_events   : float fraction with event_clarity == 1
        source             : str   'UCDP' (for provenance tracking)
    """
    if len(df) == 0:
        logger.warning("aggregate_to_monthly: input DataFrame is empty.")
        return pd.DataFrame()

    df = df.copy()
    df["month"] = df["date_start_dt"].dt.to_period("M")

    agg = (
        df.groupby(["month", "admin1"])
        .agg(
            events_ucdp=("id", "count"),
            fatalities_ucdp=("best", "sum"),
            fatalities_low=("low", "sum"),
            fatalities_high=("high", "sum"),
            mean_geo_prec=("where_prec", "mean"),
            pct_clear_events=("event_clarity", lambda x: (x == 1).mean()),
        )
        .reset_index()
    )
    agg["fatal_event_ucdp"] = (agg["fatalities_ucdp"] > 0).astype(int)
    agg["source"] = "UCDP"

    # Cast fatality columns to int (sum produces float if there are NaNs)
    for col in ["fatalities_ucdp", "fatalities_low", "fatalities_high"]:
        agg[col] = agg[col].fillna(0).astype(int)

    logger.info(
        f"  Monthly UCDP panel: {len(agg):,} rows  |  "
        f"States: {agg['admin1'].nunique()}  |  "
        f"Months: {agg['month'].min()} → {agg['month'].max()}"
    )
    return agg


# ─── Top-level convenience function ───────────────────────────────────────────

def load_ucdp_nigeria(
    raw_dir: str | pathlib.Path = "data/raw",
    version: str = "24.1",
    include_candidate: bool = False,
) -> pd.DataFrame:
    """
    Full pipeline: find CSV → parse → filter Nigeria → normalise → aggregate.

    Parameters
    ----------
    raw_dir           : path  Directory containing downloaded UCDP CSVs.
    version           : str   Annual version to load.
    include_candidate : bool  If True, also load and append candidate data.
                              Candidate rows are tagged source='UCDP_candidate'.

    Returns
    -------
    pd.DataFrame  Monthly admin1 panel, ready to merge with ACLED panel.
    """
    raw_dir = pathlib.Path(raw_dir)

    # ── Find annual CSV ───────────────────────────────────────────────────────
    ver_slug = version.replace(".", "")
    candidates = list(raw_dir.glob(f"GEDEvent_v{version}.csv"))
    if not candidates:
        # Fall back: look for any GEDEvent CSV
        candidates = sorted(raw_dir.glob("GEDEvent_v*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No UCDP GED CSV found in {raw_dir}.\n"
            f"Run: download_ucdp_ged(version='{version}', dest_dir='{raw_dir}')"
        )
    csv_path = candidates[-1]   # most recent if multiple
    logger.info(f"Using UCDP file: {csv_path.name}")

    # ── Parse pipeline ────────────────────────────────────────────────────────
    raw    = parse_ucdp_raw(csv_path)
    nga    = filter_nigeria(raw)
    nga    = normalise_admin1(nga)
    nga    = map_event_type(nga)
    annual = aggregate_to_monthly(nga)

    if not include_candidate:
        return annual

    # ── Candidate data ────────────────────────────────────────────────────────
    import re as _re
    cand_files = sorted(raw_dir.glob("ucdp_ged_*candidate*.csv"))
    if not cand_files:
        logger.warning("include_candidate=True but no candidate CSV found. Skipping.")
        return annual

    cand_path = cand_files[-1]
    logger.info(f"Appending candidate data: {cand_path.name}")
    cand_raw = parse_ucdp_raw(cand_path)
    cand_nga = filter_nigeria(cand_raw)
    cand_nga = normalise_admin1(cand_nga)
    cand_nga = map_event_type(cand_nga)
    cand_agg = aggregate_to_monthly(cand_nga)
    cand_agg["source"] = "UCDP_candidate"

    # Avoid duplicate months: candidate fills in months beyond annual
    max_annual_month = annual["month"].max() if len(annual) > 0 else pd.Period("1900-01")
    cand_new = cand_agg[cand_agg["month"] > max_annual_month]
    logger.info(f"  Candidate adds {len(cand_new):,} rows beyond annual release.")

    combined = pd.concat([annual, cand_new], ignore_index=True)
    return combined


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(
        description="Download and parse UCDP GED for Nigeria"
    )
    parser.add_argument("--version", default="24.1",
                        help="UCDP annual version (default: 24.1)")
    parser.add_argument("--raw-dir", default="data/raw",
                        help="Directory for raw downloads")
    parser.add_argument("--out-dir", default="data/processed",
                        help="Directory for processed output")
    parser.add_argument("--candidate", action="store_true",
                        help="Also download and include candidate data")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if file exists")
    args = parser.parse_args()

    raw_dir = pathlib.Path(args.raw_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download
    csv_path = download_ucdp_ged(
        version=args.version,
        dest_dir=raw_dir,
        force=args.force,
    )
    if args.candidate:
        download_ucdp_candidate(dest_dir=raw_dir, force=args.force)

    # Step 2: Parse + aggregate
    panel = load_ucdp_nigeria(
        raw_dir=raw_dir,
        version=args.version,
        include_candidate=args.candidate,
    )

    # Step 3: Save
    out_path = out_dir / "ucdp_nigeria_monthly.parquet"
    try:
        panel.to_parquet(out_path, index=False)
    except Exception:
        out_path = out_dir / "ucdp_nigeria_monthly.csv"
        panel.to_csv(out_path, index=False)
    logger.info(f"Saved: {out_path}")

    print(panel.head(10).to_string())
    print(f"\nShape: {panel.shape}")
    sys.exit(0)
