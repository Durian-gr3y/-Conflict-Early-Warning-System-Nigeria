"""
src/processing/geography.py
───────────────────────────
Configuration-driven geographic manager for Nigeria's 36 states and FCT.

Allows zero-code scaling:
  Adding, removing, or toggling monitored states requires editing config/states.yaml only.
"""

import pathlib
from typing import Any
import yaml
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT / "config" / "states.yaml"


def load_states_config(path: pathlib.Path | None = None) -> dict[str, Any]:
    """Load states configuration YAML."""
    p = path or CONFIG_PATH
    if not p.exists():
        raise FileNotFoundError(f"States config not found at: {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("states", {})


def get_state_centroids(path: pathlib.Path | None = None) -> dict[str, tuple[float, float]]:
    """Return dictionary mapping state name -> (latitude, longitude)."""
    states = load_states_config(path)
    centroids = {}
    for name, info in states.items():
        centroids[name] = (float(info["lat"]), float(info["lon"]))
        # Add title-cased alias if FCT
        if name == "FCT":
            centroids["Fct"] = (float(info["lat"]), float(info["lon"]))
    return centroids


def get_pilot_states(path: pathlib.Path | None = None) -> list[str]:
    """Return list of states designated as the Northwest pilot theatre."""
    states = load_states_config(path)
    return [name for name, info in states.items() if info.get("is_pilot", False)]


def get_enabled_states(path: pathlib.Path | None = None) -> list[str]:
    """Return list of all currently enabled states."""
    states = load_states_config(path)
    return [name for name, info in states.items() if info.get("enabled", True)]


def get_states_dataframe(path: pathlib.Path | None = None) -> pd.DataFrame:
    """Return DataFrame representation of all states in the registry."""
    states = load_states_config(path)
    rows = []
    for name, info in states.items():
        rows.append({
            "admin1": name,
            "code": info.get("code"),
            "zone": info.get("zone"),
            "capital": info.get("capital"),
            "latitude": float(info.get("lat")),
            "longitude": float(info.get("lon")),
            "enabled": bool(info.get("enabled", True)),
            "is_pilot": bool(info.get("is_pilot", False)),
            "neighbors_count": len(info.get("neighbors", [])),
        })
    return pd.DataFrame(rows)


def validate_geographic_coverage(df: pd.DataFrame, state_col: str = "admin1") -> dict[str, Any]:
    """
    Check dataset coverage against enabled states in config/states.yaml.
    """
    enabled = set(get_enabled_states())
    present = set(df[state_col].unique())

    # Account for Fct alias
    if "Fct" in present:
        present.add("FCT")

    covered = enabled.intersection(present)
    missing = enabled.difference(present)

    return {
        "total_enabled_states": len(enabled),
        "covered_states_count": len(covered),
        "coverage_percentage": round(len(covered) / len(enabled) * 100, 1) if enabled else 100.0,
        "missing_states": sorted(list(missing)),
        "is_full_coverage": len(missing) == 0,
    }
