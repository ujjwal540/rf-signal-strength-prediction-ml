"""Synthetic RF dataset generation and loading module."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    DATASET_PATH,
    ENVIRONMENT_TYPES,
    N_SAMPLES,
    RANDOM_STATE,
)


def _environment_loss_map() -> dict[str, float]:
    """Return baseline propagation loss adjustments by environment."""
    return {
        "Urban": 12.0,
        "Suburban": 7.0,
        "Rural": 3.5,
        "Indoor": 18.0,
        "Industrial": 14.0,
    }


def generate_synthetic_rf_dataset(
    n_samples: int = N_SAMPLES,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Generate a realistic synthetic RF dataset.

    The RSSI model combines free-space path loss (FSPL) with environment and
    obstacle penalties, plus stochastic fading noise.

    Args:
        n_samples: Number of rows to generate.
        random_state: Seed for reproducibility.

    Returns:
        Generated dataframe containing RF features and target RSSI.
    """
    rng = np.random.default_rng(random_state)

    distance_m = rng.uniform(10, 5000, size=n_samples)
    frequency_mhz = rng.choice([700, 800, 900, 1800, 2100, 2600, 3500], size=n_samples)
    transmit_power_dbm = rng.uniform(20, 46, size=n_samples)
    antenna_gain_dbi = rng.uniform(0, 18, size=n_samples)
    obstacle_loss_db = rng.uniform(0, 30, size=n_samples)
    environment_type = rng.choice(ENVIRONMENT_TYPES, size=n_samples)

    # FSPL in dB with distance in km and frequency in MHz
    distance_km = np.clip(distance_m / 1000.0, 0.01, None)
    fspl_db = 32.44 + 20 * np.log10(distance_km) + 20 * np.log10(frequency_mhz)

    env_loss_lookup = _environment_loss_map()
    env_loss_db = np.array([env_loss_lookup[e] for e in environment_type])

    shadowing_noise = rng.normal(0, 4.5, size=n_samples)

    rssi_dbm = (
        transmit_power_dbm
        + antenna_gain_dbi
        - fspl_db
        - obstacle_loss_db
        - env_loss_db
        + shadowing_noise
    )

    data = pd.DataFrame(
        {
            "distance_m": distance_m,
            "frequency_mhz": frequency_mhz,
            "transmit_power_dbm": transmit_power_dbm,
            "antenna_gain_dbi": antenna_gain_dbi,
            "obstacle_loss_db": obstacle_loss_db,
            "environment_type": environment_type,
            "rssi_dbm": rssi_dbm,
        }
    )

    return data


def save_dataset(df: pd.DataFrame, path: Path = DATASET_PATH) -> None:
    """Save dataset to CSV.

    Args:
        df: Dataframe to persist.
        path: Destination CSV path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_or_create_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Load dataset from path, creating synthetic data if unavailable.

    Args:
        path: Dataset file path.

    Returns:
        RF dataset as dataframe.
    """
    if path.exists():
        return pd.read_csv(path)

    df = generate_synthetic_rf_dataset()
    save_dataset(df, path)
    return df


def create_sample_prediction_input(
    df: pd.DataFrame,
    out_path: Optional[Path] = None,
    n_rows: int = 20,
) -> pd.DataFrame:
    """Create sample input file for inference.

    Args:
        df: Source dataframe.
        out_path: Output path for sample CSV.
        n_rows: Number of rows in sample.

    Returns:
        Sample feature dataframe.
    """
    sample = df.drop(columns=["rssi_dbm"]).sample(n=n_rows, random_state=RANDOM_STATE)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sample.to_csv(out_path, index=False)
    return sample
