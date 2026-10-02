"""Build the final table using validated v05 optical corrections.

The v04 spatial-context table remains the structural base. Corrected B11/B12
and NDVI/NDBI/NDWI object statistics are transferred from v05 by Object_ID.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Object_based_fusion" / "Segmentation_output"
V04 = DATA_DIR / "objects_features_v04_spatial_context.csv"
V05 = DATA_DIR / "objects_features_v05_full.csv"
OUTPUT = DATA_DIR / "objects_features_final_full.csv"
TEMP_OUTPUT = DATA_DIR / "objects_features_final_full.tmp.csv"

ID = "Object_ID"
TRANSFER = [
    "S2_B11_mean",
    "S2_B11_std",
    "S2_B11_max",
    "S2_B12_mean",
    "S2_B12_std",
    "S2_B12_max",
    "NDVI_mean",
    "NDVI_std",
    "NDVI_max",
    "NDBI_mean",
    "NDBI_std",
    "NDBI_max",
    "NDWI_mean",
    "NDWI_std",
    "NDWI_max",
]
EXPECTED_ROWS = 336_109
EXPECTED_COLUMNS = 96
CHUNK_SIZE = 50_000


def progress(rows: int) -> None:
    fraction = min(rows / EXPECTED_ROWS, 1.0)
    width = 20
    filled = round(width * fraction)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r[{bar}] {fraction:6.1%} | {rows:,}/{EXPECTED_ROWS:,} objects", end="", flush=True)


def main(force: bool = False) -> None:
    for path in (V04, V05):
        if not path.exists():
            raise FileNotFoundError(path)
    if OUTPUT.exists() and not force:
        raise FileExistsError(f"Output already exists and was not overwritten: {OUTPUT}")

    v04_columns = list(pd.read_csv(V04, nrows=0).columns)
    v05_columns = list(pd.read_csv(V05, nrows=0).columns)
    required = {ID, *TRANSFER}
    for name, columns in (("v04", v04_columns), ("v05", v05_columns)):
        missing = sorted(required.difference(columns))
        if missing:
            raise KeyError(f"Missing required {name} columns: {missing}")
    if len(v04_columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Expected {EXPECTED_COLUMNS} v04 columns; found {len(v04_columns)}")

    v04_reader = pd.read_csv(V04, chunksize=CHUNK_SIZE)
    v05_reader = pd.read_csv(V05, usecols=[ID, *TRANSFER], chunksize=CHUNK_SIZE)
    rows_written = 0
    first = True
    try:
        for old, corrected in zip(v04_reader, v05_reader, strict=True):
            if len(old) != len(corrected):
                raise ValueError("Chunk row counts differ between v04 and v05.")
            old_ids = old[ID].to_numpy()
            corrected_ids = corrected[ID].to_numpy()
            if not np.array_equal(old_ids, corrected_ids):
                raise ValueError("Object_ID order differs between v04 and v05.")
            old.loc[:, TRANSFER] = corrected[TRANSFER].to_numpy()
            old.to_csv(TEMP_OUTPUT, mode="w" if first else "a", header=first, index=False)
            first = False
            rows_written += len(old)
            progress(rows_written)
        print()
        if rows_written != EXPECTED_ROWS:
            raise ValueError(f"Expected {EXPECTED_ROWS:,} rows; wrote {rows_written:,}.")
        os.replace(TEMP_OUTPUT, OUTPUT)
    except Exception:
        if TEMP_OUTPUT.exists():
            TEMP_OUTPUT.unlink()
        raise

    print(f"Created: {OUTPUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Replace the existing final table.")
    main(force=parser.parse_args().force)
