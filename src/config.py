"""Central configuration: all file paths and global constants for the project."""
from pathlib import Path

# Project root = the folder that contains src/ (project_02)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data folders
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

# Raw M5 files
CALENDAR_CSV = DATA_DIR / "calendar.csv"
SALES_EVAL_CSV = DATA_DIR / "sales_train_evaluation.csv"
SALES_VAL_CSV = DATA_DIR / "sales_train_validation.csv"
PRICES_CSV = DATA_DIR / "sell_prices.csv"

# Processed outputs
PANEL_PARQUET = PROCESSED_DIR / "panel_subset.parquet"
SUBSET_IDS_CSV = RESULTS_DIR / "subset_series.csv"

# Global constants
SEED = 42            # fixed random seed for reproducibility
HORIZON = 28         # forecast horizon in days
N_PER_STORE = 100    # series sampled per store for the project subset

# Chronological split by day number (d_1 ... d_1941)
TRAIN_END_DAY = 1885
VAL_START_DAY, VAL_END_DAY = 1886, 1913
TEST_START_DAY, TEST_END_DAY = 1914, 1941