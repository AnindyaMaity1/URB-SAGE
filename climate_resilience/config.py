from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", ROOT / "database" / "climate_resilience.sqlite3"))
SAMPLE_DATA = DATA_DIR / "sample_zones.csv"
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
DEVICE = "cuda" if os.getenv("CLIMATE_DEVICE") == "cuda" else "cpu"