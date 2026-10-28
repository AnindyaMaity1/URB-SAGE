import sqlite3
from contextlib import contextmanager

from .config import DATABASE_PATH, SAMPLE_DATA


SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    id INTEGER PRIMARY KEY, name TEXT NOT NULL, source_type TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, row_count INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS zones (
    zone_id TEXT PRIMARY KEY, dataset_id INTEGER NOT NULL, name TEXT NOT NULL,
    latitude REAL NOT NULL, longitude REAL NOT NULL, features_json TEXT NOT NULL,
    FOREIGN KEY(dataset_id) REFERENCES datasets(id)
);
CREATE TABLE IF NOT EXISTS climate_observations (
    id INTEGER PRIMARY KEY, zone_id TEXT NOT NULL, timestep INTEGER NOT NULL,
    rainfall_mm REAL NOT NULL, temperature_c REAL NOT NULL, humidity_pct REAL NOT NULL,
    FOREIGN KEY(zone_id) REFERENCES zones(zone_id)
);
CREATE TABLE IF NOT EXISTS model_predictions (
    id INTEGER PRIMARY KEY, run_id TEXT NOT NULL, zone_id TEXT NOT NULL,
    flood_risk REAL NOT NULL, heat_risk REAL NOT NULL, vulnerability REAL NOT NULL,
    uncertainty REAL NOT NULL, confidence REAL NOT NULL, reliability REAL NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS rl_recommendations (
    id INTEGER PRIMARY KEY, run_id TEXT NOT NULL, zone_id TEXT NOT NULL,
    action TEXT NOT NULL, cost REAL NOT NULL, maintenance_cost REAL NOT NULL DEFAULT 0,
    lifecycle_cost REAL NOT NULL DEFAULT 0, flood_reduction REAL NOT NULL,
    heat_reduction REAL NOT NULL, carbon_tonnes REAL NOT NULL,
    green_gain REAL NOT NULL, scenario TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS scenarios (
    id INTEGER PRIMARY KEY, name TEXT NOT NULL, parameters_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY, run_id TEXT NOT NULL, kind TEXT NOT NULL,
    metrics_json TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


@contextmanager
def connection():
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def initialize():
    import csv
    import json

    with connection() as conn:
        conn.executescript(SCHEMA)
        recommendation_columns = {row[1] for row in conn.execute("PRAGMA table_info(rl_recommendations)")}
        for column in ("maintenance_cost", "lifecycle_cost"):
            if column not in recommendation_columns:
                conn.execute(f"ALTER TABLE rl_recommendations ADD COLUMN {column} REAL NOT NULL DEFAULT 0")
        count = conn.execute("SELECT COUNT(*) FROM zones").fetchone()[0]
        if count:
            active_dataset = conn.execute(
                "SELECT d.name, d.source_type FROM datasets d JOIN zones z ON z.dataset_id=d.id "
                "ORDER BY d.id DESC LIMIT 1"
            ).fetchone()
            if (active_dataset is None or active_dataset["name"] != "Demonstration City (simulated)"
                    or active_dataset["source_type"] != "seeded_simulation"):
                return
            conn.execute("DELETE FROM climate_observations")
            conn.execute("DELETE FROM zones")
        rows = list(csv.DictReader(SAMPLE_DATA.open(encoding="utf-8", newline="")))
        cursor = conn.execute(
            "INSERT INTO datasets(name, source_type, row_count) VALUES (?, ?, ?)",
            ("Chennai south study area (demonstration)", "seeded_simulation", len(rows)),
        )
        dataset_id = cursor.lastrowid
        numeric = ("elevation_m", "rainfall_mm", "temperature_c", "humidity_pct",
                   "vegetation_pct", "building_density", "drainage_capacity",
                   "impervious_pct", "flood_history")
        for row in rows:
            features = {key: (float(row[key]) if key in numeric else row[key]) for key in row if key not in ("zone_id", "name", "latitude", "longitude")}
            conn.execute(
                "INSERT INTO zones VALUES (?, ?, ?, ?, ?, ?)",
                (row["zone_id"], dataset_id, row["name"], float(row["latitude"]),
                 float(row["longitude"]), json.dumps(features)),
            )
            for step in range(6):
                phase = (step - 2.5) / 10
                conn.execute(
                    "INSERT INTO climate_observations(zone_id, timestep, rainfall_mm, temperature_c, humidity_pct) VALUES (?, ?, ?, ?, ?)",
                    (row["zone_id"], step, float(row["rainfall_mm"]) * (1 + phase),
                     float(row["temperature_c"]) + phase * 2.4,
                     min(99, max(20, float(row["humidity_pct"]) + phase * 9))),
                )


def get_zones():
    import json

    with connection() as conn:
        return [dict(row) | {"features": json.loads(row["features_json"])} for row in conn.execute(
            "SELECT zone_id, name, latitude, longitude, features_json FROM zones ORDER BY zone_id"
        )]