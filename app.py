import io
import json
import logging
import os
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, make_response, render_template, request, send_file

load_dotenv()

from climate_resilience import database
from climate_resilience.config import RANDOM_SEED, RESULTS_DIR
from climate_resilience.graph import build_adjacency
from climate_resilience.service import PlanningService, SCENARIOS
from climate_resilience.predictor import FEATURES, MODEL_NAMES

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
service = PlanningService()


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/api/health")
def health():
    dataset = service.dataset_info()
    return jsonify({"status": "ready", "zones": len(service.zones), "device": str(service.predictor.device),
                    "dataset": dataset, "data_label": dataset["source_type"],
                    "model_type": service.predictor.model_type, "models": MODEL_NAMES})


@app.get("/api/overview")
def overview():
    scenario = request.args.get("scenario", "Baseline")
    predictions, run_id = service.predictions(scenario)
    return jsonify({"zones": service.zones, "predictions": predictions, "run_id": run_id,
                    "metrics": service.predictor.metrics, "scenario": scenario,
                    "latest_plan": service.latest_plan, "dataset": service.dataset_info(),
                    "computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})


@app.get("/api/live")
def live_refresh():
    scenario = request.args.get("scenario", "Baseline")
    predictions, _ = service.predictions(scenario, persist=False)
    return jsonify({"predictions": predictions, "metrics": service.predictor.metrics,
                    "scenario": scenario, "dataset": service.dataset_info(),
                    "computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})


@app.get("/api/graph")
def graph():
    adjacency = build_adjacency(service.zones)
    edges = [[service.zones[i]["zone_id"], service.zones[j]["zone_id"]]
             for i in range(len(service.zones)) for j in range(i + 1, len(service.zones))
             if adjacency[i, j] > 0]
    return jsonify({"nodes": service.zones, "edges": edges,
                    "method": "Symmetric 3-nearest-neighbor graph with normalized adjacency from supplied coordinates."})


@app.post("/api/preprocess")
def preprocess():
    matrix = np.asarray([[float(zone["features"][key]) for key in FEATURES] for zone in service.zones], dtype=np.float32)
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0) + 1e-6
    normalized = (matrix - means) / scales
    return jsonify({
        "method": "Per-feature z-score normalization; population standard deviation with epsilon floor.",
        "features": list(FEATURES),
        "zones": [{"zone_id": zone["zone_id"], "normalized_features": dict(zip(FEATURES, row.tolist()))}
                  for zone, row in zip(service.zones, normalized)],
    })


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    predictions, run_id = service.predictions(payload.get("scenario", "Baseline"), int(payload.get("draws", 32)))
    return jsonify({"run_id": run_id, "predictions": predictions, "metrics": service.predictor.metrics})


@app.post("/api/optimize")
def optimize():
    payload = request.get_json(silent=True) or {}
    try:
        result = service.optimize(
            budget=payload.get("budget", 500000),
            weights={"flood": payload.get("flood_weight", .4), "heat": payload.get("heat_weight", .35),
                     "carbon": payload.get("carbon_weight", .15), "green": payload.get("green_weight", .1)},
            scenario=payload.get("scenario", "Baseline"), episodes=payload.get("episodes", 90),
            green_target=payload.get("green_target", 35),
        )
        return jsonify(result)
    except (TypeError, ValueError) as error:
        return jsonify({"error": str(error)}), 400


@app.get("/api/scenarios")
def scenarios():
    result = []
    for scenario in SCENARIOS:
        values, _ = service.predictions(scenario, draws=20, persist=False)
        result.append({"scenario": scenario, "flood_risk": sum(x["flood_risk"] for x in values) / len(values),
                       "heat_risk": sum(x["heat_risk"] for x in values) / len(values),
                       "vulnerability": sum(x["vulnerability"] for x in values) / len(values),
                       "uncertainty": sum(x["uncertainty"] for x in values) / len(values)})
    return jsonify(result)


@app.get("/api/explain/<zone_id>")
def explain(zone_id):
    try:
        return jsonify(service.explain(zone_id))
    except (ValueError, StopIteration) as error:
        return jsonify({"error": str(error)}), 404


@app.get("/api/analytics")
def analytics():
    return jsonify({"model": service.predictor.metrics, "ablations": service.ablations(),
                    "history": service.history(), "latest_plan": service.latest_plan})


@app.get("/api/figures")
def publication_figures():
    try:
        from climate_resilience.figures import generate_publication_figures
        scenario = request.args.get("scenario", "Baseline")
        return jsonify(generate_publication_figures(service, scenario))
    except Exception as error:
        app.logger.exception("Publication figure generation failed")
        return jsonify({"error": str(error)}), 500


@app.get("/api/figures/<filename>")
def publication_figure_file(filename):
    stems = [f"{index:02d}_{name}" for index, name in enumerate((
        "zone_risk_map", "zone_risk_profiles", "uncertainty_distribution",
        "scenario_comparison", "model_fit_diagnostics", "before_after_risk",
        "lifecycle_costs", "feature_sensitivity"), start=1)]
    allowed = {f"{stem}.{extension}" for stem in stems for extension in ("png", "pdf")}
    if filename not in allowed:
        return jsonify({"error": "Unknown figure filename."}), 404
    path = RESULTS_DIR / "figures" / filename
    if not path.is_file():
        publication_figures()
    if not path.is_file():
        return jsonify({"error": "Figure was not generated."}), 404
    mimetype = "image/png" if path.suffix == ".png" else "application/pdf"
    return send_file(path, mimetype=mimetype, as_attachment=request.args.get("download") == "1",
                     download_name=filename, conditional=True, max_age=0)


@app.get("/api/history")
def history():
    return jsonify(service.history())


@app.get("/api/recommendations")
def recommendations():
    if service.latest_plan is None:
        return jsonify({"recommendations": [], "message": "No saved plan exists for the active dataset."})
    return jsonify({"run_id": service.latest_run, "scenario": service.latest_plan["scenario"],
                    "recommendations": service.latest_plan["interventions"],
                    "metrics": {key: service.latest_plan[key] for key in ("total_cost", "installation_cost", "maintenance_cost", "carbon_tonnes", "resilience_after")}})


@app.post("/api/upload")
def upload():
    upload_file = request.files.get("file")
    if upload_file is None:
        return jsonify({"error": "Choose a CSV file to upload."}), 400
    required = {"zone_id", "name", "latitude", "longitude", "elevation_m", "rainfall_mm",
                "temperature_c", "humidity_pct", "vegetation_pct", "building_density",
                "drainage_capacity", "impervious_pct", "flood_history", "land_use"}
    try:
        suffix = Path(upload_file.filename or "").suffix.lower()
        raw = upload_file.read()
        if suffix == ".csv":
            frame = pd.read_csv(io.BytesIO(raw))
        elif suffix == ".xlsx":
            frame = pd.read_excel(io.BytesIO(raw))
        elif suffix in (".geojson", ".json"):
            document = json.loads(raw)
            features = document.get("features", []) if document.get("type") == "FeatureCollection" else [document]
            records = []
            for feature in features:
                properties = dict(feature.get("properties") or {})
                geometry = feature.get("geometry") or {}
                coordinates = geometry.get("coordinates", [])
                pairs = []
                def collect_pairs(value):
                    if isinstance(value, (list, tuple)) and len(value) >= 2 and all(isinstance(item, (int, float)) for item in value[:2]):
                        pairs.append(value[:2])
                    elif isinstance(value, (list, tuple)):
                        for child in value:
                            collect_pairs(child)
                collect_pairs(coordinates)
                if pairs:
                    properties.setdefault("longitude", sum(pair[0] for pair in pairs) / len(pairs))
                    properties.setdefault("latitude", sum(pair[1] for pair in pairs) / len(pairs))
                records.append(properties)
            frame = pd.DataFrame.from_records(records)
        else:
            raise ValueError("Supported upload formats are CSV, XLSX and GeoJSON.")
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError("Missing required columns: " + ", ".join(missing))
        if frame.empty or len(frame) > 5000:
            raise ValueError("Dataset must contain 1 to 5,000 zone rows.")
        if frame["zone_id"].duplicated().any():
            raise ValueError("zone_id values must be unique.")
        numeric = list(required - {"zone_id", "name", "land_use"})
        frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="raise")
        if not frame[numeric].apply(lambda col: col.map(lambda value: pd.notna(value) and abs(value) < 1e9)).all().all():
            raise ValueError("Numeric columns must contain finite values within supported bounds.")
        if not frame["latitude"].between(-90, 90).all() or not frame["longitude"].between(-180, 180).all():
            raise ValueError("Latitude/longitude values are outside geographic bounds.")
        with database.connection() as conn:
            conn.execute("DELETE FROM climate_observations")
            conn.execute("DELETE FROM zones")
            cursor = conn.execute("INSERT INTO datasets(name, source_type, row_count) VALUES (?, ?, ?)",
                                  (upload_file.filename or "Uploaded dataset", "user_upload", len(frame)))
            dataset_id = cursor.lastrowid
            feature_columns = sorted(required - {"zone_id", "name", "latitude", "longitude"})
            for row in frame.to_dict(orient="records"):
                features = {key: row[key] for key in feature_columns}
                conn.execute("INSERT INTO zones VALUES (?, ?, ?, ?, ?, ?)",
                             (str(row["zone_id"]), dataset_id, str(row["name"]), float(row["latitude"]),
                              float(row["longitude"]), json.dumps(features)))
                for step in range(6):
                    conn.execute("INSERT INTO climate_observations(zone_id, timestep, rainfall_mm, temperature_c, humidity_pct) VALUES (?, ?, ?, ?, ?)",
                                 (str(row["zone_id"]), step, float(row["rainfall_mm"]), float(row["temperature_c"]), float(row["humidity_pct"])))
        service.refresh()
        return jsonify({"status": "loaded", "zones": len(service.zones), "dataset": upload_file.filename,
                        "note": "Uploaded values are treated as user-provided data; no field validation is implied."})
    except (ValueError, pd.errors.ParserError, KeyError, json.JSONDecodeError) as error:
        return jsonify({"error": str(error)}), 400


@app.post("/api/train")
def train():
    payload = request.get_json(silent=True) or {}
    try:
        epochs = int(payload.get("epochs", 110))
        hidden_dim = int(payload.get("hidden_dim", 24))
        learning_rate = float(payload.get("learning_rate", .012))
        model_type = payload.get("model_type", "spatiotemporal_gnn")
        if not 10 <= epochs <= 500 or not 8 <= hidden_dim <= 128:
            raise ValueError("Epochs must be 10-500 and hidden dimensions 8-128.")
        if not 0.0001 <= learning_rate <= .1:
            raise ValueError("Learning rate must be between 0.0001 and 0.1.")
        if model_type not in MODEL_NAMES:
            raise ValueError("Choose a supported model from the model registry.")
        from climate_resilience.predictor import RiskPredictor
        service.predictor = RiskPredictor(epochs=epochs, hidden_dim=hidden_dim,
                                          learning_rate=learning_rate, model_type=model_type)
        service.predictor.fit(service.zones, epochs)
        service.latest_run = None
        service.latest_plan = None
        return jsonify({"status": "trained", "metrics": service.predictor.metrics,
                        "epochs": epochs, "hidden_dim": hidden_dim, "model_type": model_type,
                        "seed": RANDOM_SEED})
    except (TypeError, ValueError) as error:
        return jsonify({"error": str(error)}), 400


@app.get("/api/report/<run_id>")
def report(run_id):
    with database.connection() as conn:
        experiment = conn.execute("SELECT metrics_json FROM experiments WHERE run_id=?", (run_id,)).fetchone()
        if experiment is None:
            return jsonify({"error": "Report not found."}), 404
        predictions = [dict(row) for row in conn.execute("SELECT zone_id, flood_risk, heat_risk, vulnerability, uncertainty, confidence, reliability FROM model_predictions WHERE run_id=?", (run_id,))]
        interventions = [dict(row) for row in conn.execute("SELECT zone_id, action, cost, maintenance_cost, lifecycle_cost, flood_reduction, heat_reduction, carbon_tonnes, green_gain, scenario FROM rl_recommendations WHERE run_id=?", (run_id,))]
    metrics = json.loads(experiment["metrics_json"])
    payload = {"run_id": run_id, "data_status": metrics.get("dataset", service.dataset_info()),
               "metrics": metrics, "predictions": predictions,
               "interventions": interventions, "zones": service.zones}
    response = make_response(json.dumps(payload, indent=2))
    response.headers["Content-Type"] = "application/json"
    response.headers["Content-Disposition"] = f"attachment; filename=climate-plan-{run_id[:8]}.json"
    return response


@app.errorhandler(413)
def upload_too_large(_error):
    return jsonify({"error": "Upload exceeds the 8 MB limit."}), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")), debug=False)