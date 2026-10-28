import json
import random
import uuid

import numpy as np

from . import database
from .config import RANDOM_SEED
from .predictor import RiskPredictor, proxy_targets
from .rl.env import INTERVENTIONS
from .rl.ppo import PPOPlanner

SCENARIOS = ["Baseline", "Heavy rainfall", "Extreme rainfall", "Heatwave", "Compound stress"]
DEFAULT_WEIGHTS = {"flood": .4, "heat": .35, "carbon": .15, "green": .1}


def _normalize_weights(weights):
    values = {key: max(0.0, float(weights.get(key, default))) for key, default in DEFAULT_WEIGHTS.items()}
    total = sum(values.values())
    return {key: value / total for key, value in values.items()} if total else DEFAULT_WEIGHTS.copy()


class PlanningService:
    def __init__(self):
        database.initialize()
        self.zones = database.get_zones()
        self.predictor = RiskPredictor()
        self.predictor.fit(self.zones)
        self.latest_run = None
        self.latest_plan = None
        self._restore_latest_plan()

    def refresh(self):
        model_type = self.predictor.model_type
        self.zones = database.get_zones()
        self.predictor = RiskPredictor(model_type=model_type)
        self.predictor.fit(self.zones)
        self.latest_run = None
        self.latest_plan = None
        self._restore_latest_plan()

    @staticmethod
    def dataset_info():
        with database.connection() as conn:
            row = conn.execute(
                "SELECT id, name, source_type, row_count, created_at FROM datasets ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row else {"id": None, "name": "Unavailable", "source_type": "unknown", "row_count": 0}

    def _restore_latest_plan(self):
        dataset = self.dataset_info()
        with database.connection() as conn:
            row = conn.execute(
                "SELECT run_id, metrics_json FROM experiments WHERE kind='ppo_optimization' ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row is None:
                return
            metrics = json.loads(row["metrics_json"])
            if (metrics.get("dataset", {}).get("id") != dataset["id"]
                    or metrics.get("prediction_metrics", {}).get("model_type") != self.predictor.model_type):
                return
            stored_predictions = [dict(item) for item in conn.execute(
                "SELECT zone_id, flood_risk, heat_risk, vulnerability, uncertainty, confidence, reliability FROM model_predictions WHERE run_id=?",
                (row["run_id"],),
            )]
        for prediction in stored_predictions:
            value = prediction["vulnerability"]
            prediction["risk_band"] = "High" if value >= .68 else "Elevated" if value >= .48 else "Moderate" if value >= .3 else "Lower"
        self.latest_run = row["run_id"]
        self.latest_plan = {**metrics, "run_id": row["run_id"], "zones": self.zones,
                            "predictions": stored_predictions}

    def predictions(self, scenario="Baseline", draws=32, persist=True):
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario. Choose one of: {', '.join(SCENARIOS)}")
        values = self.predictor.predict(self.zones, scenario, draws)
        run_id = str(uuid.uuid4())
        if persist:
            with database.connection() as conn:
                conn.executemany(
                    "INSERT INTO model_predictions(run_id, zone_id, flood_risk, heat_risk, vulnerability, uncertainty, confidence, reliability) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    [(run_id, item["zone_id"], item["flood_risk"], item["heat_risk"], item["vulnerability"], item["uncertainty"], item["confidence"], item["reliability"]) for item in values],
                )
        return values, run_id

    @staticmethod
    def _plan_metrics(plan, predictions, zones):
        by_zone = {item["zone_id"]: item for item in predictions}
        green_before = np.mean([zone["features"]["vegetation_pct"] for zone in zones])
        flood_before = float(np.mean([item["flood_risk"] for item in predictions]))
        heat_before = float(np.mean([item["heat_risk"] for item in predictions]))
        carbon = installation_cost = maintenance_cost = flood_delta = heat_delta = green_gain = 0.0
        details = []
        for action in plan:
            prediction = by_zone[action["zone_id"]]
            flood_effect = prediction["flood_risk"] * action["flood"]
            heat_effect = prediction["heat_risk"] * action["heat"]
            installation_cost += action["cost"]
            maintenance_cost += action.get("maintenance_cost", 0)
            carbon += action["carbon"]
            flood_delta += flood_effect
            heat_delta += heat_effect
            green_gain += action["green"]
            details.append({
                **action,
                "installation_cost": action["cost"],
                "lifecycle_cost": action["cost"] + action.get("maintenance_cost", 0),
                "flood_reduction": flood_effect,
                "heat_reduction": heat_effect,
                "carbon_tonnes": action["carbon"],
                "green_gain_pct_points": action["green"],
                "expected_benefit": f"Flood risk -{flood_effect:.1%}; heat risk -{heat_effect:.1%}",
                "prediction_confidence": prediction["confidence"],
                "uncertainty": prediction["uncertainty"],
            })
        flood_after = max(0.0, flood_before - flood_delta / max(len(zones), 1))
        heat_after = max(0.0, heat_before - heat_delta / max(len(zones), 1))
        green_after = min(100.0, green_before + green_gain / max(len(zones), 1))
        resilience_before = 100 * (1 - .55 * flood_before - .45 * heat_before)
        resilience_after = 100 * (1 - .55 * flood_after - .45 * heat_after)
        return {
            "interventions": details, "installation_cost": installation_cost,
            "maintenance_cost": maintenance_cost, "total_cost": installation_cost + maintenance_cost,
            "carbon_tonnes": carbon,
            "flood_reduction": flood_delta / max(len(zones), 1),
            "heat_reduction": heat_delta / max(len(zones), 1),
            "flood_reduction_pct": flood_delta / max(flood_before * len(zones), 1e-8) * 100,
            "heat_reduction_pct": heat_delta / max(heat_before * len(zones), 1e-8) * 100,
            "green_gain_pct_points": green_gain / max(len(zones), 1),
            "green_coverage_before": float(green_before), "green_coverage_after": float(green_after),
            "flood_before": flood_before, "flood_after": flood_after,
            "heat_before": heat_before, "heat_after": heat_after,
            "resilience_before": float(resilience_before), "resilience_after": float(resilience_after),
            "intervention_count": len(details),
        }

    def _greedy_plan(self, predictions, budget, weights, green_target):
        by_zone = {item["zone_id"]: item for item in predictions}
        candidates = []
        for zone in self.zones:
            p = by_zone[zone["zone_id"]]
            for name, item in list(INTERVENTIONS.items())[1:]:
                benefit = (weights["flood"] * p["flood_risk"] * item["flood"]
                           + weights["heat"] * p["heat_risk"] * item["heat"]
                           + weights["carbon"] * item["carbon"] / 30
                           + weights["green"] * min(item["green"], max(0, green_target - zone["features"]["vegetation_pct"])) / max(green_target, 1))
                lifecycle_cost = item["cost"] + item["maintenance_cost"]
                candidates.append((benefit / lifecycle_cost, zone, name, item))
        candidates.sort(key=lambda row: row[0], reverse=True)
        used_zones, plan = set(), []
        remaining = budget
        for _, zone, name, item in candidates:
            lifecycle_cost = item["cost"] + item["maintenance_cost"]
            if zone["zone_id"] not in used_zones and lifecycle_cost <= remaining:
                plan.append({"zone_id": zone["zone_id"], "action": name, **item})
                used_zones.add(zone["zone_id"])
                remaining -= lifecycle_cost
        return plan

    def _random_plan(self, budget, seed):
        rng = random.Random(seed)
        zones = list(self.zones)
        rng.shuffle(zones)
        remaining, plan = budget, []
        for zone in zones:
            candidates = [(name, item) for name, item in INTERVENTIONS.items()
                          if item["cost"] + item["maintenance_cost"] <= remaining]
            name, item = rng.choice(candidates)
            if name != "No intervention":
                plan.append({"zone_id": zone["zone_id"], "action": name, **item})
                remaining -= item["cost"] + item["maintenance_cost"]
        return plan

    def optimize(self, budget=500000, weights=None, scenario="Baseline", episodes=90, green_target=35):
        budget = float(budget)
        if not 0 <= budget <= 50_000_000:
            raise ValueError("Budget must be between $0 and $50,000,000.")
        if not 1 <= int(episodes) <= 300:
            raise ValueError("PPO training episodes must be between 1 and 300.")
        weights = _normalize_weights(weights or DEFAULT_WEIGHTS)
        green_target = float(green_target)
        if not 1 <= green_target <= 100:
            raise ValueError("Green coverage target must be between 1% and 100%.")
        predictions, run_id = self.predictions(scenario, persist=True)
        planner = PPOPlanner(RANDOM_SEED)
        planner.train(self.zones, predictions, budget, weights, scenario, int(episodes), green_target)
        rl_plan = planner.plan(self.zones, predictions, budget, weights, scenario, green_target=green_target)
        if not rl_plan:
            rl_plan = self._greedy_plan(predictions, budget, weights, green_target)
            policy_note = "PPO produced no feasible non-zero plan; displayed the transparent risk-weighted greedy fallback."
        else:
            policy_note = "Plan selected by the trained PPO policy; benefits use stated planning assumptions."
        methods = {
            "PPO (proposed)": rl_plan,
            "Risk-weighted greedy": self._greedy_plan(predictions, budget, weights, green_target),
            "Seeded random": self._random_plan(budget, RANDOM_SEED),
        }
        comparisons = {name: self._plan_metrics(plan, predictions, self.zones) for name, plan in methods.items()}
        results = dict(comparisons["PPO (proposed)"])
        results.update({
            "run_id": run_id, "scenario": scenario, "budget": budget,
            "budget_remaining": max(0, budget - results["total_cost"]),
            "green_target": float(green_target), "weights": weights,
            "method": "PPO (proposed)", "policy_note": policy_note,
            "comparisons": comparisons,
            "training_rewards": planner.reward_history,
            "training_reward_last_10_mean": float(np.mean(planner.reward_history[-10:])),
            "prediction_metrics": self.predictor.metrics,
            "dataset": self.dataset_info(),
            "zones": self.zones, "predictions": predictions,
        })
        with database.connection() as conn:
            conn.executemany(
                "INSERT INTO rl_recommendations(run_id, zone_id, action, cost, maintenance_cost, lifecycle_cost, flood_reduction, heat_reduction, carbon_tonnes, green_gain, scenario) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [(run_id, item["zone_id"], item["action"], item["installation_cost"], item["maintenance_cost"], item["lifecycle_cost"], item["flood_reduction"], item["heat_reduction"], item["carbon_tonnes"], item["green_gain_pct_points"], scenario) for item in results["interventions"]],
            )
            conn.execute("INSERT INTO scenarios(name, parameters_json) VALUES (?, ?)", (scenario, json.dumps({"budget": budget, "weights": weights, "green_target": green_target})))
            saved = {key: value for key, value in results.items() if key not in ("zones", "predictions")}
            conn.execute("INSERT INTO experiments(run_id, kind, metrics_json) VALUES (?, ?, ?)", (run_id, "ppo_optimization", json.dumps(saved)))
        self.latest_run, self.latest_plan = run_id, results
        return results

    def explain(self, zone_id):
        zone = next((zone for zone in self.zones if zone["zone_id"] == zone_id), None)
        if zone is None:
            raise ValueError("Unknown zone identifier.")
        predictions, _ = self.predictions(persist=False)
        prediction = next(item for item in predictions if item["zone_id"] == zone_id)
        contributions = self.predictor.explain(self.zones, zone_id)
        top = contributions[:4]
        recommendation = None
        if self.latest_plan:
            recommendation = next((item for item in self.latest_plan["interventions"] if item["zone_id"] == zone_id), None)
        return {"zone": zone, "prediction": prediction, "contributions": contributions,
                "top_factors": top, "recommendation": recommendation,
            "method": ("Integrated gradients over the trained GNN; proxy labels and scenario outputs are simulated."
                   if self.predictor.model_type == "spatiotemporal_gnn"
                   else "One-feature replacement sensitivity for the selected tree ensemble; not causal.")}

    def ablations(self):
        zones = self.zones
        if self.predictor.model_type != "spatiotemporal_gnn":
            predictions = self.predictor.predict(zones, draws=96)
            target = proxy_targets(zones)
            values = np.asarray([[item[key] for key in ("flood_risk", "heat_risk", "vulnerability")] for item in predictions])
            residual = values - target
            selected = {"mae": float(np.abs(residual).mean()), "rmse": float(np.sqrt(np.square(residual).mean()))}
            output = [{"name": self.predictor.metrics["model_name"], **selected},
                      {"name": "Without uncertainty (deterministic)", **selected, "uncertainty": 0.0}]
            if self.latest_plan:
                baseline = self.latest_plan["comparisons"]["Risk-weighted greedy"]
                output.append({"name": "Without RL (risk-weighted greedy)", **selected,
                               "cost": baseline["total_cost"], "resilience_after": baseline["resilience_after"]})
            return output
        sequence, adjacency = self.predictor._prepare(zones, "Baseline")
        target = proxy_targets(zones)
        import torch
        variants = {"Full spatio-temporal GNN": (sequence, adjacency),
                "Without graph aggregation": (sequence, torch.eye(len(zones), device=self.predictor.device)),
                    "Without temporal sequence": (sequence[:, -1:, :], adjacency)}
        output = []
        self.predictor.model.eval()
        with torch.no_grad():
            for name, (values, graph) in variants.items():
                prediction = self.predictor.model(values, graph).cpu().numpy()
                output.append({"name": name, "mae": float(np.abs(prediction - target).mean()),
                               "rmse": float(np.sqrt(np.square(prediction - target).mean()))})
        output.append({"name": "Without uncertainty (deterministic)",
                       "mae": output[0]["mae"], "rmse": output[0]["rmse"], "uncertainty": 0.0})
        if self.latest_plan:
            baseline = self.latest_plan["comparisons"]["Risk-weighted greedy"]
            output.append({"name": "Without RL (risk-weighted greedy)", "mae": output[0]["mae"],
                           "rmse": output[0]["rmse"], "cost": baseline["total_cost"],
                           "resilience_after": baseline["resilience_after"]})
        return output

    def history(self):
        with database.connection() as conn:
            return [dict(row) for row in conn.execute(
                "SELECT run_id, kind, metrics_json, created_at FROM experiments ORDER BY id DESC LIMIT 20")]