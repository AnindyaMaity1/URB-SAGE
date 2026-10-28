import random

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from torch import nn

from .config import RANDOM_SEED
from .graph import build_adjacency

FEATURES = ("rainfall_mm", "temperature_c", "humidity_pct", "elevation_m",
            "vegetation_pct", "building_density", "drainage_capacity", "impervious_pct", "flood_history")
MODEL_NAMES = {
    "spatiotemporal_gnn": "Spatio-temporal GNN",
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
}


class SpatioTemporalGNN(nn.Module):
    """Graph message passing at each time step followed by a temporal GRU."""

    def __init__(self, feature_count, hidden_dim=24, dropout=0.2):
        super().__init__()
        self.graph_layer = nn.Linear(feature_count * 2, hidden_dim)
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 3)

    def forward(self, sequence, adjacency):
        graph_steps = []
        for timestep in range(sequence.shape[1]):
            values = sequence[:, timestep, :]
            neighbors = adjacency @ values
            graph_steps.append(torch.relu(self.graph_layer(torch.cat([values, neighbors], dim=-1))))
        temporal, _ = self.temporal(torch.stack(graph_steps, dim=1))
        return torch.sigmoid(self.head(self.dropout(temporal[:, -1, :])))


def proxy_targets(zones, scenario="Baseline"):
    targets = []
    rain_multiplier, heat_add = {
        "Baseline": (1.0, 0.0), "Heavy rainfall": (1.25, 0.0),
        "Extreme rainfall": (1.55, 0.0), "Heatwave": (1.0, 3.5),
        "Compound stress": (1.35, 2.4),
    }.get(scenario, (1.0, 0.0))
    for zone in zones:
        f = zone["features"]
        flood_linear = (-2.25 + 0.020 * f["rainfall_mm"] * rain_multiplier
                        - 0.025 * f["elevation_m"] - 1.05 * f["drainage_capacity"]
                        + 0.018 * f["impervious_pct"] + 0.70 * f["flood_history"])
        heat_linear = (-3.0 + 0.105 * (f["temperature_c"] + heat_add)
                       + 1.55 * f["building_density"] - 0.025 * f["vegetation_pct"]
                       + 0.006 * f["impervious_pct"])
        flood = 1 / (1 + np.exp(-flood_linear))
        heat = 1 / (1 + np.exp(-heat_linear))
        targets.append([flood, heat, 0.55 * flood + 0.45 * heat])
    return np.asarray(targets, dtype=np.float32)


class RiskPredictor:
    def __init__(self, epochs=110, hidden_dim=24, learning_rate=0.012, model_type="spatiotemporal_gnn"):
        if model_type not in MODEL_NAMES:
            raise ValueError(f"Unsupported model type: {model_type}")
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)
        torch.set_num_threads(1)
        self.epochs = epochs
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SpatioTemporalGNN(len(FEATURES), hidden_dim).to(self.device) if model_type == "spatiotemporal_gnn" else None
        self.tree_model = None
        self.mean = None
        self.std = None
        self.metrics = {}

    def _prepare(self, zones, scenario):
        base = np.asarray([[zone["features"][key] for key in FEATURES] for zone in zones], dtype=np.float32)
        series = []
        for step in range(6):
            adjusted = base.copy()
            adjusted[:, 0] *= 1 + (step - 2.5) * 0.018
            adjusted[:, 1] += (step - 2.5) * 0.11
            adjusted[:, 2] += (step - 2.5) * 0.3
            if scenario in ("Heavy rainfall", "Extreme rainfall", "Compound stress"):
                adjusted[:, 0] *= {"Heavy rainfall": 1.25, "Extreme rainfall": 1.55, "Compound stress": 1.35}[scenario]
            if scenario in ("Heatwave", "Compound stress"):
                adjusted[:, 1] += {"Heatwave": 3.5, "Compound stress": 2.4}[scenario]
            series.append(adjusted)
        sequence = np.stack(series, axis=1)
        if self.mean is None:
            self.mean = sequence.mean(axis=(0, 1), keepdims=True)
            self.std = sequence.std(axis=(0, 1), keepdims=True) + 1e-6
        return (torch.tensor((sequence - self.mean) / self.std, device=self.device),
            torch.tensor(build_adjacency(zones), device=self.device))

    def fit(self, zones, epochs=None):
        sequence, adjacency = self._prepare(zones, "Baseline")
        target_values = proxy_targets(zones)
        if self.model_type == "spatiotemporal_gnn":
            target = torch.tensor(target_values, device=self.device)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.model.train()
            for _ in range(epochs or self.epochs):
                optimizer.zero_grad()
                loss = nn.functional.mse_loss(self.model(sequence, adjacency), target)
                loss.backward()
                optimizer.step()
            self.model.eval()
            with torch.no_grad():
                fit = self.model(sequence, adjacency).cpu().numpy()
        else:
            matrix = sequence.mean(dim=1).cpu().numpy()
            estimator = RandomForestRegressor if self.model_type == "random_forest" else ExtraTreesRegressor
            self.tree_model = estimator(n_estimators=96, max_depth=6, min_samples_leaf=1,
                                        random_state=RANDOM_SEED, n_jobs=1)
            self.tree_model.fit(matrix, target_values)
            fit = self.tree_model.predict(matrix)
        residual = target_values - fit
        self.metrics = {
            "mae": float(np.abs(residual).mean()),
            "rmse": float(np.sqrt(np.square(residual).mean())),
            "r2_proxy_fit": float(1 - np.square(residual).sum() / (np.square(target_values - target_values.mean()).sum() + 1e-8)),
            "model_type": self.model_type,
            "model_name": MODEL_NAMES[self.model_type],
            "label_note": "In-sample fit to transparent synthetic proxy labels; not field validation.",
        }

    def predict(self, zones, scenario="Baseline", draws=32):
        sequence, adjacency = self._prepare(zones, scenario)
        if self.model_type == "spatiotemporal_gnn":
            self.model.train()
            samples = []
            with torch.no_grad():
                for _ in range(draws):
                    samples.append(self.model(sequence, adjacency).cpu().numpy())
            self.model.eval()
            stack = np.stack(samples)
        else:
            matrix = sequence.mean(dim=1).cpu().numpy()
            stack = np.stack([tree.predict(matrix) for tree in self.tree_model.estimators_])
        means = stack.mean(axis=0)
        uncertainty = stack.std(axis=0).mean(axis=1)
        result = []
        for zone, values, unc in zip(zones, means, uncertainty):
            uncertainty_score = float(np.clip(unc * 3.5, 0, 1))
            result.append({
                "zone_id": zone["zone_id"], "flood_risk": float(values[0]),
                "heat_risk": float(values[1]), "vulnerability": float(values[2]),
                "uncertainty": uncertainty_score,
                "confidence": float(1 - uncertainty_score),
                "reliability": float(max(0, 1 - uncertainty_score * 1.15)),
                "risk_band": "High" if values[2] >= .68 else "Elevated" if values[2] >= .48 else "Moderate" if values[2] >= .3 else "Lower",
            })
        return result

    def explain(self, zones, zone_id, steps=24):
        sequence, adjacency = self._prepare(zones, "Baseline")
        index = next(i for i, zone in enumerate(zones) if zone["zone_id"] == zone_id)
        if self.model_type != "spatiotemporal_gnn":
            matrix = sequence.mean(dim=1).cpu().numpy()
            baseline_risk = self.tree_model.predict(matrix[[index]])[0, 2]
            contributions = []
            for feature_index, feature in enumerate(FEATURES):
                perturbed = matrix[[index]].copy()
                perturbed[0, feature_index] = 0
                delta = baseline_risk - self.tree_model.predict(perturbed)[0, 2]
                contributions.append((feature, float(delta)))
            contributions.sort(key=lambda item: abs(item[1]), reverse=True)
            return [{"feature": feature, "contribution": value} for feature, value in contributions]
        baseline = torch.zeros_like(sequence)
        total_gradient = torch.zeros_like(sequence)
        self.model.eval()
        for alpha in torch.linspace(0, 1, steps):
            sample = (baseline + alpha * (sequence - baseline)).detach().requires_grad_(True)
            output = self.model(sample, adjacency)[index].sum()
            gradient, = torch.autograd.grad(output, sample)
            total_gradient += gradient.detach()
        attribution = ((sequence - baseline) * total_gradient / steps).mean(dim=(0, 1)).cpu().numpy()
        ranking = sorted(zip(FEATURES, attribution.tolist()), key=lambda pair: abs(pair[1]), reverse=True)
        return [{"feature": feature, "contribution": float(value)} for feature, value in ranking]