import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime, timezone

from .config import RESULTS_DIR
from .service import SCENARIOS

FIGURE_DPI = 1200
FIGURE_WIDTH = 7.2
FIGURE_HEIGHT = 4.8
GREEN = "#205541"
VIOLET = "#50386f"
BLUE = "#255c79"
TEAL = "#176d6b"
WINE = "#79394b"
TEXT = "#263128"
MUTED = "#657266"
GRID = "#e2e7e0"


def _style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "text.color": TEXT,
        "axes.labelcolor": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": "#9da99c",
        "axes.linewidth": .65,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": .55,
        "grid.alpha": .9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def _save(fig, path, title, caption):
    fig.savefig(path, dpi=FIGURE_DPI,
        metadata={"Title": title, "Description": caption,
              "Software": "Canopy research prototype; Matplotlib"})
    pdf_path = path.with_suffix(".pdf")
    fig.savefig(pdf_path, format="pdf", metadata={"Title": title, "Subject": caption,
                           "Creator": "Canopy research prototype; Matplotlib"})
    width, height = fig.get_size_inches()
    plt.close(fig)
    return {"title": title, "caption": caption, "filename": path.name,
            "url": f"/api/figures/{path.name}", "pdf_url": f"/api/figures/{pdf_path.name}",
            "pdf_filename": pdf_path.name, "dpi": FIGURE_DPI,
            "width_px": int(width * FIGURE_DPI), "height_px": int(height * FIGURE_DPI)}


def generate_publication_figures(service, scenario="Baseline"):
    """Render publication-style PNGs from current predictions and saved results."""
    _style()
    output_dir = RESULTS_DIR / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    zones = service.zones
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario. Choose one of: {', '.join(SCENARIOS)}")
    predictions, _ = service.predictions(scenario, draws=64, persist=False)
    by_zone = {item["zone_id"]: item for item in predictions}
    figure_data = []

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), constrained_layout=True)
    cmap = LinearSegmentedColormap.from_list("canopy_risk", [GREEN, BLUE, WINE])
    mean_latitude = float(np.mean([zone["latitude"] for zone in zones]))
    longitude_scale = np.cos(np.deg2rad(mean_latitude))
    scatter = ax.scatter([zone["longitude"] for zone in zones], [zone["latitude"] for zone in zones],
                         c=[by_zone[zone["zone_id"]]["vulnerability"] for zone in zones],
                         s=[38 + by_zone[zone["zone_id"]]["uncertainty"] * 300 for zone in zones],
                         cmap=cmap, vmin=0, vmax=1, edgecolor="white", linewidth=.7, zorder=3)
    for zone in zones:
        ax.annotate(zone["name"], (zone["longitude"], zone["latitude"]), xytext=(4, 4),
                    textcoords="offset points", fontsize=9, color=TEXT)
    fig.colorbar(scatter, ax=ax, label="Predicted vulnerability score", shrink=.83)
    ax.set(title=f"Neighborhood climate vulnerability · Chennai, India · {scenario}",
           xlabel="Longitude (°E)", ylabel="Latitude (°N)")
    ax.set_aspect(1 / longitude_scale)
    ax.text(.01, -.17, "Marker area encodes MC-dropout spread. Coordinates © OpenStreetMap contributors; risk attributes are demonstration inputs.",
            transform=ax.transAxes, fontsize=8, color=MUTED)
    figure_data.append(_save(fig, output_dir / "01_zone_risk_map.png", "Zone-level climate vulnerability",
                             "Predicted vulnerability with marker area proportional to dropout spread; locality coordinates © OpenStreetMap contributors, demonstration risk attributes."))

    ordered = sorted(zones, key=lambda zone: by_zone[zone["zone_id"]]["vulnerability"])
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), constrained_layout=True)
    positions = np.arange(len(ordered))
    bar_height = .23
    for offset, key, label, color in [(-bar_height, "flood_risk", "Flood", TEAL),
                                      (0, "heat_risk", "Heat", "#493d78"),
                                      (bar_height, "vulnerability", "Overall", "#31506d")]:
        ax.barh(positions + offset, [by_zone[zone["zone_id"]][key] for zone in ordered],
                height=bar_height * .92, label=label, color=color, edgecolor="white", linewidth=.35)
    ax.set_yticks(positions, [zone["name"] for zone in ordered])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Predicted risk score (0–1)")
    ax.set_title("Flood, heat and combined vulnerability by zone")
    ax.legend(frameon=False, ncol=3, loc="lower right")
    ax.grid(axis="y", visible=False)
    figure_data.append(_save(fig, output_dir / "02_zone_risk_profiles.png", "Climate risk profiles by zone",
                             "Current model scores for flood, heat and combined vulnerability."))

    uncertainty = np.asarray([by_zone[zone["zone_id"]]["uncertainty"] * 100 for zone in zones])
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), constrained_layout=True)
    upper = max(5.0, float(uncertainty.max()) * 1.08)
    ax.hist(uncertainty, bins=np.linspace(0, upper, 9), color=VIOLET, edgecolor="white", linewidth=.7)
    ax.axvline(float(np.mean(uncertainty)), color=TEAL, linestyle="--", linewidth=1.1,
               label=f"Mean {np.mean(uncertainty):.1f}%")
    ax.set(title="Monte Carlo dropout spread across zones", xlabel="Mean output spread score (%)", ylabel="Zone count")
    ax.legend(frameon=False)
    ax.text(.01, -.17, "Dispersion diagnostic; not a calibrated probability interval.",
            transform=ax.transAxes, fontsize=8, color=MUTED)
    figure_data.append(_save(fig, output_dir / "03_uncertainty_distribution.png", "Uncertainty distribution",
                             "Distribution of MC-dropout spread scores across active zones."))

    scenario_rows = []
    for scenario_name in SCENARIOS:
        values, _ = service.predictions(scenario_name, draws=32, persist=False)
        scenario_rows.append([np.mean([row["flood_risk"] for row in values]),
                              np.mean([row["heat_risk"] for row in values]),
                              np.mean([row["vulnerability"] for row in values])])
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), constrained_layout=True)
    scenario_values = np.asarray(scenario_rows)
    positions = np.arange(len(SCENARIOS))
    bar_width = .23
    for index, label, color in [(0, "Flood", GREEN), (1, "Heat", WINE), (2, "Vulnerability", BLUE)]:
        ax.bar(positions + (index - 1) * bar_width, scenario_values[:, index], width=bar_width,
               color=color, edgecolor="white", linewidth=.4, label=label)
    ax.set_xticks(positions, SCENARIOS, rotation=18, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean predicted score")
    ax.set_title("Climate scenario comparison")
    ax.legend(frameon=False, ncol=3)
    figure_data.append(_save(fig, output_dir / "04_scenario_comparison.png", "Climate scenario comparison",
                             "Mean predicted flood, heat and vulnerability scores under five simulated stress scenarios."))

    metrics = service.predictor.metrics
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), constrained_layout=True)
    labels = ["MAE", "RMSE", "R² proxy fit"]
    values = [metrics.get("mae", 0), metrics.get("rmse", 0), metrics.get("r2_proxy_fit", 0)]
    colors = ["#285741", "#244f70", "#703b65"]
    bars = ax.bar(labels, values, color=colors, width=.52, edgecolor="white", linewidth=.5)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=12)
    ax.axhline(0, color="#7d887d", linewidth=.6)
    ax.set_title(f"Training-fit diagnostics · {service.predictor.metrics.get('model_name', 'selected model')}")
    ax.set_ylabel("Metric value")
    ax.text(.01, -.17, "In-sample generated proxy-label diagnostics; not external validation.",
            transform=ax.transAxes, fontsize=8, color=MUTED)
    figure_data.append(_save(fig, output_dir / "05_model_fit_diagnostics.png", "Model fit diagnostics",
                     "In-sample errors and R² against generated proxy labels; no field-validation claim."))

    plan = service.latest_plan if service.latest_plan and service.latest_plan["scenario"] == scenario else None
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), constrained_layout=True)
    metrics_to_plot = ["flood_before", "heat_before"]
    labels = ["Flood", "Heat"]
    if plan:
        before = [plan["flood_before"], plan["heat_before"]]
        after = [plan["flood_after"], plan["heat_after"]]
        positions = np.arange(2)
        ax.bar(positions - .18, before, width=.34, color=WINE, label="Before")
        ax.bar(positions + .18, after, width=.34, color=GREEN, label="After estimate")
        ax.legend(frameon=False)
        caption = "Before and plan-adjusted zone-average risk using stated intervention-effect assumptions."
    else:
        before = [float(np.mean([row["flood_risk"] for row in predictions])),
                  float(np.mean([row["heat_risk"] for row in predictions]))]
        ax.bar(labels, before, color=[GREEN, VIOLET], width=.48)
        ax.text(.5, .88, "No plan saved for the active dataset", transform=ax.transAxes,
                ha="center", fontsize=8, color=MUTED)
        caption = "Current zone-average baseline risk. Post-plan estimates appear after optimization."
    ax.set_xticks(np.arange(2), labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean predicted risk score")
    ax.set_title("Before / after risk comparison")
    figure_data.append(_save(fig, output_dir / "06_before_after_risk.png", "Before and after comparison", caption))

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), constrained_layout=True)
    if plan and plan["interventions"]:
        selected = plan["interventions"]
        y = np.arange(len(selected))
        install = [row["installation_cost"] / 1000 for row in selected]
        maintenance = [row["maintenance_cost"] / 1000 for row in selected]
        ax.barh(y, install, color=BLUE, label="Installation")
        ax.barh(y, maintenance, left=install, color=VIOLET, label="10-year maintenance")
        names_by_id = {zone["zone_id"]: zone["name"] for zone in zones}
        ax.set_yticks(y, [names_by_id[row["zone_id"]] for row in selected])
        ax.legend(frameon=False)
        caption = "Lifecycle cost components for the saved plan; monetary values are planning assumptions."
    else:
        ax.text(.5, .55, "No intervention plan available", transform=ax.transAxes,
                ha="center", va="center", color=MUTED, fontsize=10)
        ax.text(.5, .43, "Run RL optimization to populate this figure with cost and carbon outputs.",
                transform=ax.transAxes, ha="center", va="center", color=MUTED, fontsize=8)
        ax.set_axis_off()
        caption = "Plan-dependent figure; no recommendations are fabricated when no plan exists."
    ax.set_xlabel("Lifecycle cost (thousands of USD)")
    ax.set_title("Installation and maintenance cost by zone")
    figure_data.append(_save(fig, output_dir / "07_lifecycle_costs.png", "Lifecycle intervention costs", caption))

    representative = max(zones, key=lambda zone: by_zone[zone["zone_id"]]["vulnerability"])
    attributions = service.predictor.explain(zones, representative["zone_id"])
    attributions.sort(key=lambda item: abs(item["contribution"]))
    attribution_values = [item["contribution"] for item in attributions]
    attribution_colors = [TEAL if value >= 0 else WINE for value in attribution_values]
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), constrained_layout=True)
    y_positions = np.arange(len(attributions))
    ax.barh(y_positions, attribution_values, color=attribution_colors, edgecolor="white", linewidth=.4)
    ax.set_yticks(y_positions, [item["feature"].replace("_", " ") for item in attributions])
    ax.axvline(0, color="#657266", linewidth=.7)
    ax.set_xlabel("Signed local sensitivity in normalized input space")
    ax.set_title(f"Feature sensitivity · {representative['name']} ({service.predictor.metrics.get('model_name')})")
    method = ("Integrated gradients" if service.predictor.model_type == "spatiotemporal_gnn"
              else "One-feature replacement sensitivity")
    ax.text(.01, -.17, f"{method}; descriptive model attribution, not causal evidence.",
            transform=ax.transAxes, fontsize=8, color=MUTED)
    figure_data.append(_save(fig, output_dir / "08_feature_sensitivity.png", "Feature sensitivity",
                             f"{method} for the highest predicted-vulnerability zone, {representative['name']}."))

    return {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "dpi": FIGURE_DPI,
            "scenario": scenario, "data_status": service.dataset_info(), "model": service.predictor.metrics.get("model_name"),
            "figures": figure_data}