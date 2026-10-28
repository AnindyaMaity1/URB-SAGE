# URB-SAGE Climate Resilience Planning

It is Built under the guidance of **Prof. Maurizio Giacobbe ORCID: [0000−0001−6178−7132](https://orcid.org/0000-0001-6178-7132)**

Department of Mathematics and Computer Sciences, Physical Sciences and Earth Sciences, University of Messina, Italy.

**URB-SAGE: An Uncertainty-Aware Spatio-Temporal Graph Reinforcement Learning Framework for Climate-Resilient Green Infrastructure Planning in Smart Cities**

URB-SAGE is a local, research-oriented decision-support prototype designed for climate adaptation planning in urban environments. It combines geospatial analysis, uncertainty-aware learning, and reinforcement learning to recommend green infrastructure strategies that reduce climate risk under a constrained budget.

## 1. Project Overview

This project addresses a practical urban planning problem: how to decide where and how to invest in green infrastructure when a city must manage flood risk, heat stress, and environmental degradation under limited resources.

The system loads a set of urban zones, each with spatial coordinates and environmental attributes such as rainfall, temperature, humidity, vegetation cover, building density, drainage capacity, imperviousness, and flood history. It then:

- Builds a spatial graph between nearby urban zones.
- Trains a risk prediction model using synthetic or proxy climate labels.
- Estimates uncertainty and reliability alongside risk values.
- Evaluates multiple climate stress scenarios.
- Optimizes intervention choices with a PPO-based policy.
- Generates figures and reports for decision support.

Although the project is designed around Chennai-like urban zoning data, the framework is modular and can be adapted to similar city datasets.

## 2. Research Motivation

The motivating idea is to support transparent, explainable decision-making in urban climate adaptation. The project combines multiple components:

- Spatial representation of neighborhoods or localities.
- Climate-sensitive risk prediction.
- Consideration of uncertainty in model outputs.
- Optimization of infrastructure investments.
- Evaluation of alternative climate scenarios.
- Communication of results through an interactive dashboard.

The result is a research prototype for exploring how machine learning and reinforcement learning can support climate-resilient planning in smart cities.

## 3. Core Features

### 3.1 Uncertainty-aware risk prediction

The application estimates:

- Flood risk.
- Heat risk.
- Combined vulnerability.
- Uncertainty score.
- Confidence.
- Reliability.

For the neural model, uncertainty is approximated using repeated stochastic predictions and model dropout. This supports a more informative view than a single deterministic score.

### 3.2 Spatio-temporal graph learning

The project implements a spatial-temporal graph model that uses local adjacency and a temporal sequence of climate-related features. The model combines:

- Graph-based neighborhood aggregation.
- A temporal GRU layer.
- Output heads for flood, heat, and vulnerability prediction.

This architecture is designed to capture both local spatial patterns and temporal variation in climate stress.

### 3.3 Multiple model baselines

The system supports model comparison using three options:

- Spatio-temporal GNN
- Random Forest
- Extra Trees

This allows experimentation with both structural learning and classical tree-based regression models.

### 3.4 Reinforcement learning for intervention planning

The project includes a Gymnasium-style environment and a PPO planner for budget-constrained green infrastructure allocation. Available intervention actions include:

- Urban trees
- Green roof
- Rain garden
- Permeable pavement
- Urban wetland
- No intervention

The optimization balances:

- Flood reduction
- Heat reduction,
- Carbon impact,
- Green coverage gains,
- Maintenance and installation cost.

### 3.5 Visualization and reporting

The dashboard provides panels for:

- Overview and summary metrics
- Urban risk maps
- Flood and heat risk views
- Uncertainty analysis
- Green infrastructure recommendations
- RL optimization outcomes
- Scenario comparisons
- Explainable AI output
- Before/after comparison
- Analytics and historical runs
- Downloadable results and figure exports


## 4. System Architecture

The project follows a modular local architecture:

- Flask app: web interface and API layer
- Planning service: orchestration logic
- Predictor module: risk estimation and uncertainty logic
- Graph module: spatial adjacency construction
- RL environment and PPO planner: intervention optimization
- SQLite database: persistence for datasets, predictions, recommendations, and experiments
- Static assets and templates: frontend UI and dashboard views

The application flow is:

1. Load a dataset or upload CSV/XLSX/GeoJSON data.
2. Validate the required columns and numeric fields.
3. Build a spatial graph from locality coordinates.
4. Train or reload a model.
5. Generate flood and heat predictions under selected climate scenarios.
6. Optimize green infrastructure interventions under budget constraints.
7. Save results and display them to the user.

## 5. Technologies Used

The project uses the following core technologies:

- Python 3.10+
- Flask
- PyTorch
- NumPy
- pandas
- scikit-learn
- Matplotlib
- Gymnasium
- SQLite
- python-dotenv

## 6. Data Model and Inputs

The app expects urban zone records with fields such as:

- zone_id
- name
- latitude
- longitude
- rainfall_mm
- temperature_c
- humidity_pct
- elevation_m
- vegetation_pct
- building_density
- drainage_capacity
- impervious_pct
- flood_history
- land_use

The dataset represents a demonstration area in Chennai, with the values designed for research use and model testing rather than operational municipal reporting.

## 7. Climate Scenarios

The system supports five scenarios:

- Baseline
- Heavy rainfall
- Extreme rainfall
- Heatwave
- Compound stress

These scenarios change the environmental stress pattern used during prediction and optimization so the planner can compare possible adaptation strategies under different climate futures.

## 8. Scientific and Practical Limitations

Important caveats include:

- Model metrics are in-sample diagnostics and not field-calibrated estimates.
- Intervention costs and effect sizes are planning assumptions.
- Spatial coordinates are used as locality references rather than survey-grade boundaries.
- Outputs are intended for exploratory analysis and research communication.

Before using the system for real-world planning, data provenance, calibration, and local validation should be reviewed carefully.

## 9. Example Use Cases

This project is useful for:

- Research prototypes in urban resilience.
- Climate adaptation scenario exploration.
- Green infrastructure prioritization under budget constraints.
- Teaching graph learning and reinforcement learning concepts.
- Discussion of smart-city planning strategies.

## 10. Summary

URB-SAGE demonstrates a complete local workflow for climate-resilient urban planning, combining:

- Geospatial risk modeling
- Climate scenario analysis
- Uncertainty-aware prediction
- Graph-based learning
- Reinforcement learning for intervention selection
- Dashboard-based result exploration

URB-SAGE is best understood as a transparent research prototype that can be extended to real-world and policy analysis workflows.#
