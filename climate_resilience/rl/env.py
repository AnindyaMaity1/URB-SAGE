import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    BaseEnv = gym.Env
except ImportError:
    gym = None
    spaces = None
    BaseEnv = object


INTERVENTIONS = {
    "No intervention": {"cost": 0, "maintenance_cost": 0, "flood": 0, "heat": 0, "carbon": 0, "green": 0},
    "Urban trees": {"cost": 45000, "maintenance_cost": 12000, "flood": .07, "heat": .15, "carbon": 18, "green": 9},
    "Green roof": {"cost": 120000, "maintenance_cost": 28000, "flood": .08, "heat": .13, "carbon": 25, "green": 12},
    "Rain garden": {"cost": 65000, "maintenance_cost": 15000, "flood": .19, "heat": .05, "carbon": 7, "green": 10},
    "Permeable pavement": {"cost": 80000, "maintenance_cost": 10000, "flood": .17, "heat": .06, "carbon": 9, "green": 5},
    "Urban wetland": {"cost": 160000, "maintenance_cost": 35000, "flood": .24, "heat": .08, "carbon": 30, "green": 18},
}
ACTION_NAMES = list(INTERVENTIONS)


class GreenInfrastructureEnv(BaseEnv):
    """One decision per zone; Gymnasium API with a local fallback."""

    metadata = {"render_modes": []}

    def __init__(self, zones, predictions, budget=500000, weights=None, scenario="Baseline", green_target=35):
        super().__init__()
        self.zones = zones
        self.predictions = {item["zone_id"]: item for item in predictions}
        self.budget = max(0.0, float(budget))
        self.weights = weights or {"flood": .4, "heat": .35, "carbon": .15, "green": .1}
        self.scenario = scenario
        self.green_target = max(1.0, float(green_target))
        if spaces:
            self.action_space = spaces.Discrete(len(ACTION_NAMES))
            self.observation_space = spaces.Box(-5, 5, shape=(11,), dtype=np.float32)
        self.reset()

    def _observation(self):
        if self.index >= len(self.zones):
            return np.zeros(11, dtype=np.float32)
        zone = self.zones[self.index]
        p = self.predictions[zone["zone_id"]]
        f = zone["features"]
        return np.asarray([
            p["flood_risk"], p["heat_risk"], p["vulnerability"], p["uncertainty"],
            f["vegetation_pct"] / 100, f["building_density"], f["drainage_capacity"],
            f["impervious_pct"] / 100, self.remaining / max(self.budget, 1),
            self.index / max(len(self.zones) - 1, 1),
            1.0 if self.scenario == "Compound stress" else 0.0,
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        if gym:
            super().reset(seed=seed)
        self.index = 0
        self.remaining = self.budget
        self.plan = []
        observation = self._observation()
        return (observation, {}) if gym else observation

    def step(self, action):
        action = int(action)
        action = action if 0 <= action < len(ACTION_NAMES) else 0
        zone = self.zones[self.index]
        prediction = self.predictions[zone["zone_id"]]
        features = zone["features"]
        intervention = INTERVENTIONS[ACTION_NAMES[action]]
        lifecycle_cost = float(intervention["cost"] + intervention["maintenance_cost"])
        if lifecycle_cost > self.remaining:
            reward = -1.0
            chosen = 0
            intervention = INTERVENTIONS[ACTION_NAMES[0]]
            lifecycle_cost = 0.0
        else:
            chosen = action
            self.remaining -= lifecycle_cost
            reward = 5 * (
                self.weights["flood"] * prediction["flood_risk"] * intervention["flood"]
                + self.weights["heat"] * prediction["heat_risk"] * intervention["heat"]
            ) + self.weights["carbon"] * intervention["carbon"] / 30
            remaining_gap = max(0.0, self.green_target - features["vegetation_pct"])
            reward += self.weights["green"] * min(intervention["green"], remaining_gap) / self.green_target
            reward -= .42 * lifecycle_cost / max(self.budget, 1)
            reward -= .025
        if chosen:
            self.plan.append({"zone_id": zone["zone_id"], "action": ACTION_NAMES[chosen], **intervention})
        self.index += 1
        terminated = self.index >= len(self.zones)
        result = self._observation()
        info = {"zone_id": zone["zone_id"], "remaining_budget": self.remaining, "plan": self.plan}
        return (result, float(reward), terminated, False, info) if gym else (result, float(reward), terminated, info)