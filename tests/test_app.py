import io
import unittest
from pathlib import Path

from PIL import Image

from app import app
from climate_resilience.database import get_zones
from climate_resilience.predictor import MODEL_NAMES, RiskPredictor


class ClimateResilienceApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_health_and_seeded_overview(self):
        health = self.client.get("/api/health")
        overview = self.client.get("/api/overview")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json["status"], "ready")
        self.assertGreaterEqual(len(overview.json["zones"]), 3)
        self.assertEqual(overview.json["dataset"]["name"], "Chennai south study area (demonstration)")
        self.assertIn("Adyar", {zone["name"] for zone in overview.json["zones"]})
        self.assertTrue(all(12 <= zone["latitude"] <= 14 and 79 <= zone["longitude"] <= 82
                    for zone in overview.json["zones"]))
        self.assertFalse(any(zone["zone_id"].startswith("Z0") for zone in overview.json["zones"]))
        for prediction in overview.json["predictions"]:
            self.assertTrue(0 <= prediction["uncertainty"] <= 1)
            self.assertTrue(0 <= prediction["confidence"] <= 1)

    def test_graph_and_scenario_endpoints(self):
        graph = self.client.get("/api/graph")
        scenarios = self.client.get("/api/scenarios")
        self.assertEqual(graph.status_code, 200)
        self.assertTrue(graph.json["edges"])
        self.assertEqual(len(scenarios.json), 5)

    def test_zone_explanation_returns_feature_attributions(self):
        response = self.client.get("/api/explain/Adyar")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json["contributions"]), 9)
        self.assertIn("Integrated gradients", response.json["method"])

    def test_live_refresh_is_timestamped_and_does_not_need_sensor_data(self):
        response = self.client.get("/api/live?scenario=Heatwave")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["scenario"], "Heatwave")
        self.assertIn("computed_at", response.json)
        self.assertEqual(len(response.json["predictions"]), 12)

    def test_supported_models_fit_and_predict(self):
        zones = get_zones()
        for model_type in MODEL_NAMES:
            with self.subTest(model_type=model_type):
                predictor = RiskPredictor(epochs=10, model_type=model_type)
                predictor.fit(zones)
                predictions = predictor.predict(zones, draws=4)
                self.assertEqual(len(predictions), len(zones))
                self.assertEqual(predictor.metrics["model_type"], model_type)

    def test_publication_figures_are_1200dpi_and_downloadable(self):
        response = self.client.get("/api/figures?scenario=Heatwave")
        self.assertEqual(response.status_code, 200, response.json)
        self.assertEqual(response.json["dpi"], 1200)
        self.assertEqual(response.json["scenario"], "Heatwave")
        self.assertEqual(len(response.json["figures"]), 8)
        self.assertEqual(response.json["figures"][-1]["filename"], "08_feature_sensitivity.png")
        self.assertTrue(all("synthetic" not in figure["caption"].lower() for figure in response.json["figures"]))
        figure = response.json["figures"][0]
        with Image.open(Path("results/figures") / figure["filename"]) as image:
            self.assertAlmostEqual(image.info["dpi"][0], 1200, delta=1)
            self.assertGreaterEqual(image.width, 8000)
        download = self.client.get(figure["url"] + "?download=1")
        self.assertEqual(download.status_code, 200)
        self.assertEqual(download.mimetype, "image/png")
        download.close()
        pdf_download = self.client.get(figure["pdf_url"] + "?download=1")
        self.assertEqual(pdf_download.status_code, 200)
        self.assertEqual(pdf_download.mimetype, "application/pdf")
        self.assertTrue(pdf_download.data.startswith(b"%PDF-"))
        pdf_download.close()
        final_download = self.client.get(response.json["figures"][-1]["url"] + "?download=1")
        self.assertEqual(final_download.status_code, 200)
        final_download.close()

    def test_upload_rejects_missing_schema(self):
        response = self.client.post("/api/upload", data={
            "file": (io.BytesIO(b"zone_id\nZ1\n"), "invalid.csv")
        }, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)
        self.assertIn("Missing required columns", response.json["error"])

    def test_optimization_is_budget_bounded_and_reportable(self):
        response = self.client.post("/api/optimize", json={"budget": 300000, "episodes": 5})
        self.assertEqual(response.status_code, 200, response.json)
        result = response.json
        self.assertLessEqual(result["total_cost"], 300000)
        self.assertEqual(result["total_cost"], result["installation_cost"] + result["maintenance_cost"])
        self.assertIn("Risk-weighted greedy", result["comparisons"])
        report = self.client.get(f"/api/report/{result['run_id']}")
        self.assertEqual(report.status_code, 200)
        self.assertIn("predictions", report.json)
        self.assertTrue(report.json["interventions"])
        self.assertIn("maintenance_cost", report.json["interventions"][0])
        self.assertIn("lifecycle_cost", report.json["interventions"][0])

    def test_preprocessing_and_saved_recommendations(self):
        processed = self.client.post("/api/preprocess")
        recommendations = self.client.get("/api/recommendations")
        self.assertEqual(processed.status_code, 200)
        self.assertEqual(len(processed.json["zones"]), 12)
        self.assertTrue(processed.json["features"])
        self.assertEqual(recommendations.status_code, 200)
        self.assertTrue(recommendations.json["recommendations"])


if __name__ == "__main__":
    unittest.main()