"""Completed-project validation: ML core + demo backend + frontend wiring (no DB/docker needed)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eeg-mi-bci" / "src"))
sys.path.insert(0, str(ROOT / "neuro-move-release" / "backend"))

import numpy as np


class TestMiniRocketCore(unittest.TestCase):
    def test_fit_predict_proba(self):
        from minirocket_engine import MiniRocketPipeline

        X = np.random.randn(12, 8, 128).astype(np.float32)
        y = np.random.randint(0, 2, size=(12,))
        mr = MiniRocketPipeline(num_kernels=200, in_channels=8, seq_len=128, head_epochs=2)
        mr.fit(X, y)
        preds = mr.predict(X)
        proba = mr.predict_proba(X)
        self.assertEqual(len(preds), 12)
        self.assertEqual(proba.shape, (12, len(np.unique(y))))
        # save/load round-trip
        tmp = ROOT / "artifacts_check_tmp.pth"
        try:
            mr.save(str(tmp))
            mr2 = MiniRocketPipeline(num_kernels=200, in_channels=8, seq_len=128, head_epochs=1)
            mr2.load(str(tmp))
            self.assertEqual(mr2.predict(X).shape, (12,))
        finally:
            if tmp.exists():
                tmp.unlink()


class TestDemoBackend(unittest.TestCase):
    def test_routes_and_flow(self):
        from main import app

        paths = sorted([r.path for r in app.routes])
        for p in ["/health", "/datasets", "/train", "/infer", "/evaluate"]:
            self.assertIn(p, paths)
        # service-level flow (no httpx/TestClient needed)
        from neuromove.service import NeuroMoveService

        s = NeuroMoveService()
        self.assertTrue(s.health()["ok"])
        self.assertGreater(len(s.dataset_list()), 0)
        job = s.submit_training("BCI Competition IV 2a", "minirocket", epochs=1, batch_size=8)
        updated = s.run_small_training_preview(job.job_id, "minirocket", "BCI Competition IV 2a", epochs=1, batch_size=8)
        self.assertEqual(updated.status, "completed")
        inf = s.infer("minirocket", "BCI Competition IV 2a", sample_count=4, channels=8, samples=128)
        self.assertTrue(inf.ok)
        self.assertEqual(inf.num_samples, 4)


class TestFrontendWiring(unittest.TestCase):
    def test_api_service_exports(self):
        svc = ROOT / "neuro-move-release" / "frontend" / "src" / "services" / "api.ts"
        text = svc.read_text(encoding="utf-8", errors="ignore")
        for name in ["getHealth", "getDatasets", "submitTraining", "getJobStatus", "runPreview", "infer", "evaluate", "apiService"]:
            self.assertIn(name, text)
        self.assertIn("VITE_API_URL", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
