from __future__ import annotations

import sys
import traceback

sys.path.insert(0, ".")

from neuromove.service import NeuroMoveService

s = NeuroMoveService()
print("health ok =", s.health()["ok"])

print("\n--- preprocess ---")
print(s.preprocess("BCI Competition IV 2a"))

print("\n--- train queue ---")
job = s.submit_training("BCI Competition IV 2a", "minirocket", epochs=2, batch_size=32)
print(job)

print("\n--- run preview ---")
try:
    updated = s.run_small_training_preview(
        job.job_id, "minirocket", "BCI Competition IV 2a", epochs=2, batch_size=32, learning_rate=1e-3
    )
    print(updated)
except Exception as exc:
    traceback.print_exc()

print("\n--- infer ---")
try:
    result = s.infer("minirocket", "BCI Competition IV 2a", sample_count=8, channels=22, samples=656)
    print(result)
except Exception as exc:
    traceback.print_exc()
