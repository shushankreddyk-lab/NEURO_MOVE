from __future__ import annotations

import sys
import traceback

sys.path.insert(0, ".")

from neuromove.service import NeuroMoveService
from neuromove.models import TrainJobUpdate

s = NeuroMoveService()

job = s.submit_training("BCI Competition IV 2a", "minirocket", epochs=2, batch_size=32)
print("queued:", job.job_id)

try:
    updated = s.run_small_training_preview(
        job.job_id, "minirocket", "BCI Competition IV 2a", epochs=2, batch_size=32, learning_rate=1e-3
    )
    print("preview done:", updated.status)
except Exception as exc:
    traceback.print_exc()
    raise SystemExit(2) from exc

try:
    status = s.job_status(job.job_id)
    print("status returned type:", type(status).__name__)
    print("status:", status)
    print("model_dump:", status.model_dump())
    print("as TrainJobUpdate:", TrainJobUpdate.model_validate(status.model_dump()))
except Exception as exc:
    traceback.print_exc()
