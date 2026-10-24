"""NeuroMove Demo Backend — thin launcher over neuromove.api:app (real engine service)."""
from neuromove.api import app  # noqa: F401  (uvicorn entry: main:app)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
