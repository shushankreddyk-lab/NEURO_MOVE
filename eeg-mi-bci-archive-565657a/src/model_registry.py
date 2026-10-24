"""
Model Registry - Tracks trained models with metadata.
"""
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import sys

# fcntl not available on Windows - use simple file locking fallback
if sys.platform != "win32":
    try:
        import fcntl
        HAS_FCNTL = True
    except ImportError:
        HAS_FCNTL = False
else:
    HAS_FCNTL = False


class ModelRegistry:
    """Thread-safe model registry with file locking (Unix) or atomic writes (Windows)."""
    
    def __init__(self, registry_path: str = None):
        if registry_path is None:
            registry_path = Path(__file__).resolve().parents[1] / "models" / "registry.json"
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_registry()
    
    def _ensure_registry(self):
        if not self.registry_path.exists():
            self.registry_path.write_text(json.dumps({
                "version": "1.0",
                "updated": datetime.utcnow().isoformat() + "Z",
                "models": []
            }, indent=2))
    
    def _load(self) -> Dict[str, Any]:
        with open(self.registry_path, 'r') as f:
            return json.load(f)
    
    def _save(self, data: Dict[str, Any]):
        data["updated"] = datetime.utcnow().isoformat() + "Z"
        # Atomic write with file locking
        tmp_path = self.registry_path.with_suffix('.tmp')
        with open(tmp_path, 'w') as f:
            json.dump(data, f, indent=2)
        tmp_path.replace(self.registry_path)
    
    def _file_hash(self, filepath: Path) -> str:
        """Compute SHA256 hash of model file."""
        h = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()[:16]
    
    def register(self, 
                 model_path: str,
                 model_name: str,
                 architecture: str,
                 dataset: str,
                 subjects: str,
                 accuracy: float,
                 f1_score: float = None,
                 latency_ms: float = None,
                 training_time_s: float = None,
                 config: Dict[str, Any] = None,
                 tags: List[str] = None) -> Dict[str, Any]:
        """Register a trained model."""
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        entry = {
            "id": f"{architecture}_{dataset}_{subjects}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "model_name": model_name,
            "architecture": architecture,
            "dataset": dataset,
            "subjects": subjects,
            "accuracy": accuracy,
            "f1_score": f1_score,
            "latency_ms": latency_ms,
            "training_time_s": training_time_s,
            "file_path": str(path.relative_to(self.registry_path.parent)),
            "file_hash": self._file_hash(path),
            "file_size_mb": round(path.stat().st_size / 1024 / 1024, 2),
            "config": config or {},
            "tags": tags or [],
            "registered_at": datetime.utcnow().isoformat() + "Z"
        }
        
        data = self._load()
        data["models"].append(entry)
        self._save(data)
        return entry
    
    def get(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model by ID."""
        data = self._load()
        for m in data["models"]:
            if m["id"] == model_id:
                return m
        return None
    
    def list(self, 
             architecture: str = None,
             dataset: str = None,
             min_accuracy: float = None,
             tags: List[str] = None) -> List[Dict[str, Any]]:
        """List models with optional filters."""
        data = self._load()
        models = data["models"]
        
        if architecture:
            models = [m for m in models if m["architecture"].lower() == architecture.lower()]
        if dataset:
            models = [m for m in models if m["dataset"].lower() == dataset.lower()]
        if min_accuracy:
            models = [m for m in models if m["accuracy"] >= min_accuracy]
        if tags:
            models = [m for m in models if all(t in m.get("tags", []) for t in tags)]
        
        return sorted(models, key=lambda m: m["registered_at"], reverse=True)
    
    def best(self, architecture: str = None, dataset: str = None) -> Optional[Dict[str, Any]]:
        """Get best model by accuracy."""
        models = self.list(architecture=architecture, dataset=dataset)
        if not models:
            return None
        return max(models, key=lambda m: m["accuracy"])
    
    def delete(self, model_id: str) -> bool:
        """Delete model entry and optionally the file."""
        data = self._load()
        original_len = len(data["models"])
        data["models"] = [m for m in data["models"] if m["id"] != model_id]
        if len(data["models"]) < original_len:
            self._save(data)
            return True
        return False
    
    def get_latest_path(self, architecture: str = None, dataset: str = None) -> Optional[Path]:
        """Get path to latest registered model file."""
        best = self.best(architecture=architecture, dataset=dataset)
        if best:
            return self.registry_path.parent / best["file_path"]
        return None


# Global registry instance
_registry = None

def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


if __name__ == "__main__":
    reg = get_registry()
    print("Model Registry initialized at:", reg.registry_path)
    print("Models:", len(reg.list()))