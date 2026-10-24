"""
Unit tests for NeuroMove Archive.
Run with: pytest tests/ -v
"""
import sys
from pathlib import Path
import pytest
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _check_torch_available():
    try:
        import torch
        return True
    except (ImportError, OSError):
        return False


class TestConfigLoader:
    """Test configuration loading."""
    
    def test_config_loads(self):
        from config_loader import get_config
        cfg = get_config()
        assert cfg is not None
    
    def test_minirocket_config(self):
        from config_loader import get_config
        cfg = get_config()
        assert cfg['minirocket.num_kernels'] == 10000
        assert cfg['minirocket.hidden_dim'] == 512
        assert cfg['minirocket.head_epochs'] == 30
    
    def test_preprocessing_config(self):
        from config_loader import get_config
        cfg = get_config()
        assert cfg['preprocessing.sfreq'] == 160
        assert cfg['preprocessing.low_freq'] == 4.0
        assert cfg['preprocessing.high_freq'] == 38.0
        assert cfg['preprocessing.apply_car'] is True


class TestSyntheticDataGenerator:
    """Test synthetic EEG data generation."""
    
    def test_generate_basic(self):
        from scripts.generate_synthetic import generate_synthetic_eeg
        X, y, ch_names = generate_synthetic_eeg(n_trials=20, n_channels=20, n_samples=656)
        assert X.shape == (20, 20, 656)
        assert y.shape == (20,)
        assert len(ch_names) == 20
        assert len(np.unique(y)) == 4
    
    def test_generate_bci2a(self):
        from scripts.generate_synthetic import generate_synthetic_eeg
        X, y, ch_names = generate_synthetic_eeg(n_trials=20, dataset='bci2a')
        assert len(ch_names) == 22
    
    def test_generate_highgamma(self):
        from scripts.generate_synthetic import generate_synthetic_eeg
        X, y, ch_names = generate_synthetic_eeg(n_trials=20, dataset='highgamma')
        assert len(ch_names) == 128
    
    def test_continuous_generation(self):
        from scripts.generate_synthetic import generate_continuous_eeg
        cont, events, ch_names, event_id = generate_continuous_eeg(duration_sec=5.0)
        assert cont.shape[1] == 20
        assert events.shape[0] > 0
        assert len(event_id) == 4


class TestModelRegistry:
    """Test model registry."""
    
    def test_registry_init(self, tmp_path):
        from model_registry import ModelRegistry
        registry_path = tmp_path / "test_registry.json"
        reg = ModelRegistry(str(registry_path))
        assert reg.registry_path.exists()
        assert len(reg.list()) == 0
    
    def test_register_model(self, tmp_path):
        from model_registry import ModelRegistry
        
        registry_path = tmp_path / "test_registry.json"
        reg = ModelRegistry(str(registry_path))
        
        # Create dummy model file without torch
        model_file = tmp_path / "dummy_model.pth"
        model_file.write_bytes(b"dummy model data")
        
        entry = reg.register(
            model_path=str(model_file),
            model_name="test_model",
            architecture="minirocket",
            dataset="physionetmi",
            subjects="1-10",
            accuracy=0.95
        )
        
        assert entry['architecture'] == 'minirocket'
        assert entry['accuracy'] == 0.95
        assert 'file_hash' in entry
    
    def test_list_models(self, tmp_path):
        from model_registry import ModelRegistry
        
        registry_path = tmp_path / "test_registry.json"
        reg = ModelRegistry(str(registry_path))
        
        for i in range(3):
            model_file = tmp_path / f"model_{i}.pth"
            model_file.write_bytes(f"model data {i}".encode())
            reg.register(
                model_path=str(model_file),
                model_name=f"model_{i}",
                architecture="minirocket",
                dataset="physionetmi",
                subjects=f"1-{10+i}",
                accuracy=0.90 + i * 0.01
            )
        
        models = reg.list()
        assert len(models) == 3
        # Should be sorted by registration date (newest first)
        assert models[0]['accuracy'] > models[1]['accuracy']


class TestMinimalPipeline:
    """Test MiniRocket pipeline (if dependencies available)."""
    
    @pytest.mark.skipif(
        not _check_torch_available(),
        reason="PyTorch not available"
    )
    def test_minirocket_init(self):
        from minirocket_engine import MiniRocketPipeline
        pipeline = MiniRocketPipeline(num_kernels=1000, in_channels=20, seq_len=656)
        assert pipeline.num_kernels == 1000
        assert pipeline.in_channels == 20
        assert pipeline.seq_len == 656
        assert pipeline.is_fitted is False
    
    @pytest.mark.skipif(
        not _check_torch_available(),
        reason="PyTorch not available"
    )
    def test_minirocket_not_fitted_raises(self):
        from minirocket_engine import MiniRocketPipeline
        pipeline = MiniRocketPipeline(num_kernels=100, in_channels=20, seq_len=656)
        X = np.random.randn(10, 20, 656).astype(np.float32)
        
        with pytest.raises(RuntimeError, match="not fitted"):
            pipeline.predict(X)
        
        with pytest.raises(RuntimeError, match="not fitted"):
            pipeline.predict_proba(X)


class TestPreprocessing:
    """Test preprocessing functions."""
    
    def test_preprocessing_import(self):
        from preprocessing import apply_car, apply_bandpass_filter
        assert callable(apply_car)
        assert callable(apply_bandpass_filter)
    
    def test_apply_car(self):
        import mne
        from preprocessing import apply_car
        # Create MNE Raw object
        data = np.random.randn(10, 1000).astype(np.float32)
        info = mne.create_info(ch_names=[f"CH{i}" for i in range(10)], sfreq=160, ch_types='eeg')
        raw = mne.io.RawArray(data, info)
        raw_car = apply_car(raw)
        # Mean across channels should be ~0
        assert np.abs(raw_car.get_data().mean(axis=0)).max() < 1e-5
    
    def test_apply_bandpass(self):
        import mne
        from preprocessing import apply_bandpass_filter
        # Create MNE Raw object
        data = np.random.randn(10, 1000).astype(np.float32)
        info = mne.create_info(ch_names=[f"CH{i}" for i in range(10)], sfreq=160, ch_types='eeg')
        raw = mne.io.RawArray(data, info)
        raw_filtered = apply_bandpass_filter(raw, l_freq=4, h_freq=38)
        assert raw_filtered.get_data().shape == raw.get_data().shape


class TestConfigYaml:
    """Test config.yaml parsing."""
    
    def test_config_yaml_exists(self):
        config_path = Path(__file__).resolve().parents[1] / "config.yaml"
        assert config_path.exists()
    
    def test_config_yaml_valid(self):
        import yaml
        config_path = Path(__file__).resolve().parents[1] / "config.yaml"
        with open(config_path) as f:
            data = yaml.safe_load(f)
        assert 'minirocket' in data
        assert 'preprocessing' in data
        assert 'dashboard' in data
        assert 'model_registry' in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])