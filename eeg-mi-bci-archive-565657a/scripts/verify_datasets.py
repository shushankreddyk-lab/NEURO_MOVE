"""
Dataset Verification & Download Script.
Checks for required datasets and provides download instructions.
"""
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import hashlib
import json


DATASET_SPECS = {
    'physionet_eegmmidb': {
        'name': 'PhysioNet EEG Motor Movement/Imagery Database (EEGMMIDB)',
        'url': 'https://physionet.org/content/eegmmidb/1.0.0/',
        'subjects': 109,
        'channels': 64,
        'sfreq': 160,
        'format': '.edf',
        'runs_per_subject': 14,
        'motor_runs': [3, 4, 7, 8, 11, 12],  # Motor imagery runs
        'expected_files': 109 * 14,
        'size_gb': 3.5,
        'local_path': 'datasets/physionet',
    },
    'bci_competition_iv_2a': {
        'name': 'BCI Competition IV Dataset 2a',
        'url': 'http://bnci-horizon-2020.eu/database/data-sets',
        'subjects': 9,
        'channels': 22,
        'sfreq': 250,
        'format': '.gdf',
        'runs_per_subject': 2,
        'motor_runs': [0, 1],
        'expected_files': 9 * 2,
        'size_gb': 0.5,
        'local_path': 'datasets/BCICIV_2a_gdf',
    },
    'high_gamma': {
        'name': 'High-Gamma Dataset (Schirrmeister et al.)',
        'url': 'https://doi.org/10.1101/083393',
        'subjects': 14,
        'channels': 128,
        'sfreq': 500,
        'format': '.mat',
        'size_gb': 2.0,
        'local_path': 'datasets/high_gamma',
    },
    'kaya_fingers': {
        'name': 'Kaya Finger Movement Dataset',
        'url': 'https://figshare.com/articles/dataset/EEG_data_for_finger_movement/1194900',
        'subjects': 10,
        'channels': 19,
        'sfreq': 1000,
        'format': '.mat',
        'size_gb': 1.0,
        'local_path': 'datasets/kaya',
    },
    'way_eeg_gal': {
        'name': 'WAY-EEG-GAL Grasp-and-Lift Dataset',
        'url': 'https://figshare.com/articles/dataset/WAY-EEG-GAL/1194900',
        'subjects': 10,
        'channels': 32,
        'sfreq': 500,
        'format': '.csv',
        'size_gb': 1.5,
        'local_path': 'datasets/way_eeg_gal',
    },
    'nemar_finger_mi': {
        'name': 'NEMAR Finger Motor Imagery (OpenNeuro ds004504)',
        'url': 'https://openneuro.org/datasets/ds004504',
        'subjects': 8,
        'channels': 65,
        'sfreq': 1000,
        'format': '.edf',
        'size_gb': 4.0,
        'local_path': 'datasets/nemar_nm000172',
    },
}


def verify_dataset(dataset_key: str, base_path: str = None) -> Tuple[bool, Dict]:
    """
    Verify dataset exists and has expected structure.
    
    Returns:
        (is_valid, info_dict)
    """
    if dataset_key not in DATASET_SPECS:
        return False, {'error': f'Unknown dataset: {dataset_key}'}
    
    spec = DATASET_SPECS[dataset_key]
    base = Path(base_path) if base_path else Path.cwd()
    dataset_path = base / spec['local_path']
    
    info = {
        'dataset': dataset_key,
        'name': spec['name'],
        'path': str(dataset_path),
        'exists': dataset_path.exists(),
        'files_found': 0,
        'total_size_mb': 0,
        'subjects_found': 0,
        'issues': [],
    }
    
    if not dataset_path.exists():
        info['issues'].append(f"Dataset directory not found: {dataset_path}")
        return False, info
    
    # Count files
    ext = spec['format']
    files = list(dataset_path.rglob(f'*{ext}'))
    info['files_found'] = len(files)
    info['total_size_mb'] = sum(f.stat().st_size for f in files) / 1024 / 1024
    
    # Count unique subjects
    subjects = set()
    for f in files:
        # Try to extract subject ID from filename
        name = f.stem.lower()
        for part in name.replace('-', '_').split('_'):
            if part.isdigit() or (part.startswith('s') and part[1:].isdigit()):
                subjects.add(int(part.replace('s', '')))
                break
    info['subjects_found'] = len(subjects)
    
    # Validate
    if info['files_found'] < spec.get('expected_files', 0) * 0.5:
        info['issues'].append(f"Too few files: {info['files_found']} < {spec.get('expected_files', '?')}")
    
    if info['total_size_mb'] < spec.get('size_gb', 0) * 1024 * 0.5:
        info['issues'].append(f"Size seems small: {info['total_size_mb']:.0f} MB < {spec.get('size_gb', '?')} GB")
    
    is_valid = len(info['issues']) == 0
    return is_valid, info


def verify_all_datasets(base_path: str = None) -> Dict:
    """Verify all known datasets."""
    results = {}
    for key in DATASET_SPECS:
        valid, info = verify_dataset(key, base_path)
        results[key] = info
    return results


def print_verification_report(results: Dict):
    """Print formatted verification report."""
    print("\n" + "="*80)
    print("DATASET VERIFICATION REPORT")
    print("="*80)
    
    for key, info in results.items():
        status = "VALID" if info.get('exists', False) and not info.get('issues') else "MISSING/ISSUES"
        print(f"\n{info.get('name', key)}")
        print(f"  Path:     {info.get('path', 'N/A')}")
        print(f"  Status:   {status}")
        print(f"  Files:    {info.get('files_found', 0)}")
        print(f"  Size:     {info.get('total_size_mb', 0):.1f} MB")
        print(f"  Subjects: {info.get('subjects_found', 0)}")
        
        for issue in info.get('issues', []):
            print(f"  WARNING: {issue}")
    
    print("\n" + "="*80)
    valid_count = sum(1 for r in results.values() if r.get('exists') and not r.get('issues'))
    print(f"Valid: {valid_count}/{len(results)} datasets")
    print("="*80)


def generate_download_instructions(dataset_key: str) -> str:
    """Generate download instructions for a dataset."""
    if dataset_key not in DATASET_SPECS:
        return f"Unknown dataset: {dataset_key}"
    
    spec = DATASET_SPECS[dataset_key]
    base = Path.cwd() / spec['local_path']
    
    instructions = f"""
# Download Instructions for {spec['name']}

## Dataset: {dataset_key}
- Subjects: {spec['subjects']}
- Channels: {spec['channels']}
- Sampling: {spec['sfreq']} Hz
- Format: {spec['format']}
- Size: ~{spec['size_gb']} GB
- Source: {spec['url']}

## Expected Location
{base}

## Download Commands

### Option 1: PhysioNet (EEGMMIDB)
```bash
# Requires PhysioNet account + wget
wget -r -N -c -np -P {base} https://physionet.org/files/eegmmidb/1.0.0/
# Or use physionet-python client:
pip install physionet
python -m physionet download eegmmidb {base}
```

### Option 2: BCI Competition IV 2a
```bash
# Manual download from: http://bnci-horizon-2020.eu/database/data-sets
# Place .gdf files in {base}
```

### Option 3: Use MOABB (recommended for multiple datasets)
```bash
pip install moabb
python -c "
from moabb.datasets import EEGMMI, BNCI2014001
EEGMMM().download({base}/physionet)
BNCI2014001().download({base}/BCICIV_2a_gdf)
"
```

## Verification
After download, run:
```bash
python scripts/verify_datasets.py {dataset_key}
```
"""
    return instructions


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Verify NeuroMove datasets')
    parser.add_argument('dataset', nargs='?', help='Dataset key to verify (or all)')
    parser.add_argument('--base', default='.', help='Base path for datasets')
    parser.add_argument('--download-info', action='store_true', help='Print download instructions')
    
    args = parser.parse_args()
    
    if args.download_info and args.dataset:
        print(generate_download_instructions(args.dataset))
    elif args.dataset:
        valid, info = verify_dataset(args.dataset, args.base)
        print_verification_report({args.dataset: info})
    else:
        results = verify_all_datasets(args.base)
        print_verification_report(results)


if __name__ == "__main__":
    main()