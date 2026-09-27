
import numpy as np
import sys
import os

# Ensure src is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from binary_parser import load_local_eeg_data, normalize_channel_names

def test_normalize_channel_names():
    raw_names = ['FC3.', 'C3..', 'Cz  ', 'O1']
    normalized = normalize_channel_names(raw_names)
    assert normalized['FC3.'] == 'FC3'
    assert normalized['C3..'] == 'C3'
    assert normalized['Cz  '] == 'CZ'
    assert normalized['O1'] == 'O1'

def test_label_mapping():
    # Only try to load subject 1, all 14 runs
    runs_to_test = list(range(1, 15))
    raws, events_list, mappings = load_local_eeg_data(1, runs_to_test)
    
    # We should have successfully loaded some runs
    assert len(raws) > 0
    
    for i, r in enumerate(runs_to_test):
        # We index by the actual loaded position, but we assume S001 has all 14 runs
        # If it doesn't, this test will need to be more robust, but S001 has them.
        raw = raws[i]
        events = events_list[i]
        mapping = mappings[i]
        
        # Verify sampling rate
        assert raw.info['sfreq'] == 160.0
        
        # Verify specific mappings based on run
        if r in [1, 2]:
            assert 0 in mapping
            assert 'eyes' in mapping[0]
            assert len(events) == 0 # we return empty events for baseline
        elif r in [3, 7, 11]:
            labels = list(mapping.values())
            assert 'rest' in labels
            assert 'left_fist_real' in labels
            assert 'right_fist_real' in labels
        elif r in [4, 8, 12]:
            labels = list(mapping.values())
            assert 'rest' in labels
            assert 'left_fist_imagined' in labels
            assert 'right_fist_imagined' in labels
        elif r in [5, 9, 13]:
            labels = list(mapping.values())
            assert 'rest' in labels
            assert 'both_fists_real' in labels
            assert 'both_feet_real' in labels
        elif r in [6, 10, 14]:
            labels = list(mapping.values())
            assert 'rest' in labels
            assert 'both_fists_imagined' in labels
            assert 'both_feet_imagined' in labels

