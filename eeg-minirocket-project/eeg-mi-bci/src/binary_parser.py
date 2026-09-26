import mne
import numpy as np
import os
import glob

def get_run_group(run_id):
    """
    Returns the condition group based on run_id.
    """
    if run_id in [1, 2]:
        return 1
    elif run_id in [3, 7, 11]:
        return 2
    elif run_id in [4, 8, 12]:
        return 3
    elif run_id in [5, 9, 13]:
        return 4
    elif run_id in [6, 10, 14]:
        return 5
    return None

def get_label_mapping(run_id):
    """
    Returns a dictionary mapping T0, T1, T2 to specific labels based on the run_id.
    """
    if run_id == 1:
        return {'T0': 'eyes_open'}
    elif run_id == 2:
        return {'T0': 'eyes_closed'}
    elif run_id in [3, 7, 11]:
        return {'T0': 'rest', 'T1': 'left_fist_real', 'T2': 'right_fist_real'}
    elif run_id in [4, 8, 12]:
        return {'T0': 'rest', 'T1': 'left_fist_imagined', 'T2': 'right_fist_imagined'}
    elif run_id in [5, 9, 13]:
        return {'T0': 'rest', 'T1': 'both_fists_real', 'T2': 'both_feet_real'}
    elif run_id in [6, 10, 14]:
        return {'T0': 'rest', 'T1': 'both_fists_imagined', 'T2': 'both_feet_imagined'}
    return {}

def normalize_channel_names(ch_names):
    """
    Strips whitespace and dots from channel names, and converts to uppercase.
    """
    return {ch: ch.upper().strip().replace('.', '') for ch in ch_names}

def load_local_eeg_data(subject_id, runs, data_dir=r'd:\eeg-minirocket-project\physionet'):
    """
    Loads local EDF files for a given subject and run list.
    
    Returns:
    - raws: list of mne.io.Raw objects
    - events_list: list of event arrays
    - event_id_mappings: list of dicts mapping integer event IDs to string labels
    """
    subject_str = f"S{subject_id:03d}"
    subject_dir = os.path.join(data_dir, subject_str)
    
    raws = []
    events_list = []
    event_id_mappings = []
    
    for r in runs:
        run_str = f"R{r:02d}"
        edf_file = os.path.join(subject_dir, f"{subject_str}{run_str}.edf")
        if not os.path.exists(edf_file):
            print(f"File not found: {edf_file}")
            continue
            
        try:
            # Read EDF file
            raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
            
            # Fix sampling rate to 160Hz
            if raw.info['sfreq'] != 160.0:
                raw.resample(160.0, verbose=False)
                
            # Normalize channel names
            raw.rename_channels(normalize_channel_names(raw.ch_names))
            
            # Select relevant motor channels (and occipital for alpha)
            target_channels = ['FC3', 'FC4', 'C3', 'C4', 'CP3', 'CP4', 'C1', 'C2', 'C5', 'C6', 'CZ', 'FCZ', 'CPZ', 'F3', 'F4', 'P3', 'P4', 'O1', 'O2', 'OZ']
            available_channels = raw.ch_names
            picked_channels = [ch for ch in target_channels if ch in available_channels]
            
            if len(picked_channels) > 0:
                raw.pick_channels(picked_channels)
            
            # Convert to microvolts
            raw.apply_function(lambda x: x * 1e6, verbose=False)
            
            # Event parsing
            string_mapping = get_label_mapping(r)
            
            if r in [1, 2]:
                # Baseline runs have no event markers. Create a dummy event list for slicing later.
                events = np.empty((0, 3), dtype=int)
                int_to_str_mapping = {0: string_mapping['T0']}
            else:
                # Task runs have T0, T1, T2
                try:
                    events, event_dict = mne.events_from_annotations(raw, verbose=False)
                    # event_dict is typically {'T0': 1, 'T1': 2, 'T2': 3}
                    int_to_str_mapping = {}
                    for annotation_str, event_int in event_dict.items():
                        if annotation_str in string_mapping:
                            int_to_str_mapping[event_int] = string_mapping[annotation_str]
                except Exception as e:
                    print(f"Error parsing annotations for {edf_file}: {e}")
                    events = np.empty((0, 3), dtype=int)
                    int_to_str_mapping = {}
            
            raws.append(raw)
            events_list.append(events)
            event_id_mappings.append(int_to_str_mapping)
            
        except Exception as e:
            print(f"Error processing {edf_file}: {e}")
            
    return raws, events_list, event_id_mappings

def load_eegbci_data(subject_id, runs):
    """
    Wrapper for backward compatibility with dashboard/app.py.
    Concatenates multiple runs into a single raw object.
    """
    raws, evs, maps = load_local_eeg_data(subject_id, runs)
    if not raws:
        return None, None
    if len(raws) == 1:
        return raws[0], evs[0]
    
    raw = mne.concatenate_raws(raws, verbose=False)
    events, _ = mne.events_from_annotations(raw, verbose=False)
    return raw, events

    raw, events = load_eegbci_data(1, [4])
    print(raw.info['nchan'], events.shape)

def generate_dataset_toc(data_dir=r'd:\eeg-minirocket-project\physionet'):
    import json
    
    # We define the 5 groups based on run IDs
    group_runs = {
        1: [1, 2],
        2: [3, 7, 11],
        3: [4, 8, 12],
        4: [5, 9, 13],
        5: [6, 10, 14]
    }
    
    # Initialize counts
    toc_data = {
        "1": {"Category": "Ocular Baseline", "Runs": "R01, R02", "Intent": "Continuous Windows", "Desc": "Eyes Open vs. Eyes Closed", "Trials": 0},
        "2": {"Category": "LF / RF (Real Fists)", "Runs": "R03, R07, R11", "Intent": "T1, T2 (Rest T0 dropped)", "Desc": "Left Fist Real (LF) vs. Right Fist Real (RF)", "Trials": 0},
        "3": {"Category": "LF / RF (Imagined Fists)", "Runs": "R04, R08, R12", "Intent": "T1, T2 (Rest T0 dropped)", "Desc": "Left Fist Imagined (LF) vs. Right Fist Imagined (RF)", "Trials": 0},
        "4": {"Category": "BFs / BF (Bilateral Real)", "Runs": "R05, R09, R13", "Intent": "T1, T2 (Rest T0 dropped)", "Desc": "Both Fists Real (BFs) vs. Both Feet Real (BF)", "Trials": 0},
        "5": {"Category": "BFs / BF (Bilateral Imagined)", "Runs": "R06, R10, R14", "Intent": "T1, T2 (Rest T0 dropped)", "Desc": "Both Fists Imagined (BFs) vs. Both Feet Imagined (BF)", "Trials": 0}
    }
    
    detailed_toc = []
    
    for subject_id in range(1, 110):
        subject_str = f"S{subject_id:03d}"
        subject_dir = os.path.join(data_dir, subject_str)
        
        if not os.path.exists(subject_dir):
            continue
            
        # Dynamically scan all files instead of using hardcoded run loops
        for edf_file in glob.glob(os.path.join(subject_dir, "*.edf")):
            filename = os.path.basename(edf_file)
            try:
                raw = mne.io.read_raw_edf(edf_file, preload=False, verbose=False)
                events, event_dict = mne.events_from_annotations(raw, verbose=False)
                
                event_counts = {}
                for evt_name, evt_id in event_dict.items():
                    event_counts[evt_name] = int(np.sum(events[:, 2] == evt_id))
                    
                event_summary = ", ".join([f"{k}:{v}" for k, v in event_counts.items()])
                if not event_summary:
                    event_summary = "No Events"
                    
                # Trials are tasks (T1, T2, T3)
                trials = sum([count for name, count in event_counts.items() if name in ['T1', 'T2', 'T3']])
                
                # Dynamic Classification based on actual contents!
                if trials == 0:
                    # Baseline data found
                    g_id = 1
                    trials = event_counts.get('T0', 15)
                else:
                    # Task data found. Use filename as a hint for the specific task group.
                    run_num_str = filename.replace('.edf', '').replace(subject_str + 'R', '')
                    if run_num_str.isdigit():
                        r = int(run_num_str)
                        if r in [3, 7, 11]: g_id = 2
                        elif r in [4, 8, 12]: g_id = 3
                        elif r in [5, 9, 13]: g_id = 4
                        elif r in [6, 10, 14]: g_id = 5
                        else: g_id = 2 # default fallback
                    else:
                        g_id = 2
                        
                toc_data[str(g_id)]["Trials"] += trials
                
                if g_id == 1:
                    task_type = "Baseline"
                    t0 = "Eyes Open/Closed"
                    t1 = ""
                    t2 = ""
                elif g_id == 2:
                    task_type = "Motor Execution - Unilateral Fist"
                    t0 = "Rest"
                    t1 = "Left Fist"
                    t2 = "Right Fist"
                elif g_id == 3:
                    task_type = "Motor Imagery - Unilateral Fist"
                    t0 = "Rest"
                    t1 = "Left Fist MI"
                    t2 = "Right Fist MI"
                elif g_id == 4:
                    task_type = "Motor Execution - Bilateral Hand/Foot"
                    t0 = "Rest"
                    t1 = "Both Fists"
                    t2 = "Both Feet"
                elif g_id == 5:
                    task_type = "Motor Imagery - Bilateral Hand/Foot"
                    t0 = "Rest"
                    t1 = "Both Fists MI"
                    t2 = "Both Feet MI"

                # Parse run number
                run_num = filename.replace('.edf', '').replace(subject_str + 'R', '')
                run_num = int(run_num) if run_num.isdigit() else ""

                if subject_id in [88, 89, 92, 100, 104, 106]:
                    status = "FLAG - KNOWN INCOMPLETE SUBJECT"
                    quality_flag = "Known inconsistent/incomplete recording"
                else:
                    status = "VALID - PROTOCOL CLASSIFIED"
                    quality_flag = ""
                    
                subgroup_name = f"Group {run_num}" if isinstance(run_num, int) else "Group Unknown"

                detailed_toc.append({
                    "Subject": subject_str,
                    "EDF_File": filename,
                    "Run": run_num,
                    "Task_Type": task_type,
                    "T0": t0,
                    "T1": t1,
                    "T2": t2,
                    "Status": status,
                    "Quality_Flag": quality_flag,
                    "Group": subgroup_name,
                    "Trials": trials,
                    "Trials_T0": event_counts.get('T0', 0),
                    "Trials_T1": event_counts.get('T1', 0),
                    "Trials_T2": event_counts.get('T2', 0)
                })
            except Exception as e:
                print(f"Error parsing {edf_file} for TOC: {e}")
                        
    # Save caches
    artifacts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'artifacts'))
    os.makedirs(artifacts_dir, exist_ok=True)
    
    cache_path = os.path.join(artifacts_dir, 'toc_cache.json')
    with open(cache_path, 'w') as f:
        json.dump(toc_data, f, indent=4)
        
    import pandas as pd
    detailed_df = pd.DataFrame(detailed_toc)
    detailed_csv_path = os.path.join(artifacts_dir, 'toc_detailed.csv')
    detailed_df.to_csv(detailed_csv_path, index=False)
        
    print(f"Table of Contents caches generated in {artifacts_dir}")
    return toc_data

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--generate-toc':
        generate_dataset_toc()
    else:
        raws, evs, maps = load_local_eeg_data(1, [1, 3, 4])
        for r, e, m in zip(raws, evs, maps):
            print(r.info['nchan'], e.shape, m)
