import sys

with open(r'd:\eeg-minirocket-project\eeg-mi-bci\src\train_master.py', 'r', encoding='utf-8') as f:
    content = f.read()

import re

# Find the loop block
start_marker = "                for ev in events:"
end_marker = "                        print(f\"Warning: Got X_batch of size {X_batch.shape}\")"

start_idx = content.find(start_marker)
end_idx = content.find(end_marker) + len(end_marker)

if start_idx == -1 or end_idx < start_idx:
    print("Could not find block to replace")
    sys.exit(1)

new_block = """                valid_events = []
                valid_event_ids = {} 
                event_to_label = {} 

                for ev in events:
                    if len(ev) == 0: continue
                    onset, duration, marker_int = ev
                    desc_str = mapping.get(marker_int)
                    if not desc_str: continue
                    marker_str = inv_orig.get(desc_str)
                    if not marker_str: continue

                    target_group = map_run_and_marker_to_group(run, task_type, marker_str)
                    if target_group == -1:
                        continue
                        
                    if mode == "ovr":
                        lbl = 1 if str(target_group) == str(group_id) else 0
                    else:
                        lbl = target_group
                        
                    valid_events.append(ev)
                    valid_event_ids[str(marker_int)] = marker_int
                    event_to_label[marker_int] = lbl
                    
                if not valid_events:
                    continue
                    
                import mne
                tmax_adj = 4.1 - (1 / raw.info['sfreq'])
                epochs = mne.Epochs(raw, np.array(valid_events), event_id=valid_event_ids, tmin=0, tmax=tmax_adj, baseline=None, preload=True, verbose=False)
                X_batch = epochs.get_data(copy=False)
                
                target_samples = 656
                if X_batch.shape[2] > target_samples:
                    X_batch = X_batch[:, :, :target_samples]
                elif X_batch.shape[2] < target_samples:
                    pad_width = target_samples - X_batch.shape[2]
                    X_batch = np.pad(X_batch, ((0,0), (0,0), (0,pad_width)), mode='constant')
                
                if X_batch.shape[0] > 0:
                    y_batch = [event_to_label[ev[2]] for ev in epochs.events]
                    X_list.append(X_batch)
                    y_list.extend(y_batch)"""

new_content = content[:start_idx] + new_block + content[end_idx:]

with open(r'd:\eeg-minirocket-project\eeg-mi-bci\src\train_master.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("Patched train_master.py successfully.")
