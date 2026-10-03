import re
import os

app_path = r'D:\eeg-minirocket-project\eeg-mi-bci\dashboard\app.py'
with open(app_path, 'r', encoding='utf-8') as f:
    text = f.read()

start_str = "if selected_tab == '📊 Training Process':"
start_idx = text.find(start_str)
if start_idx != -1:
    end_idx = text.find("if selected_tab == '⚙️ Preprocessing':", start_idx)
    if end_idx == -1: 
        end_idx = text.find("if selected_tab == '🚀 Live Training':", start_idx)
    if end_idx != -1:
        snippet = text[start_idx:end_idx]
        with open(r'D:\eeg-minirocket-project\eeg-mi-bci-archive-565657a\training_process_snippet.py', 'w', encoding='utf-8') as out:
            out.write(snippet)
        print('Snippet found and saved, length:', len(snippet))
    else:
        print('End bound not found')
else:
    print('Start bound not found')
