import re

file_path = r'd:\eeg-minirocket-project\eeg-mi-bci\dashboard\app.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the orphaned code
pattern = r"except Exception as e:\s+st\.error\(f'Error during prediction: \{e\}'\)\s+with col_perf2:\s+st\.metric\(\"Inference Latency Comparison\", \"0\.6 ms \(1x\)\", \"-7\.4 ms vs CNN-LSTM\", delta_color=\"inverse\"\)\s+st\.caption\(\"CNN-LSTM: 8\.0 ms \(13\.3x slower\)\"\)"
replacement = r"except Exception as e:\n                            st.error(f'Error during prediction: {e}')"

new_content = re.sub(pattern, replacement, content)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(new_content)
print("Cleaned up orphaned code.")
