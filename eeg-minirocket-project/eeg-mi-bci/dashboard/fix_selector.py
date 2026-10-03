import os
p = r'D:\eeg-minirocket-project\eeg-mi-bci\dashboard\app.py'
with open(p, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the first dataset selector
old_str1 = '["PhysioNet EEGMMIDB", "BCI Competition IV 2a", "High-Gamma Dataset", "Kaya Finger Movements", "WAY-EEG-GAL", "DREAMER Emotion"]'
new_str1 = '["High-Gamma Dataset", "Kaya Finger Movements", "WAY-EEG-GAL", "DREAMER Emotion"]'
content = content.replace(old_str1, new_str1)

# Replace the second dataset selector (for tab 1, or another one)
old_str2 = '["PhysioNet EEGMMIDB", "BCI Competition IV 2a", "High-Gamma Dataset", "Kaya Finger Movements", "WAY-EEG-GAL", "DREAMER Emotion"]'
new_str2 = '["High-Gamma Dataset", "Kaya Finger Movements", "WAY-EEG-GAL", "DREAMER Emotion"]'
content = content.replace(old_str2, new_str2)

with open(p, 'w', encoding='utf-8') as f:
    f.write(content)
