import re
import os

app_path = r"D:\eeg-minirocket-project\eeg-mi-bci\dashboard\app.py"
with open(app_path, "r", encoding="utf-8") as f:
    text = f.read()

# We need to find places where we have two consecutive `if "High-Gamma" in ...:` or `elif "High-Gamma" in ...:` blocks 
# because I replaced "BCI Competition" and "Physionet" with "High-Gamma".
# Actually, the user wants me to fix the app. Let me just restore app.py from a backup if one exists.
