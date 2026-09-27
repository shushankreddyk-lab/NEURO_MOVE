import re

with open('d:/temp_yesterday.py', 'r', encoding='utf-8') as f:
    old_content = f.read()
    
with open('d:/eeg-minirocket-project/eeg-mi-bci/dashboard/app.py', 'r', encoding='utf-8') as f:
    new_content = f.read()

match_old = re.search(r'(# --- TAB 1: OVERVIEW ---.*?)(?=# --- TAB 2: MODEL ARCHITECTURES ---)', old_content, re.DOTALL)
if not match_old:
    print('Could not find old overview')
    exit(1)
old_overview = match_old.group(1)

new_content_mod = re.sub(r'(# --- TAB 1: OVERVIEW ---.*?)(?=# --- TAB 2: MODEL ARCHITECTURES ---)', old_overview.replace('\\', '\\\\'), new_content, flags=re.DOTALL)

with open('d:/eeg-minirocket-project/eeg-mi-bci/dashboard/app.py', 'w', encoding='utf-8') as f:
    f.write(new_content_mod)

print('Restored old overview tab successfully.')
