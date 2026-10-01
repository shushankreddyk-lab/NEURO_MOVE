import os
filepath = r'C:\Users\SHUSHANK\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\streamlit\watcher\folder_black_list.py'
with open(filepath, 'r') as f:
    content = f.read()
if '**/pip_packages' not in content:
    content = content.replace('"**/site-packages",', '"**/site-packages",\n    "**/pip_packages",')
    with open(filepath, 'w') as f:
        f.write(content)
print('Patched folder_black_list.py')
