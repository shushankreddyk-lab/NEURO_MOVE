with open(r'd:\eeg-minirocket-project\eeg-mi-bci-archive-565657a\dashboard\app.py', 'r', encoding='utf-8') as f:
    content = f.read()
    print('EEGNet block exists:', 'selected_model_view == "EEGNet"' in content)
    print('Shallow ConvNet block exists:', 'selected_model_view == "Shallow ConvNet"' in content)
