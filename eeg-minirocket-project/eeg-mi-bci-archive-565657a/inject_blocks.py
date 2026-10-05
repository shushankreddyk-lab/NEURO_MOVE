import re

filepath = r'd:\eeg-minirocket-project\eeg-mi-bci-archive-565657a\dashboard\app.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

eegnet_block = """elif selected_model_view == "EEGNet":
    st.markdown('''
    <div class="glass-card" style="margin-bottom:24px;">
    <h3 style="color:#a855f7; font-size:1.1rem; text-transform:uppercase; letter-spacing:0.08em; margin-top:0;">EEGNet Training Execution</h3>
    <p style="color:#8a99a8; font-size:0.95rem; margin-bottom:12px;">EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces.</p>
    </div>
    ''', unsafe_allow_html=True)
elif selected_model_view == "Shallow ConvNet":
    st.markdown('''
    <div class="glass-card" style="margin-bottom:24px;">
    <h3 style="color:#a855f7; font-size:1.1rem; text-transform:uppercase; letter-spacing:0.08em; margin-top:0;">Shallow ConvNet Training Execution</h3>
    <p style="color:#8a99a8; font-size:0.95rem; margin-bottom:12px;">Shallow Convolutional Network inspired by FBCSP, optimized for decoding band power features.</p>
    </div>
    ''', unsafe_allow_html=True)
"""

# There are two instances of "13-Layer CNN-LSTM" block in app.py. We need to find them and append the new blocks after.
# A simpler way is to find the end of the CNN-LSTM block.

content = re.sub(
    r'(elif selected_model_view == "13-Layer CNN-LSTM":.*?</div>\s*\'\'\', unsafe_allow_html=True\))',
    r'\1\n' + eegnet_block,
    content,
    flags=re.DOTALL
)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print("Injected EEGNet and Shallow ConvNet blocks.")
