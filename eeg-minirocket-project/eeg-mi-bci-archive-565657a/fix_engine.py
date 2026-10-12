import re

with open(r'd:\eeg-minirocket-project\eeg-mi-bci-archive-565657a\src\advanced_eeg_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

new_block = '''            if progress_callback:
                progress_callback(epoch + 1, avg_train_loss, train_acc, avg_val_loss, val_acc)
            else:
                print(json.dumps({
                    "type": "epoch",
                    "epoch": f"{epoch + 1}/{self.epochs}",
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "train_acc": train_acc,
                    "val_acc": val_acc
                }), flush=True)'''

new_content = re.sub(r'            print\(json\.dumps\(\{.*?"val_acc": val_acc\s*\}\), flush=True\)', new_block, content, flags=re.DOTALL)

with open(r'd:\eeg-minirocket-project\eeg-mi-bci-archive-565657a\src\advanced_eeg_engine.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print('Done fixing advanced_eeg_engine.py')
