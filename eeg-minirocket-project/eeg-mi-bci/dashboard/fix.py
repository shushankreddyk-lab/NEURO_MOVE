
with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i in range(825, 835):
    if 'except Exception as e:' in lines[i]:
        lines[i] = '        except Exception as e:\n'
    if 'st.error(' in lines[i]:
        lines[i] = '            st.error(f\
Error
loading
detailed
TOC:
e
\)\n'
    if lines[i].strip() == 'else:':
        lines[i] = '            else:\n'

with open('app.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)
print('Done!')

