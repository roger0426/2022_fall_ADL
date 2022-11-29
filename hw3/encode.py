import json
import sys

path = sys.argv[1]
if path[0] == '.':
    path2 = '.' + path.split('.')[1] + '_utf8.' + path.split('.')[-1]
else:
    path2 = path.split('.')[0] + '_utf8.' + path.split('.')[-1]

data, refs = [], []

with open(path) as file:
    for line in file:
        line = json.loads(line)
        # refs[line['id']] = line['title'].strip() + '\n'
        refs.append(line)
# print(refs)

with open(path2, 'w') as file2:
    for row in refs:
        file2.write(f'{row}\n')