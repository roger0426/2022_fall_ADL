import json

path = r'hw3_data/public.jsonl'
path2 = r'hw3_data/public_2.jsonl'

data = []
refs = []
with open(path) as file:
    for line in file:
        line = json.loads(line)
        # refs[line['id']] = line['title'].strip() + '\n'
        refs.append(line)
print(refs)

with open(path2, 'w') as file2:
    for row in refs:
        file2.write(f'{row}\n')