import json

nb_path = 'd:/Github/Quant/projects/workStrategy/TX/TX.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'bond_2_month_df' in source and 'merge' in source:
            print(f"Found in cell index: {i}")
            print("Content:")
            print(source)
            break
