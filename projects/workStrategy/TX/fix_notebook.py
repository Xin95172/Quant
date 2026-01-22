import json
import re

nb_path = 'd:/Github/Quant/projects/workStrategy/TX/TX.ipynb'

def fix_code(source_lines):
    new_source = []
    # Identify if this is the problematic cell
    source_text = "".join(source_lines)
    if "temp_df = temp_df.merge(bond_1_month_df" not in source_text:
        return source_lines
    
    # Keep the top part (getting data)
    keep_lines = []
    in_merge_section = False
    
    for line in source_lines:
        if "temp_df = temp_df.merge(bond_1_month_df" in line:
            in_merge_section = True
            # Inject the new logic here
            new_source.append("\n")
            new_source.append("# Rename columns to avoid MergeError\n")
            new_source.append("bond_info = [\n")
            new_source.append("    (bond_1_month_df, '1m'), (bond_2_month_df, '2m'), (bond_3_month_df, '3m'),\n")
            new_source.append("    (bond_6_month_df, '6m'), (bond_1_year_df, '1y'), (bond_3_year_df, '3y'),\n")
            new_source.append("    (bond_5_year_df, '5y')\n")
            new_source.append("]\n")
            new_source.append("\n")
            new_source.append("for df, suffix in bond_info:\n")
            new_source.append("    if 'value' in df.columns:\n")
            new_source.append("        df = df[['date', 'value']].rename(columns={'value': f'US_bond_{suffix}'})\n")
            new_source.append("    temp_df = temp_df.merge(df, on='date', how='left')\n")
            # Skip the original merge lines
        elif in_merge_section and "temp_df = temp_df.merge(" in line:
            continue # Skip old merges
        else:
            new_source.append(line)
            
    return new_source

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        original_source = cell['source']
        if "bond_2_month_df = fm.get_US_bond" in "".join(original_source):
            print("Found target cell.")
            current_source = cell['source']
            # Re-write the source code properly
            # It's easier to just reconstruct the whole block since we know what it looks like from locate_cell.py
            
            # Constructing the new source code block
            new_code_block = [
                "bond_1_month_df = fm.get_US_bond('United States 1-Month', START, END)\n",
                "bond_2_month_df = fm.get_US_bond('United States 2-Month', START, END)\n",
                "bond_3_month_df = fm.get_US_bond('United States 3-Month', START, END)\n",
                "bond_6_month_df = fm.get_US_bond('United States 6-Month', START, END)\n",
                "bond_1_year_df = fm.get_US_bond('United States 1-Year', START, END)\n",
                "bond_3_year_df = fm.get_US_bond('United States 3-Year', START, END)\n",
                "bond_5_year_df = fm.get_US_bond('United States 5-Year', START, END)\n",
                "\n",
                "temp_df = analyzer.display_df()\n",
                "temp_df.reset_index(inplace=True)\n",
                "\n",
                "# Merge bonds with renaming to avoid duplicates\n",
                "bond_map = {\n",
                "    '1m': bond_1_month_df,\n",
                "    '2m': bond_2_month_df,\n",
                "    '3m': bond_3_month_df,\n",
                "    '6m': bond_6_month_df,\n",
                "    '1y': bond_1_year_df,\n",
                "    '3y': bond_3_year_df,\n",
                "    '5y': bond_5_year_df\n",
                "}\n",
                "\n",
                "for suffix, df in bond_map.items():\n",
                "    # Extract only necessary columns and rename\n",
                "    if 'value' in df.columns:\n",
                "        sub_df = df[['date', 'value']].rename(columns={'value': f'US_bond_{suffix}'})\n",
                "        temp_df = temp_df.merge(sub_df, on='date', how='left')\n",
                "    else:\n",
                "        # Fallback if columns are different, merge directly but be careful\n",
                "        temp_df = temp_df.merge(df, on='date', how='left')\n",
                "\n",
                "temp_df.set_index('date', inplace=True)\n",
                "\n",
                "analyzer.update_df(temp_df)\n"
            ]
            cell['source'] = new_code_block
            print("Cell updated.")
            break

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False) # indent=1 to match typical ipynb
