import pandas as pd
import os

# Paths
DATA_DIR = r'../../data/disposal'
STOCK_INFO_PATH = r'../../data/台股總覽.csv'

def verify():
    # Load disposal data (unfiltered)
    disposal_path = os.path.join(DATA_DIR, 'disposal_df_long.csv')
    if not os.path.exists(disposal_path):
        print(f"Error: {disposal_path} not found.")
        return

    print("Loading disposal_df_long.csv...")
    disposal_long = pd.read_csv(disposal_path, low_memory=False)
    disposal_long['Stock_id'] = disposal_long['Stock_id'].astype(str)
    
    # OLD LOGIC
    s_id = disposal_long['Stock_id']
    old_filtered = disposal_long[
        (s_id.str.len() == 4) &
        (~s_id.str.startswith('00')) &
        (~s_id.str.startswith('91'))
    ]
    print(f"Old Logic Count: {len(old_filtered)} (Removed {len(disposal_long) - len(old_filtered)})")
    
    # NEW LOGIC
    if not os.path.exists(STOCK_INFO_PATH):
        print(f"Error: {STOCK_INFO_PATH} not found.")
        return

    print("Loading stock info...")
    stock_info = pd.read_csv(STOCK_INFO_PATH)
    stock_info['Stock_id'] = stock_info['股票代碼'].astype(str)
    stock_map = stock_info.drop_duplicates(subset=['Stock_id'], keep='last')[['Stock_id', '產業別']]
    
    print("Unique Industries found:", stock_map['產業別'].unique())

    # Merge
    merged = disposal_long.merge(stock_map, on='Stock_id', how='left')
    
    # Define Exclude
    exclude_industries = [
        'ETF', '上櫃指數股票型基金(ETF)', '上櫃ETF',
        '存託憑證', 
        'ETN', '指數投資證券(ETN)',
        '受益證券',
        '債券'
    ]
    
    mask_exclude = merged['產業別'].isin(exclude_industries)
    new_filtered = merged[~mask_exclude]
    
    print(f"New Logic Count: {len(new_filtered)} (Removed {len(merged) - len(new_filtered)})")
    
    # Comparison
    old_ids = set(old_filtered['Stock_id'])
    new_ids = set(new_filtered['Stock_id'])
    
    print(f"\nIn Old but NOT New (Missed by New?): {len(old_ids - new_ids)}")
    if len(old_ids - new_ids) > 0:
        diff_ids = list(old_ids - new_ids)
        sample = merged[merged['Stock_id'].isin(diff_ids)]
        print(sample[['Stock_id', '產業別']].drop_duplicates().head(10))

    print(f"\nIn New but NOT Old (Saved by New?): {len(new_ids - old_ids)}")
    if len(new_ids - old_ids) > 0:
        diff_ids = list(new_ids - old_ids)
        sample = merged[merged['Stock_id'].isin(diff_ids)]
        print(sample[['Stock_id', '產業別']].drop_duplicates().head(10))

if __name__ == "__main__":
    verify()
