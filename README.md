# Quant

This repo is for research, analysis, backtests, and derived outputs.

## Data Rule

- Read shared data from `/Users/xinc/GitHub/google_drive/Data`.
- Write only derived research outputs or project artifacts.
- Do not download or scrape source data from this repo.
- Do not import provider clients from `/Users/xinc/GitHub/note/module`.
- Use `pandas.read_parquet`, `pandas.read_csv`, `pandas.read_pickle`, or the
  local helpers in `cloud_data.py`.

Source data updates belong in `/Users/xinc/GitHub/note`.

`cloud_data.py` contains the shared path table, local readers, and a lightweight
HTTP guard so research notebooks do not silently refresh remote data.

When a shared source file is missing, update it from note:

```bash
cd /Users/xinc/GitHub/note
python scripts/data_updates/update_quant_market_data.py
```
