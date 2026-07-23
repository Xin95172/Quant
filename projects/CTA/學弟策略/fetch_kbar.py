"""Deprecated Quant-side entrypoint for 1-minute Taiwan stock k-bar updates.

Data downloads are owned by the note repo. Run this instead:

    python /Users/xinc/GitHub/note/scripts/data_updates/fetch_tw_stock_kbar_1min.py
"""

from __future__ import annotations


def main() -> None:
    raise RuntimeError(
        "Quant is read/write-only for data. "
        "Run /Users/xinc/GitHub/note/scripts/data_updates/"
        "fetch_tw_stock_kbar_1min.py to download or refresh k-bar data."
    )


if __name__ == "__main__":
    main()
