## MA Divergence

| Factor | Training conclusion | optimize | Decision |
|---|---|---|---|
| ma_divergence x hist_vol（日盤） | 波動率大概在60~80 分位數，且 divergence 在 15 分位數以下時，偏負 | 大概 ma 10 ~ 40 都有穩定現象，所以 ma 參數選平均，ma 25, vol_window 40 是因為40比較好 | `做空`divergence(window=25) <= -0.021, 0.0075 <= HistVol(window=40) <= 0.00963 |
| night_divergence (夜盤) | 高波動 (Q5, 80~100 分位數) 且 5MA divergence 在最低 10 分位數時，夜盤報酬偏正 | vol_window 對夜盤影響不大，選跟日盤一樣的 40；ma_window 短一點比較有效，因為是捕捉極端短期補漲 | `做多` divergence(window=5) <= -0.018, HistVol(window=40) >= Q5 |