## MA Divergence

| Factor | Training conclusion | optimize | Decision |
|---|---|---|---|
| ma_divergence x hist_vol | 波動率大概在60~80 分位數，且 divergence 在 15 分位數以下時，偏負 | 大概 ma 10 ~ 40 都有穩定現象，所以 ma 參數選平均，ma 25, vol_window 40 是因為40比較好 | divergence(window=25) <= -0.021, 0.0075 <= HistVol(window=40) <= 0.00963 |