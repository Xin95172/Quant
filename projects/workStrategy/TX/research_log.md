## MA Divergence

| Factor | Training conclusion | optimize | Decision | Alpha source |
|---|---|---|---|---|
| ma_divergence x hist_vol（日盤） | HistVol 在 Q4（60~80 分位數）且 25MA divergence 在該 regime 最低 15% 時，日盤報酬偏負 | ma 10~40 都有穩定現象，選 ma 25；vol_window 40 的訊號分布較佳 | `做空` divergence(window=25) <= -0.02720，0.00749 <= HistVol(window=40) <= 0.00963 | 賺抄底太早傻逼的錢（動能延續） |
| night_divergence（夜盤） | HistVol 在 Q3（40~60 分位數）且 5MA divergence 在該 regime 最低 10% 時，夜盤報酬偏正 | vol_window 對夜盤影響不大，沿用 40；短 ma 較有效，捕捉短期 oversold 後修正 | `做多` divergence(window=5) <= -0.01479，0.00671 <= HistVol(window=40) <= 0.00749 | 賺日盤恐慌 oversold 後、於中等波動環境修正回來的錢 |
