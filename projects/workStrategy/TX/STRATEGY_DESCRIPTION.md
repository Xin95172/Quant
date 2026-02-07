# 長天期台指期策略說明（TX 夜/日盤雙段）

## 資料來源與預處理
- 價格：台指期日盤/夜盤 OHLC（`Close`, `Close_a`）。
- 外資選擇權部位：
  - 日盤：`foreign_opt_pos` = (多 CALL - 空 CALL - 多 PUT + 空 PUT) / 總成交金額。
  - 夜盤：`foreign_opt_pos_a` 同上，但使用夜盤成交金額。
- 其他：交易日曆預先補齊索引，夜盤信號可 shift 至隔日。

## 指標計算
- `3ma`：`Close_a` 的 3 日移動平均。
- `divergence`：`Close_a / 3ma - 1`，衡量現價相對均線的乖離，>0 表示價位高於短均線、<0 表示偏弱。
- `foreign_opt_pos_a`：外資夜盤選擇權淨額（多 CALL - 空 CALL - 多 PUT + 空 PUT）除以夜盤總成交金額，反映夜盤外資多空傾向，介於 -1 ~ 1。
- `foreign_opt_pos`：同上，但使用日盤成交金額。
- `opt_pos_continue`：外資選擇權部位的延續性指標  
  `opt_pos_continue = foreign_opt_pos_a + foreign_opt_pos_a.shift(1) + foreign_opt_pos`，再在夜盤信號應用處 shift(1)，代表用「昨夜＋前夜＋當日日盤」的總和，隔一日後決定夜盤方向。

## 進出場邏輯
### 夜盤倉位 `pos_night`（訊號生效延後一天）
- 若 `opt_pos_continue < 0.012` → 建立多頭倉位 `pos_night = 1`。
- 若 `opt_pos_continue > 0.012` → 保持空倉 `pos_night = 0`。
- 信號計算後整體 `shift(1)`，避免用到當日未完成資料，並用前一日資訊決定今日夜盤。

### 日盤倉位 `pos_day`（訊號當天生效）
- 外資夜盤部位過度偏空：`foreign_opt_pos_a < -0.0035` → `pos_day = -1`（做空）。
- 若未過度偏空且乖離不弱：`foreign_opt_pos_a >= -0.0035` 且 `divergence > -0.05` → `pos_day = 1`（做多）。
- 其他情況維持原值（預設 0）。

## 訊號時間軸
- 日盤訊號：使用當日夜盤部位與當日價格乖離，當天日盤生效。
- 夜盤訊號：使用「當日夜盤 + 前一晚夜盤 + 當日日盤」計算後，整體往後 shift 一天，避免前視。

## 風險與資料品質檢查
- 若外資選擇權部位缺值（尤其是日盤未更新），會以期交所資料補齊；索引會預先補上今日以避免 KeyError。
- 關鍵欄位缺值時，訊號可能留空或無倉位；需在回測/實盤前確認資料完整性。

## 簡化流程（pseudo-code）
```python
# 計算指標
foreign_opt_pos_a = f_aftermarket_positions()  # 比例化後
foreign_opt_pos   = f_day_positions()
divergence = Close_a / Close_a.rolling(3).mean() - 1
opt_pos_continue = foreign_opt_pos_a + foreign_opt_pos_a.shift(1) + foreign_opt_pos

# 夜盤訊號（隔日生效）
pos_night = 0
pos_night[opt_pos_continue < 0.012] = 1
pos_night[opt_pos_continue > 0.012] = 0
pos_night = pos_night.shift(1)

# 日盤訊號（當日生效）
pos_day = 0
pos_day[foreign_opt_pos_a < -0.0035] = -1
pos_day[(foreign_opt_pos_a >= -0.0035) & (divergence > -0.05)] = 1
```

## 使用注意
- 確保資料包含明天的交易日（夜盤 shift 後有落點）。
- 若要在 GitHub 顯示圖表，可用 Plotly+kaleido 同時輸出 HTML 與 PNG。
