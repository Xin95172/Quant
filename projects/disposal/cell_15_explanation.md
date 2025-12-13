# Cell 15 (Analysis Step 3) 詳細運作原理解析

這份文件詳細解釋了 Notebook 中 **[Step 3] 執行 Event Study 分析** 的每一行程式碼功能、背後的資料處理邏輯，以及產出的資料結構。

## 1. 程式碼逐行功能

這個 Cell 雖然只有短短幾行，但卻是整個專案的「引擎」，它驅動了從資料到訊號的轉化。

```python
# 1. 讀取「股價」與「整理好的處置事件」檔
price_df = pd.read_csv('../../data/disposal/price_df.csv') 
processed_disposal = pd.read_csv('../../data/disposal/processed_disposal_events.csv')

# 2. 【核心引擎】執行事件研究法 (配對 -> 切割 -> 貼標籤)
disposal_wide, disposal_long = run_event_study(price_df, processed_disposal, offset_days=offset_days)

# 3. 檢查結果並顯示寬格式的形狀
if not disposal_wide.empty:
    print(f"Wide Format Shape: {disposal_wide.shape}")
```

### 每一行在做什麼？

| 行數 | 程式碼片段 | 功能白話文 |
| :--- | :--- | :--- |
| **L4** | `pd.read_csv(...)` | **備料**：把之前辛苦抓下來的「股價歷史資料 (`price_df`)」和「處置股名單 (`processed_disposal`)」讀進來。 |
| **L5** | `run_event_study(...)` | **加工**：這是魔法發生的地方。它呼叫 `utils.py` 裡的函式，把股價跟處置事件「對起來」，然後算出每天是處置第幾天。 |
| **L7** | `if not ... empty` | **品管**：檢查一下有沒有算出東西來，避免後面程式碼空轉報錯。 |

---

## 2. 黑盒子裡面發生了什麼？ (`run_event_study`)

當您執行第 5 行時，`utils.py` 裡面的 `run_event_study` 函式幫您做了五件苦差事：

1.  **資料配對 (Merge)**：
    *   它把 `price_df` (股價) 和 `processed_disposal` (事件) 用 `Stock_id` 連起來。
    *   這時候資料會變超大，因為一支股票的每一天股價都會被重複貼上該股票所有的處置事件資訊。

2.  **時間過濾 (Filter)**：
    *   它知道我們只關心「處置期間」以及前後 `offset_days` (例如 20 天) 的股價。
    *   所以它會把範圍外的日子全部刪掉，只留下精華片段。

3.  **座標定位 (Trading Index)**：
    *   它會建立一個「交易日索引」。例如，處置開始那天是第 0 天，開始後第 5 個交易日就是 index + 5。
    *   這比單純用日曆天 (Calendar Days) 準確，因為它會自動跳過週末和國定假日。

4.  **貼標籤 (Labeling)**：
    *   這是最重要的功能！它會幫每一行資料加上 `t_label` 欄位：
        *   `s+1`：代表這是「處置開始 (Start) 後第 1 個交易日」。
        *   `e-1`：代表這是「處置結束 (End) 前 1 個交易日」。
    *   有了這個標籤，我們才能把不同股票、不同日期的資料「疊」在一起算平均。

5.  **雙軌輸出 (Output)**：
    *   最後它把它整理成兩種格式：`Wide` (寬) 和 `Long` (長)，分別給回測和畫圖用。

---

## 3. 資料結構說明

### A. `disposal_long` (長格式) - 畫圖專用

這是 Analysis Step 2 (畫圖區塊) 的主要輸入。它的特點是 **「一天一列」**。

| Date | Stock_id | Close | t_label | daily_ret |
| :--- | :--- | :--- | :--- | :--- |
| 2023-01-01 | 2330 | 500  | s-1 | 1.2% |
| 2023-01-02 | 2330 | 510  | s+0 | 2.0% |
| 2023-01-03 | 2330 | 505  | s+1 | -0.9% |

*   **直觀理解**：這就像是把每一支被處置股票的「心電圖」都剪下來，然後標上「這是發病第幾天」。
*   **用途**：計算「平均」表現。例如我們可以把所有標記為 `s+1` 的列拿出來平均，就知道「處置第一天通常會漲還是跌」。

### B. `disposal_wide` (寬格式) - 回測專用

這是給機器學習或回測系統用的。它的特點是 **「一場事件一列」**。

| Stock_id | event_start | event_end | s-1_ret | s+0_ret | s+1_ret | ... |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2330 | 2023-01-02 | 2023-01-15 | 1.2% | 2.0% | -0.9% | ... |
| 2603 | 2021-05-01 | 2021-05-14 | 5.0% | -3.0% | 1.1% | ... |

*   **直觀理解**：這就像是每一場處置事件的「總結報告」。它把這場事件期間每天的漲跌幅都攤平變成欄位。
*   **用途**：訓練模型。每個 row 就是一個樣本 (Sample)，欄位是特徵 (Features)。

## 4. 流程圖解

```mermaid
graph TD
    A[price_df.csv<br>(所有股價)] --> C(run_event_study)
    B[processed_disposal.csv<br>(處置名單)] --> C
    
    C --> D{配對與運算}
    D -- 取範圍 --> E[Filter Date Range]
    E -- 算座標 --> F[Calculate t-index]
    F -- 貼標籤 --> G[Generate t_label]
    
    G --> H[disposal_long<br>(長格式: 畫圖用)]
    G --> I[disposal_wide<br>(寬格式: 回測用)]
```
