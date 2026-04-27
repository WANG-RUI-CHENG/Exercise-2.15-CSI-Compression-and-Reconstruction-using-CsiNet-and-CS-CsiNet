# Exercise 2.15: CSI Compression and Reconstruction using CsiNet and CS-CsiNet

本資料夾維持原始 `wcmlbook/ch2/Exercise_2.15` 的繳交格式與檔名，包含：

```text
CS-CsiNet_onlytest.txt
CS-CsiNet_train.txt
CsiNet_onlytest.txt
CsiNet_train.txt
README.md
```

本次 Exercise 2.15 的主要修改放在 `CsiNet_train.txt` 與 `CsiNet_onlytest.txt`。`CS-CsiNet_train.txt` 與 `CS-CsiNet_onlytest.txt` 保留原始版本，放在資料夾中是為了維持原本 Exercise 2.15 repository 的完整格式。

---

## 1. 題目目標

Exercise 2.15 要求測試 CsiNet 在不同 COST 2100 channel datasets 上的 CSI reconstruction performance，並觀察混合多種 channel distribution 進行訓練後，模型的 generalization 是否改善。

題目分成三個部分：

1. 使用 COST 2100 channel model 產生超過五組不同 channel datasets，例如改變 user distribution。
2. 使用已訓練好的 CsiNet model，分別測試每一組 dataset 的 CSI reconstruction NMSE。
3. 將多組不同 channel datasets 混合後重新訓練 CsiNet，並與第 2 步的結果比較。

---

## 2. 修改重點

### `CsiNet_onlytest.txt`

此檔案對應題目 part (b)。

新增功能：

- 支援一次測試超過五組 COST 2100 datasets。
- 對每一組 dataset 計算 NMSE。
- 如果有 `HF_all`，同時計算 correlation coefficient。
- 將所有測試結果輸出成 CSV。

輸出檔案：

```text
result/ex215_eval_pretrained_CsiNet_indoor_dim512.csv
```

### `CsiNet_train.txt`

此檔案對應題目 part (c)。

新增功能：

- 支援讀取多組 COST 2100 training/validation datasets。
- 將多組資料 concatenate 成 mixed training dataset。
- 使用混合後的資料重新訓練 CsiNet。
- 訓練完成後，分別在每一組 test dataset 上測試 NMSE。
- 若 part (b) 的 CSV 已存在，會自動產生比較結果。

輸出檔案範例：

```text
result/ex215_eval_mixed_CsiNet_indoor_dim512_MM_DD.csv
result/ex215_compare_CsiNet_indoor_dim512_MM_DD.csv
```

---

## 3. 預設資料集設定

程式預設使用六組不同 user distribution 的 COST 2100 datasets：

```text
user_center
user_edge
user_uniform
user_left_cluster
user_right_cluster
user_ring
```

請將 COST 2100 產生的 `.mat` 檔放在 `data/` 資料夾中。

---

## 4. `.mat` 檔案命名格式

每一組 dataset 需要 train、validation、test 三種資料。程式預設檔名如下：

```text
data/DATA_Htrainin_user_center.mat
data/DATA_Hvalin_user_center.mat
data/DATA_Htestin_user_center.mat
data/DATA_HtestFin_user_center_all.mat

data/DATA_Htrainin_user_edge.mat
data/DATA_Hvalin_user_edge.mat
data/DATA_Htestin_user_edge.mat
data/DATA_HtestFin_user_edge_all.mat

data/DATA_Htrainin_user_uniform.mat
data/DATA_Hvalin_user_uniform.mat
data/DATA_Htestin_user_uniform.mat
data/DATA_HtestFin_user_uniform_all.mat

data/DATA_Htrainin_user_left_cluster.mat
data/DATA_Hvalin_user_left_cluster.mat
data/DATA_Htestin_user_left_cluster.mat
data/DATA_HtestFin_user_left_cluster_all.mat

data/DATA_Htrainin_user_right_cluster.mat
data/DATA_Hvalin_user_right_cluster.mat
data/DATA_Htestin_user_right_cluster.mat
data/DATA_HtestFin_user_right_cluster_all.mat

data/DATA_Htrainin_user_ring.mat
data/DATA_Hvalin_user_ring.mat
data/DATA_Htestin_user_ring.mat
data/DATA_HtestFin_user_ring_all.mat
```

如果自己產生的檔名不同，只要修改 `CsiNet_train.txt` 與 `CsiNet_onlytest.txt` 上方的 `cost2100_datasets` list 即可。

---

## 5. `.mat` 變數格式

每一個 `.mat` 檔至少需要包含：

| Variable | 說明 |
|---|---|
| `HT` | CsiNet 使用的 normalized CSI data，shape 應可 reshape 成 `[num_samples, 2, 32, 32]` |
| `HF_all` | optional，用於計算 frequency-domain correlation coefficient |

其中 `HT` 通常為：

```text
[num_samples, 2048]
```

因為：

```text
2048 = 2 × 32 × 32
```

---

## 6. 執行方式

原始 repository 使用 `.txt` 存放 Python code。實際執行時可選擇其中一種方式：

### 方法一：直接複製成 `.py` 後執行

Windows：

```bash
copy CsiNet_onlytest.txt CsiNet_onlytest.py
copy CsiNet_train.txt CsiNet_train.py
```

macOS / Linux：

```bash
cp CsiNet_onlytest.txt CsiNet_onlytest.py
cp CsiNet_train.txt CsiNet_train.py
```

接著執行：

```bash
python CsiNet_onlytest.py
python CsiNet_train.py
```

### 方法二：若你的環境允許，也可以直接執行 txt

```bash
python CsiNet_onlytest.txt
python CsiNet_train.txt
```

---

## 7. 執行 part (b)

請先準備 trained CsiNet model，並放在 `saved_model/` 中：

```text
saved_model/model_CsiNet_indoor_dim512.json
saved_model/model_CsiNet_indoor_dim512.h5
```

然後執行：

```bash
python CsiNet_onlytest.txt
```

此步驟會測試原本訓練好的 CsiNet 在六組不同 COST 2100 datasets 上的 NMSE。

---

## 8. 執行 part (c)

確認六組 train/validation/test datasets 都已經放進 `data/` 後，執行：

```bash
python CsiNet_train.txt
```

此步驟會把六組 training data 混合後重新訓練 CsiNet，並逐一測試每一組 test dataset。

---

## 9. NMSE 判讀方式

程式計算 complex CSI 的 normalized mean square error：

```text
NMSE = 10 * log10( E[ ||H - H_hat||^2 / ||H||^2 ] )
```

判讀方式：

- NMSE 越小越好。
- dB 數值越負，代表 reconstruction error 越低。
- 如果 mixed training 後 NMSE 下降，表示模型對不同 channel distribution 的 generalization 變好。

---

## 10. 結果討論方向

完成實驗後，可從以下方向討論：

1. 原本 pretrained CsiNet 若只在單一 channel distribution 上訓練，遇到不同 user distribution 時 NMSE 可能變差。
2. 多個 COST 2100 datasets 混合訓練可以降低 distribution mismatch。
3. 實際系統若要提升 CSI feedback 的泛化能力，可以使用 multi-scenario training、data augmentation、transfer learning 或 domain adaptation。
