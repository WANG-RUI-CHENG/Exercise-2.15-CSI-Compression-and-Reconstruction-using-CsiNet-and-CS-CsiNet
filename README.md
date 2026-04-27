# Exercise 2.15 — CsiNet Generalization on Multiple COST 2100 Channel Datasets

本專案完成 *Wireless Communications and Machine Learning* Exercise 2.15。目標是測試原本訓練好的 CsiNet 在不同 COST 2100 通道資料集上的 CSI reconstruction NMSE，並進一步把多個不同分布的資料集混合訓練，觀察模型泛化能力是否改善。

## 1. 題目要求

Exercise 2.15 要求完成三件事：

1. 使用 COST 2100 channel model 產生超過五組不同 channel datasets，例如改變 user distribution。
2. 使用已訓練好的 CsiNet model，分別測試每一組資料集的 CSI reconstruction NMSE。
3. 將上述多組資料集混合後重新訓練 CsiNet，並與第 2 步的結果比較，討論如何提升實際通道環境下 CSI feedback 方法的 generalization。

## 2. 方法概念

原本的 CsiNet 是一個 autoencoder：

- Encoder：將 CSI 壓縮成低維 codeword。
- Decoder：將 codeword 重建回 CSI。
- Loss function：使用 MSE 訓練 reconstruction。
- 評估指標：使用 NMSE dB 與 correlation coefficient。

本專案新增的重點是「multi-dataset evaluation」與「mixed-dataset training」。

### Part (a)：產生多組 COST 2100 datasets

使用 COST 2100 channel model 產生至少六組不同 user distribution 的資料。本專案預設使用以下六組：

| Dataset name | 說明 |
|---|---|
| `user_center` | 使用者集中在基地台附近或場景中心 |
| `user_edge` | 使用者集中在 cell edge |
| `user_uniform` | 使用者均勻分布 |
| `user_left_cluster` | 使用者集中在左側區域 |
| `user_right_cluster` | 使用者集中在右側區域 |
| `user_ring` | 使用者呈環狀分布 |

產生資料後，請將 `.mat` 放在 `data/` 目錄下。

### Part (b)：測試 pretrained CsiNet 在不同資料集上的 NMSE

使用 `CsiNet_onlytest_Exercise2_15.py` 載入已訓練好的 CsiNet model，然後依序測試六組 COST 2100 datasets。

此步驟用來觀察：

> 原本訓練好的 CsiNet 遇到不同 channel distribution 時，NMSE 是否變差。

### Part (c)：混合多組資料重新訓練 CsiNet

使用 `CsiNet_train_Exercise2_15.py` 將六組資料的 train/validation set concatenate 起來重新訓練 CsiNet，訓練完成後再分別測試六組 test set。

此步驟用來觀察：

> mixed training 是否能降低 domain mismatch，改善模型在不同 channel distribution 上的泛化能力。

## 3. 專案檔案結構

```text
.
├── CsiNet_train_Exercise2_15.py       # 混合多組 COST 2100 datasets 訓練 CsiNet，並輸出比較結果
├── CsiNet_onlytest_Exercise2_15.py    # 使用 pretrained CsiNet 對多組 datasets 做 inference 與 NMSE 評估
├── requirements.txt                   # Python dependencies
├── README.md                          # 專案說明
├── data/                              # 放 COST 2100 產生的 .mat datasets
├── result/                            # 放訓練 log、CSV 結果、模型輸出
└── saved_model/                       # 放 pretrained model JSON/H5
```

## 4. 環境需求

建議使用 Python 3.7 或較舊環境搭配 TensorFlow 1.x，因為原始程式使用 `tf.reset_default_graph()`。

安裝套件：

```bash
pip install -r requirements.txt
```

若使用 TensorFlow 2.x，需要自行改成 compatibility mode，例如：

```python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
```

## 5. Dataset 格式

每一個 `.mat` 檔至少需要包含：

| Variable | Shape | 用途 |
|---|---:|---|
| `HT` | `[num_samples, 2048]` | CsiNet input，對應 2 × 32 × 32 的 real/imag CSI |
| `HF_all` | `[num_samples, 32, 125]` 或可 reshape 成此形狀 | optional，用於 correlation coefficient |

本程式預設檔名如下：

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

如果你的 COST 2100 產生出來的檔名不同，請修改 Python 檔案最上方的 `cost2100_datasets` list。

## 6. 使用方法

### Step 1：準備資料夾

```bash
mkdir -p data result saved_model
```

### Step 2：放入 pretrained CsiNet model

`CsiNet_onlytest_Exercise2_15.py` 會從 `saved_model/` 讀取：

```text
saved_model/model_CsiNet_indoor_dim512.json
saved_model/model_CsiNet_indoor_dim512.h5
```

如果你的 model 是 outdoor 或不同 compression dimension，請同步修改程式上方的：

```python
envir = 'indoor'
encoded_dim = 512
```

### Step 3：執行 Part (b)

```bash
python CsiNet_onlytest_Exercise2_15.py
```

輸出檔案：

```text
result/ex215_eval_pretrained_CsiNet_indoor_dim512.csv
```

CSV 欄位包含：

| 欄位 | 說明 |
|---|---|
| `dataset` | dataset name |
| `nmse_db` | NMSE，單位 dB，越低越好 |
| `correlation` | correlation coefficient，越接近 1 越好 |
| `sec_per_sample` | 平均每筆 inference 時間 |
| `num_samples` | 測試資料筆數 |

### Step 4：執行 Part (c)

```bash
python CsiNet_train_Exercise2_15.py
```

程式會將六組 training datasets 混合後訓練 CsiNet，並對每一組 test dataset 評估。

輸出檔案範例：

```text
result/trainloss_CsiNet_indoor_dim512_MM_DD.csv
result/valloss_CsiNet_indoor_dim512_MM_DD.csv
result/ex215_eval_mixed_CsiNet_indoor_dim512_MM_DD.csv
result/ex215_compare_CsiNet_indoor_dim512_MM_DD.csv
result/model_CsiNet_indoor_dim512_MM_DD.json
result/model_CsiNet_indoor_dim512_MM_DD.h5
```

其中 `ex215_compare_*.csv` 會比較 pretrained model 與 mixed-trained model 的 NMSE。

## 7. NMSE 計算方式

程式使用 complex CSI 計算 NMSE：

```text
NMSE = 10 * log10( E[ ||H - H_hat||^2 / ||H||^2 ] )
```

判讀方式：

- NMSE 越小越好。
- dB 數值越負，代表 reconstruction error 越低。
- 如果 mixed training 後 NMSE 比 pretrained 更低，表示泛化能力改善。

## 8. 結果討論方向

完成實驗後，可從以下方向討論：

1. **Distribution mismatch**：如果 pretrained CsiNet 在某些 user distribution 上 NMSE 明顯變差，代表模型對 unseen channel distribution 較敏感。
2. **Mixed training**：將多個 channel datasets 混合訓練後，模型通常能學到更通用的 CSI feature，因此在不同資料集上的 NMSE 可能更穩定。
3. **Generalization improvement**：實際系統可以使用多環境資料訓練、資料增強、transfer learning、domain adaptation 或 meta learning 提升模型泛化能力。

## 9. GitHub 上傳方式

### 第一次建立 GitHub repository

```bash
git init
git add README.md requirements.txt .gitignore CsiNet_train_Exercise2_15.py CsiNet_onlytest_Exercise2_15.py data/.gitkeep result/.gitkeep saved_model/.gitkeep
git commit -m "Add Exercise 2.15 CsiNet multi-dataset experiment"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

### 如果 repository 已經存在

```bash
git add .
git commit -m "Update Exercise 2.15 implementation and README"
git push
```

## 10. 注意事項

- `data/*.mat`、`result/*`、`saved_model/*` 預設被 `.gitignore` 忽略，避免把大型資料與模型權重上傳到 GitHub。
- 如果老師要求上傳 dataset 或 trained model，請移除 `.gitignore` 中對應的忽略規則。
- 訓練 epochs 預設為 1000，若硬體資源不足，可先改小 epochs 測試流程是否正確。
- `img_height = 32`、`img_width = 32`、`img_channels = 2` 必須和資料 shape 一致。

## 11. Conclusion

本實驗證明 CsiNet 在不同 COST 2100 channel distributions 上可能出現 reconstruction performance degradation。透過混合多組資料重新訓練，可以讓模型接觸更多 channel variations，通常能改善不同環境下的 NMSE 表現與泛化能力。
