# Exercise 2.15: CSI Compression and Reconstruction using CsiNet and CS-CsiNet

*Wireless Communications and Machine Learning* Exercise 2.15 目標是使用 **COST2100 channel model** 產生多組不同通道資料集，評估 CsiNet 在不同通道分佈下的 CSI reconstruction NMSE，並使用混合資料集重新訓練 CsiNet，觀察模型在複雜通道環境中的泛化能力。

使用六種使用者分佈：

```text
user_center
user_edge
user_uniform
user_left_cluster
user_right_cluster
user_ring
```

---

## 1. 題目要求與完成內容

Exercise 2.15 包含三個小題：

### (a) 使用 COST2100 產生超過五組不同 channel datasets

使用 MATLAB 版 COST2100 channel model 產生六組不同使用者分佈的 CSI 資料。每組資料都切成 train、validation、test 三個 split。

每一筆 CSI 會被整理成 CsiNet 所需格式：

```text
HT shape = [num_samples, 2048]
2048 = 2 × 32 × 32
```

其中 `2` 代表 real / imaginary 兩個 channel。

### (b) 評估 trained CsiNet 在每一組 dataset 上的 NMSE

提供 `CsiNet_onlytest.py`，可讀取已訓練的 CsiNet model，並逐一測試六組 COST2100 test dataset。

若沒有原始 pretrained model，也可以使用 `CsiNet_train.py` 中的：

```python
train_dataset_mode = 'single_cost2100'
```

只使用單一分佈訓練 CsiNet，作為 baseline。

### (c) 混合不同 channel datasets 重新訓練 CsiNet 並比較

使用：

```python
train_dataset_mode = 'mixed_cost2100'
```

將六組 COST2100 training datasets 合併後訓練 CsiNet，再分別測試六組 test datasets，並與 single-dataset baseline 比較 NMSE。

---

## 2. 原理說明

### 2.1 CsiNet 架構

CsiNet 是一個用於 CSI compression and reconstruction 的 autoencoder。其主要流程如下：

```text
Original CSI → Encoder → Compressed codeword → Decoder → Reconstructed CSI
```

在此設定中，輸入 CSI 被表示為：

```text
[batch, 2, 32, 32]
```

其中：

- channel 0：real part
- channel 1：imaginary part
- `32 × 32`：CSI 空間 / 頻率維度

Encoder 使用 convolution、reshape 與 dense layer 將 CSI 壓縮成低維向量。Decoder 使用 dense layer、reshape 與 residual blocks 重建原始 CSI。

使用設定：

```text
encoded_dim = 512
compression rate = 1/4
residual_num = 2
```

### 2.2 NMSE 評估指標

主要評估 CSI reconstruction NMSE：

```text
NMSE = E[ ||H - H_hat||² / ||H||² ]
NMSE(dB) = 10 log10(NMSE)
```

其中：

- `H`：原始 CSI
- `H_hat`：CsiNet reconstructed CSI
- NMSE 越低越好
- dB 數值越負代表重建效果越好

產生的資料只輸出 `HT`，沒有額外輸出 `HF_all`，因此 correlation 欄位顯示為 `NaN`。主要比較指標為 NMSE。

### 2.3 為什麼要做 mixed training？

單一分佈訓練的 CsiNet 容易只學到某一種通道型態。實際無線系統中，使用者位置、移動方向、散射環境與場景都會改變，因此模型需要具備較好的泛化能力。

混合多種 COST2100 channel datasets 訓練，可以讓 CsiNet 在訓練階段看到更多通道變化，因此有機會在不同使用者分佈下得到更穩定的 NMSE。

---

## 3. 檔案結構

```text
.
├── README.md
├── CsiNet_train.py
├── CsiNet_onlytest.py
├── CS-CsiNet_train.py
├── CS-CsiNet_onlytest.py
├── matlab/
│   └── generate_cost2100_csinet_data.m
└── result/
    ├── final_single_1500epochs.csv
    ├── final_mixed_1500epochs.csv
    └── final_compare_1500epochs.csv

```



## 4. 環境建立

### 4.1 Python / Colab 套件

主要在 Google Colab 執行。

```bash
pip install 以下內容：

tensorflow
numpy
scipy
matplotlib
pandas
h5py
```

### 4.2 Colab GPU 設定

在 Colab 上方選單設定：

```text
Runtime → Change runtime type → Hardware accelerator → T4 GPU → Save
```

確認 GPU：

```python
!nvidia-smi
```

使用設備：

```text
GPU: NVIDIA Tesla T4
```

執行時間約為：

```text
Part (b) single / baseline evaluation: 約 10 分鐘
Part (c) mixed training, 1500 epochs: 約 40 分鐘
```

實際時間會依 Colab 當下 GPU 狀態、batch size 與資料大小而不同。

---

## 5. MATLAB / COST2100 資料產生

下載 COST2100 MATLAB code：

```text
https://github.com/cost2100/cost2100
```

解壓縮後放到：

```text
C:\Users\user\Desktop\AIwireless\cost2100-master
```

在 MATLAB 執行：

```matlab
cd('C:\Users\user\Desktop\AIwireless')
addpath(genpath('cost2100-master'))
```

接著執行資料產生 script：

```matlab
matlab/generate_cost2100_csinet_data.m
```

若 MATLAB 的 `savepath` 出現權限 warning，可以忽略，只要目前 session 已經 `addpath` 成功即可。

產生的資料會存到：

```text
CsiNetData/
```

---

## 6. 需要的資料檔案

程式會讀取以下 18 個 `.mat` 檔案：

```text
DATA_Htrainin_user_center.mat
DATA_Hvalin_user_center.mat
DATA_Htestin_user_center.mat
DATA_Htrainin_user_edge.mat
DATA_Hvalin_user_edge.mat
DATA_Htestin_user_edge.mat
DATA_Htrainin_user_uniform.mat
DATA_Hvalin_user_uniform.mat
DATA_Htestin_user_uniform.mat
DATA_Htrainin_user_left_cluster.mat
DATA_Hvalin_user_left_cluster.mat
DATA_Htestin_user_left_cluster.mat
DATA_Htrainin_user_right_cluster.mat
DATA_Hvalin_user_right_cluster.mat
DATA_Htestin_user_right_cluster.mat
DATA_Htrainin_user_ring.mat
DATA_Hvalin_user_ring.mat
DATA_Htestin_user_ring.mat
```

每個 `.mat` 內含變數：

```text
HT
```

`HT` shape：

```text
train: [3000, 2048]
val:   [600, 2048]
test:  [600, 2048]
```

六組合併後：

```text
training samples = 18,000
validation samples = 3,600
test samples per dataset = 600
```

---

## 7. Google Colab 執行流程

### 7.1 Clone GitHub repository

```python
!git clone https://github.com/WANG-RUI-CHENG/Exercise-2.15-CSI-Compression-and-Reconstruction-using-CsiNet-and-CS-CsiNet.git
%cd Exercise-2.15-CSI-Compression-and-Reconstruction-using-CsiNet-and-CS-CsiNet
!ls
```

### 7.2 安裝套件

```python
!pip install 

tensorflow
numpy
scipy
matplotlib
pandas
h5py
```

### 7.3 建立資料夾

```python
!mkdir -p data result saved_model
```

### 7.4 從 Google Drive 複製資料

先將 MATLAB 產生的 `CsiNetData` 壓縮為 `CsiNetData.zip`，上傳到 Google Drive。

```python
from google.colab import drive
drive.mount('/content/drive')

!cp /content/drive/MyDrive/CsiNetData.zip .
!unzip -o CsiNetData.zip
!cp CsiNetData/*.mat data/
!ls data | wc -l
```

正常應顯示：

```text
18
```

---

## 8. 執行程式

### 8.1 Part (b)：single-dataset baseline

若要做 baseline，可在 `CsiNet_train.py` 中設定：

```python
train_dataset_mode = 'single_cost2100'
```

然後執行：

```python
!python CsiNet_train.py
```

完成後將結果另存成：

```text
result/final_single_1500epochs.csv
```

### 8.2 Part (c)：mixed-dataset training

正式 mixed training 設定：

```python
use_cost2100_multi_dataset = True
train_dataset_mode = 'mixed_cost2100'
evaluate_all_cost2100_datasets = True
epochs = 1500
encoded_dim = 512
```

執行：

```python
!python CsiNet_train.py
```

完成後將結果另存成：

```text
result/final_mixed_1500epochs.csv
```

### 8.3 讀取結果表格

```python
import pandas as pd

single = pd.read_csv('result/final_single_1500epochs.csv')
mixed = pd.read_csv('result/final_mixed_1500epochs.csv')
compare = pd.read_csv('result/final_compare_1500epochs.csv')

display(single)
display(mixed)
display(compare)
```

---

## 9. 模擬結果

### 9.1 Single-dataset baseline 結果

此結果使用 `user_center` 作為單一分佈訓練資料，然後測試六組使用者分佈。

| Dataset | Single NMSE (dB) | Test Samples |
|---|---:|---:|
| `user_center` | 2.124157 | 600 |
| `user_edge` | 2.115973 | 600 |
| `user_uniform` | 2.120627 | 600 |
| `user_left_cluster` | 2.102291 | 600 |
| `user_right_cluster` | 2.097350 | 600 |
| `user_ring` | 2.106925 | 600 |

### 9.2 Mixed-dataset training 結果

此結果使用六組 COST2100 datasets 混合訓練，然後測試六組使用者分佈。

| Dataset | Mixed NMSE (dB) | Test Samples |
|---|---:|---:|
| `user_center` | -1.228356 | 600 |
| `user_edge` | -1.248122 | 600 |
| `user_uniform` | -1.235212 | 600 |
| `user_left_cluster` | -1.243599 | 600 |
| `user_right_cluster` | -1.248775 | 600 |
| `user_ring` | -1.245125 | 600 |

平均 mixed NMSE：

```text
約 -1.2415 dB
```

### 9.3 Single vs Mixed 比較

`mixed_minus_single_db < 0` 代表 mixed training 的 NMSE 更低，效果較好。

| Dataset | Single NMSE (dB) | Mixed NMSE (dB) | Mixed - Single (dB) |
|---|---:|---:|---:|
| `user_center` | 2.124157 | -1.228356 | -3.352513 |
| `user_edge` | 2.115973 | -1.248122 | -3.364095 |
| `user_uniform` | 2.120627 | -1.235212 | -3.355839 |
| `user_left_cluster` | 2.102291 | -1.243599 | -3.345890 |
| `user_right_cluster` | 2.097350 | -1.248775 | -3.346125 |
| `user_ring` | 2.106925 | -1.245125 | -3.352050 |

結果顯示 mixed training 在六組測試分佈上都比 single-dataset baseline 更低，約改善 `3.35 dB`。這代表多分佈訓練有助於提升 CsiNet 對不同通道分佈的泛化能力。

---

## 10. 結論與討論

1. 使用 COST2100 產生六組不同使用者分佈的通道資料後，可以觀察 CsiNet 在不同通道分佈下的重建能力。
2. Single-dataset training 只看到一種通道分佈，因此泛化能力較弱。
3. Mixed-dataset training 將多組使用者分佈合併訓練，使模型能學到更多樣的通道特徵。
4. 實驗結果中，mixed training 在所有測試分佈上都得到更低 NMSE，平均約改善 3.35 dB。
5. 實際系統中，通道環境會隨使用者位置、移動方向、散射環境與室內/室外場景改變，因此 CSI feedback 方法應使用多場景資料訓練，並可搭配資料增強、domain randomization 或少量新環境資料 fine-tuning 改善泛化能力。

---

## 11. 備註

主要比較 NMSE，因此 MATLAB script 只輸出 `HT`。若要計算 correlation coefficient，需要額外輸出頻域 CSI `HF_all`，並放入：

```text
DATA_HtestFin_user_center_all.mat
DATA_HtestFin_user_edge_all.mat
...
```

若未提供 `HF_all`，程式仍可正常計算 NMSE，correlation 欄位會顯示為 `NaN`。

---
## 12. colab 模擬結果截圖

```text
Mixed-dataset training 結果
```
<img width="817" height="481" alt="image" src="https://github.com/user-attachments/assets/3fd0a107-cc84-403b-aede-fecc84be9d4a" />

```text
Single vs Mixed 比較
```

<img width="828" height="450" alt="image" src="https://github.com/user-attachments/assets/3eb4605a-9fa0-48d4-b03d-b5ad12f86268" />
