# Exercise 2.15 — CsiNet 在多組 COST2100 通道資料集上的泛化能力測試

本專案完成 *Wireless Communications and Machine Learning* Exercise 2.15。目標是使用 COST2100 channel model 產生多組不同使用者分佈的 CSI 資料集，測試 CsiNet 在不同通道分佈下的 CSI reconstruction NMSE，並使用混合資料集重新訓練 CsiNet，觀察模型泛化能力。

## 1. 作業目標

Exercise 2.15 分成三個部分：

1. 使用 COST2100 channel model 產生超過五組不同 channel datasets。
2. 使用已訓練 CsiNet 模型，在每一組資料集上計算 CSI reconstruction NMSE。
3. 將多組 channel datasets 混合後重新訓練 CsiNet，並比較 NMSE，討論對實際複雜通道環境的泛化能力。

本專案使用六種使用者分佈：`user_center`、`user_edge`、`user_uniform`、`user_left_cluster`、`user_right_cluster`、`user_ring`。

## 2. 專案檔案結構

建議 GitHub repository 結構如下：

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
    └── final_mixed_1500epochs.csv
```

不建議上傳大型檔案：`data/*.mat`、`CsiNetData.zip`、`saved_model/*.h5`、`saved_model/*.weights.h5`、`result/model_*.weights.h5`。

## 3. MATLAB / COST2100 資料產生

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
generate_cost2100_csinet_data
```

若 `savepath` 出現權限 warning，可以忽略。

產生的資料會存到 `CsiNetData/`。每個 `.mat` 檔案內含 `HT`，shape 為 `[num_samples, 2048]`，因為 CsiNet 使用 `2 × 32 × 32 = 2048`。

需要產生的 18 個檔案：

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

## 4. Google Colab 執行環境

開啟 GPU：

```text
Runtime → Change runtime type → T4 GPU → Save
```

確認 GPU：

```python
!nvidia-smi
```

複製 GitHub 專案：

```python
!git clone https://github.com/WANG-RUI-CHENG/Exercise-2.15-CSI-Compression-and-Reconstruction-using-CsiNet-and-CS-CsiNet.git
%cd Exercise-2.15-CSI-Compression-and-Reconstruction-using-CsiNet-and-CS-CsiNet
!ls
```

安裝套件：

```python
!pip install scipy matplotlib pandas h5py
```

建立資料夾：

```python
!mkdir -p data result saved_model
```

## 5. 將資料放入 Colab

先將 MATLAB 產生的 `CsiNetData` 壓縮成 `CsiNetData.zip`，上傳到 Google Drive。

```python
from google.colab import drive
drive.mount('/content/drive')

!cp /content/drive/MyDrive/CsiNetData.zip .
!unzip -o CsiNetData.zip
!cp CsiNetData/*.mat data/
!ls data | wc -l
```

正常應顯示 `18`。

## 6. 執行 CsiNet mixed training

```python
!python CsiNet_train.py
```

`CsiNet_train.py` 預設：

```python
use_cost2100_multi_dataset = True
train_dataset_mode = 'mixed_cost2100'
evaluate_all_cost2100_datasets = True
epochs = 1500
encoded_dim = 512
```

## 7. 實驗結果

本實驗設定：CsiNet、COST2100 indoor、encoded dimension = 512、mixed six datasets、epochs = 1500、training samples = 18,000、validation samples = 3,600、test samples per dataset = 600。

| Dataset | NMSE (dB) | Test Samples |
|---|---:|---:|
| `user_center` | -1.228356 | 600 |
| `user_edge` | -1.248122 | 600 |
| `user_uniform` | -1.235212 | 600 |
| `user_left_cluster` | -1.243599 | 600 |
| `user_right_cluster` | -1.248775 | 600 |
| `user_ring` | -1.245125 | 600 |

平均 NMSE 約為 `-1.2415 dB`。`correlation` 為 `NaN` 是因為本次資料只輸出 CsiNet reconstruction 所需的 `HT`，沒有額外輸出 `HF_all`。本題主要評估指標為 CSI reconstruction NMSE。

## 8. 結論

六組不同使用者分佈的 COST2100 channel datasets 混合訓練後，CsiNet 在不同測試分佈上得到接近的 NMSE。這代表混合資料訓練能讓模型學到較多樣的通道特徵，降低對單一使用者分佈過度擬合的風險。

實際系統中，通道會隨使用者位置、移動方向、散射環境與場景改變。因此可以透過多場景資料混合訓練、資料增強、domain randomization，以及少量新環境資料 fine-tuning 來改善 CSI feedback 方法的泛化能力。
