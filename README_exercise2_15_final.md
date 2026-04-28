# Exercise 2.15 — CsiNet Generalization on Multiple COST2100 Channel Datasets

本專案完成 *Wireless Communications and Machine Learning* Exercise 2.15。目標是測試 CsiNet 在不同 COST2100 通道資料集上的 CSI reconstruction NMSE，並比較單一分佈訓練與混合分佈訓練對模型泛化能力的影響。

---

## 1. Project Objective

Exercise 2.15 要求完成以下三件事：

1. 使用 COST2100 channel model 產生超過五組不同 channel datasets，例如改變 user distribution。
2. 使用 trained CsiNet 評估每一組 dataset 的 CSI reconstruction NMSE。
3. 將多組 channel datasets 混合後重新訓練 CsiNet，並與單一資料分佈訓練結果比較，觀察模型泛化能力。

本專案使用六種 user distribution：

| Dataset name | Description |
|---|---|
| `user_center` | Users are distributed near the center region. |
| `user_edge` | Users are distributed near the cell / room edge. |
| `user_uniform` | Users are uniformly distributed over the simulation area. |
| `user_left_cluster` | Users are clustered on the left side. |
| `user_right_cluster` | Users are clustered on the right side. |
| `user_ring` | Users are distributed around a ring-shaped region. |

---

## 2. Repository Structure

The GitHub repository should keep the original four Python files and one README file:

```text
Exercise_2.15/
├── CsiNet_train.py
├── CsiNet_onlytest.py
├── CS-CsiNet_train.py
├── CS-CsiNet_onlytest.py
├── README.md
├── result/
│   ├── final_mixed_1500epochs.csv
│   ├── final_single_1500epochs.csv          # optional, if baseline is executed
│   └── final_compare_1500epochs.csv         # optional, if baseline is executed
└── matlab/
    └── generate_cost2100_csinet_data.m       # recommended
```

Large generated `.mat` files are not uploaded to GitHub because they are dataset files and may be large. They should be generated locally and placed in `data/` before running the Python scripts.

Recommended local / Colab runtime folders:

```text
├── data/          # COST2100 .mat datasets, not uploaded to GitHub
├── result/        # CSV results, selected final CSV files may be uploaded
└── saved_model/   # trained models or pretrained models, optional
```

---

## 3. Required Code Changes

The original four Python files should keep their original names:

```text
CsiNet_train.py
CsiNet_onlytest.py
CS-CsiNet_train.py
CS-CsiNet_onlytest.py
```

For this exercise, the main modified file is:

```text
CsiNet_train.py
```

Required modifications:

1. Add Exercise 2.15 multi-dataset configuration.
2. Add six COST2100 dataset paths.
3. Load and concatenate multiple datasets for mixed training.
4. Evaluate the trained CsiNet model on all six test datasets.
5. Save NMSE results to CSV.
6. Support MATLAB `-v7.3` `.mat` files with `h5py`.
7. Use TensorFlow 2 / Colab-compatible imports.
8. Save Keras weights using `.weights.h5` filename format.

Recommended modification:

```text
CsiNet_onlytest.py
```

This file can also be updated to support TensorFlow 2 / Colab and MATLAB `-v7.3` files if pretrained model testing is required.

The two CS-CsiNet files can remain unchanged unless CS-CsiNet is also evaluated.

---

## 4. Environment Setup

### 4.1 MATLAB Environment for COST2100 Dataset Generation

MATLAB was used to generate COST2100 channel datasets.

Example setup:

```matlab
cd('C:\Users\user\Desktop\AIwireless')
addpath(genpath('cost2100-master'))
```

If `savepath` shows a permission warning, it can be ignored for the current MATLAB session. The path can be added again next time MATLAB is opened.

The generated `.mat` files are saved in:

```text
CsiNetData/
```

Each `.mat` file contains variable:

```text
HT
```

with shape:

```text
[num_samples, 2048]
```

because CsiNet uses:

```text
2 × 32 × 32 = 2048
```

The value range of `HT` is normalized to `[0, 1]`.

Example MATLAB check:

```matlab
S = load('CsiNetData/DATA_Htrainin_user_center.mat');
size(S.HT)
min(S.HT(:))
max(S.HT(:))
```

Expected output:

```text
3000   2048
0
1
```

---

## 5. Generated Dataset Files

The following 18 `.mat` files are required:

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

In this experiment:

| Split | Samples per dataset | Total samples |
|---|---:|---:|
| Train | 3000 | 18000 |
| Validation | 600 | 3600 |
| Test | 600 | 3600 |

---

## 6. Google Colab Setup

### 6.1 Enable GPU

In Colab:

```text
Runtime → Change runtime type → Hardware accelerator → T4 GPU → Save
```

Check GPU:

```python
!nvidia-smi
```

Expected GPU example:

```text
Tesla T4
```

---

### 6.2 Clone Repository

```python
!git clone https://github.com/WANG-RUI-CHENG/Exercise-2.15-CSI-Compression-and-Reconstruction-using-CsiNet-and-CS-CsiNet.git
%cd Exercise-2.15-CSI-Compression-and-Reconstruction-using-CsiNet-and-CS-CsiNet
!ls
```

Expected files:

```text
CS-CsiNet_onlytest.py
CS-CsiNet_train.py
CsiNet_onlytest.py
CsiNet_train.py
README.md
```

---

### 6.3 Install Python Packages

```python
!pip install scipy matplotlib pandas h5py
```

---

### 6.4 Create Runtime Folders

```python
!mkdir -p data result saved_model
```

---

### 6.5 Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

Upload `CsiNetData.zip` to Google Drive, then unzip it in Colab:

```python
!cp /content/drive/MyDrive/CsiNetData.zip .
!unzip -o CsiNetData.zip
!cp CsiNetData/*.mat data/
!ls data | wc -l
```

Expected number of `.mat` files:

```text
18
```

---

## 7. Training Configuration

The main configuration in `CsiNet_train.py` is:

```python
envir = 'indoor'
encoded_dim = 512
use_cost2100_multi_dataset = True
train_dataset_mode = 'mixed_cost2100'
evaluate_all_cost2100_datasets = True
```

For formal training, this experiment uses:

```python
epochs = 1500
batch_size = 200
```

---

## 8. Run Training

Run:

```python
!python CsiNet_train.py
```

The script will:

1. Load all six COST2100 datasets.
2. Mix all training datasets.
3. Train CsiNet using the mixed dataset.
4. Evaluate NMSE on each individual test dataset.
5. Save results to CSV.

---

## 9. Experimental Results

### 9.1 Mixed-Dataset Training Result

CsiNet was trained using the mixed COST2100 datasets for 1500 epochs. The encoded dimension was 512.

| Dataset | NMSE (dB) | Correlation | Test Samples |
|---|---:|---:|---:|
| `user_center` | -1.2284 | NaN | 600 |
| `user_edge` | -1.2481 | NaN | 600 |
| `user_uniform` | -1.2352 | NaN | 600 |
| `user_left_cluster` | -1.2436 | NaN | 600 |
| `user_right_cluster` | -1.2488 | NaN | 600 |
| `user_ring` | -1.2451 | NaN | 600 |

Average NMSE:

```text
-1.2415 dB
```

The correlation coefficient is shown as `NaN` because only `HT` was generated in this experiment. `HF_all` was not generated, so frequency-domain correlation was not evaluated. The main metric for this exercise is NMSE.

---

## 10. How to Display Result CSV in Colab

```python
import pandas as pd
import glob

for f in glob.glob("result/ex215*.csv"):
    print(f)
    display(pd.read_csv(f))
```

If final results were copied to separate files:

```python
mixed = pd.read_csv("result/final_mixed_1500epochs.csv")
display(mixed)
```

---

## 11. Optional Baseline Comparison

To compare mixed training with single-distribution training, change:

```python
train_dataset_mode = 'mixed_cost2100'
```

to:

```python
train_dataset_mode = 'single_cost2100'
```

Then rerun:

```python
!python CsiNet_train.py
```

Save the baseline result as:

```text
result/final_single_1500epochs.csv
```

Then compare:

```python
import pandas as pd

mixed = pd.read_csv("result/final_mixed_1500epochs.csv")
single = pd.read_csv("result/final_single_1500epochs.csv")

compare = pd.DataFrame({
    "dataset": mixed["dataset"],
    "single_nmse_db": single["nmse_db"],
    "mixed_nmse_db": mixed["nmse_db"],
    "mixed_minus_single_db": mixed["nmse_db"] - single["nmse_db"]
})

display(compare)
compare.to_csv("result/final_compare_1500epochs.csv", index=False)
```

Because NMSE is better when it is more negative:

```text
mixed_minus_single_db < 0
```

means mixed training improves reconstruction performance compared with single-distribution training.

---

## 12. Discussion

The 1500-epoch mixed-dataset result gives stable NMSE across all six user distributions. This shows that training CsiNet with diverse COST2100 channel datasets can improve generalization to different user distributions. In practical CSI feedback systems, the channel distribution may change due to user mobility, environment changes, and different propagation conditions. Therefore, using mixed or more diverse channel datasets during training is helpful for improving robustness.

Possible improvements include:

1. Generate more COST2100 scenarios with different environments and mobility patterns.
2. Include both indoor and outdoor channel datasets.
3. Use data augmentation on CSI matrices.
4. Train with multiple compression rates.
5. Add `HF_all` generation to evaluate both NMSE and correlation coefficient.
6. Use domain adaptation or fine-tuning for unseen channel distributions.

---

## 13. Notes

- Large `.mat` datasets are not uploaded to GitHub.
- The `data/` folder should be created locally or in Colab before running the scripts.
- The `result/` folder may contain selected final CSV files for reporting.
- The `saved_model/` folder is optional and can be used to store trained model weights.
- For TensorFlow / Keras compatibility in Colab, model weights should be saved with filename ending in `.weights.h5`.
