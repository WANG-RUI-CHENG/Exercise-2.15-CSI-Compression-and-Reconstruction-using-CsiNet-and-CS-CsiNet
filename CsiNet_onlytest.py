"""
CsiNet Inference Only for CSI Compression and Reconstruction
This script loads a pre-trained CsiNet autoencoder model (architecture + weights)
to perform only inference on test CSI data for indoor/outdoor wireless environments.
It evaluates model performance with NMSE (Normalized Mean Square Error) and correlation coefficient,
measures per-sample inference time, and visualizes the original vs. reconstructed CSI amplitude.
No model training is performed in this script.
"""
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Reshape, Conv2D, add, LeakyReLU
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.callbacks import TensorBoard, Callback
import scipy.io as sio 
import numpy as np
import math
import time
import os
import csv
# tf.reset_default_graph()  # TensorFlow 2 / Colab 不需要使用
# ────────────────────────  Environment & CSI Configuration  ───────────────────────── #
envir = 'indoor' #'indoor' or 'outdoor' -> Select the wireless propagation environment
# ────────────────────────  CSI Image Parameters  ────────────────────────────── #
img_height = 32        # CSI matrix height (spatial dimension)
img_width = 32         # CSI matrix width (frequency dimension)
img_channels = 2       # Real and imaginary parts of CSI (2 separate channels)
img_total = img_height*img_width*img_channels  # Total flattened CSI feature dimensions
# ────────────────────────  Network Hyperparameters  ─────────────────────────── #
residual_num = 2       # Number of residual blocks in the decoder (match pre-trained model)
encoded_dim = 512      # Compression dimension (match pre-trained model): 1/4→512,1/16→128,1/32→64,1/64→32

# ────────────────────────  Exercise 2.15 多資料集設定  ───────── #
# 設為 True 時，對應題目 (b)，使用同一個已訓練 CsiNet 模型測試超過五組 COST 2100 資料集。
use_cost2100_multi_dataset = True
# 請將 COST 2100 產生的 MAT 檔放到 data/，並維持變數名稱為 HT 與 HF_all。
# HT 的形狀應為 [num_samples, 2048]；HF_all 可選，但若提供即可計算 correlation。
cost2100_datasets = [
    {
        'name': 'user_center',
        'test': 'data/DATA_Htestin_user_center.mat',
        'test_f': 'data/DATA_HtestFin_user_center_all.mat'
    },
    {
        'name': 'user_edge',
        'test': 'data/DATA_Htestin_user_edge.mat',
        'test_f': 'data/DATA_HtestFin_user_edge_all.mat'
    },
    {
        'name': 'user_uniform',
        'test': 'data/DATA_Htestin_user_uniform.mat',
        'test_f': 'data/DATA_HtestFin_user_uniform_all.mat'
    },
    {
        'name': 'user_left_cluster',
        'test': 'data/DATA_Htestin_user_left_cluster.mat',
        'test_f': 'data/DATA_HtestFin_user_left_cluster_all.mat'
    },
    {
        'name': 'user_right_cluster',
        'test': 'data/DATA_Htestin_user_right_cluster.mat',
        'test_f': 'data/DATA_HtestFin_user_right_cluster_all.mat'
    },
    {
        'name': 'user_ring',
        'test': 'data/DATA_Htestin_user_ring.mat',
        'test_f': 'data/DATA_HtestFin_user_ring_all.mat'
    }
]
os.makedirs('result', exist_ok=True)
# Generate model file name (consistent with training script naming convention)
file = 'CsiNet_'+(envir)+'_dim'+str(encoded_dim)
# ────────────────────────  Load Pre-trained CsiNet Model  ────────────────────── #
# Load model architecture from JSON file
outfile = "saved_model/model_%s.json"%file
json_file = open(outfile, 'r')
loaded_model_json = json_file.read()
json_file.close()
autoencoder = model_from_json(loaded_model_json)  # Reconstruct model from JSON

# Load pre-trained model weights from HDF5 file
outfile = "saved_model/model_%s.h5"%file
if not os.path.exists(outfile):
    outfile = "saved_model/model_%s.weights.h5"%file
autoencoder.load_weights(outfile)  # Load weights into the reconstructed model
# ────────────────────────  Test Data Loading and Preprocessing  ─────────────── #
# 以下輔助函式是為 Exercise 2.15 的多資料集評估所新增。
def require_file(path):
    """讀取資料前先確認必要的 MAT 檔是否存在。"""
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Missing file: %s\n"
            "Please generate this COST 2100 dataset first, put it under data/, "
            "or update cost2100_datasets to match your file name." % path
        )

def load_ht_data(path):
    """從 MAT 檔讀取正規化後的 CSI 資料 HT；支援一般 MAT 檔與 MATLAB v7.3 HDF5 MAT 檔。"""
    require_file(path)
    try:
        mat = sio.loadmat(path)
        if 'HT' not in mat:
            raise KeyError("%s does not contain variable 'HT'." % path)
        return mat['HT'].astype('float32')
    except NotImplementedError:
        import h5py
        with h5py.File(path, 'r') as f:
            if 'HT' not in f:
                raise KeyError("%s does not contain variable 'HT'." % path)
            HT = np.array(f['HT'])
            # MATLAB v7.3 以 HDF5 儲存時，Python 讀入後常會轉置；這裡自動修正成 [N, 2048]。
            if HT.ndim == 2 and HT.shape[0] == img_total and HT.shape[1] != img_total:
                HT = HT.T
            return HT.astype('float32')

def load_hf_data(path):
    """若檔案存在，則讀取頻域 CSI 資料 HF_all；支援一般 MAT 檔與 MATLAB v7.3 HDF5 MAT 檔。"""
    if path is None or not os.path.exists(path):
        return None
    try:
        mat = sio.loadmat(path)
        if 'HF_all' not in mat:
            return None
        return mat['HF_all']
    except NotImplementedError:
        import h5py
        with h5py.File(path, 'r') as f:
            if 'HF_all' not in f:
                return None
            HF_all = np.array(f['HF_all'])
            return HF_all

def reshape_to_channels_first(x):
    """將展平的 CSI 樣本轉成原始 CsiNet 使用的 channels_first 格式。"""
    return np.reshape(x.astype('float32'), (len(x), img_channels, img_height, img_width))

def load_cost2100_tensor(dataset_spec):
    """讀取一組 COST 2100 測試資料集，並轉成 CsiNet 輸入格式。"""
    return reshape_to_channels_first(load_ht_data(dataset_spec['test']))

def evaluate_csi_reconstruction(model, x_eval, X_eval=None, dataset_name='dataset'):
    """在單一 CSI 資料集上計算 NMSE，並在有 HF_all 時額外計算 correlation coefficient。"""
    tStart = time.time()
    x_hat_eval = model.predict(x_eval)
    tEnd = time.time()

    x_eval_real = np.reshape(x_eval[:, 0, :, :], (len(x_eval), -1))
    x_eval_imag = np.reshape(x_eval[:, 1, :, :], (len(x_eval), -1))
    x_eval_C = x_eval_real-0.5 + 1j*(x_eval_imag-0.5)

    x_hat_real = np.reshape(x_hat_eval[:, 0, :, :], (len(x_hat_eval), -1))
    x_hat_imag = np.reshape(x_hat_eval[:, 1, :, :], (len(x_hat_eval), -1))
    x_hat_C = x_hat_real-0.5 + 1j*(x_hat_imag-0.5)

    power = np.sum(abs(x_eval_C)**2, axis=1)
    mse = np.sum(abs(x_eval_C-x_hat_C)**2, axis=1)
    nmse_db = 10*math.log10(np.mean(mse/np.maximum(power, 1e-12)))

    rho_mean = np.nan
    rho = np.array([])
    if X_eval is not None:
        X_eval = np.reshape(X_eval, (len(X_eval), img_height, 125))
        x_hat_F = np.reshape(x_hat_C, (len(x_hat_C), img_height, img_width))
        X_hat_eval = np.fft.fft(
            np.concatenate((x_hat_F, np.zeros((len(x_hat_C), img_height, 257-img_width))), axis=2),
            axis=2
        )
        X_hat_eval = X_hat_eval[:, :, 0:125]
        n1 = np.sqrt(np.sum(np.conj(X_eval)*X_eval, axis=1)).astype('float64')
        n2 = np.sqrt(np.sum(np.conj(X_hat_eval)*X_hat_eval, axis=1)).astype('float64')
        aa = abs(np.sum(np.conj(X_eval)*X_hat_eval, axis=1))
        rho = np.mean(aa/np.maximum(n1*n2, 1e-12), axis=1)
        rho_mean = float(np.mean(rho))

    print("In " + dataset_name + " dataset")
    print("When dimension is", encoded_dim)
    print("NMSE is ", nmse_db)
    print("Correlation is ", rho_mean)
    print("It cost %f sec" % ((tEnd - tStart)/x_eval.shape[0]))

    return {
        'dataset': dataset_name,
        'nmse_db': nmse_db,
        'correlation': rho_mean,
        'sec_per_sample': (tEnd - tStart)/x_eval.shape[0],
        'num_samples': x_eval.shape[0]
    }, x_hat_eval, rho

if use_cost2100_multi_dataset:
    # 對應 Exercise 2.15(b)：將已訓練的 CsiNet 逐一測試在每組 COST 2100 資料集上。
    ex215_results = []
    x_hat = None
    rho = np.array([])
    for idx, ds in enumerate(cost2100_datasets):
        x_eval = load_cost2100_tensor(ds)
        X_eval = load_hf_data(ds.get('test_f'))
        result_row, x_hat_eval, rho_eval = evaluate_csi_reconstruction(
            autoencoder, x_eval, X_eval, dataset_name=ds['name']
        )
        ex215_results.append(result_row)
        if idx == 0:
            x_test = x_eval
            x_hat = x_hat_eval
            rho = rho_eval

    filename = 'result/ex215_eval_pretrained_%s.csv' % file
    with open(filename, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['dataset', 'nmse_db', 'correlation', 'sec_per_sample', 'num_samples'])
        writer.writeheader()
        writer.writerows(ex215_results)
    print("Exercise 2.15 pretrained-model results saved to", filename)
else:
    # ────────────────────────  Test Data Loading and Preprocessing  ─────────────── #
    # Load MATLAB-formatted test CSI data for the selected environment
    if envir == 'indoor':
        mat = sio.loadmat('data/DATA_Htestin.mat')
        x_test = mat['HT'] # Indoor test CSI data array
    elif envir == 'outdoor':
        mat = sio.loadmat('data/DATA_Htestout.mat')
        x_test = mat['HT'] # Outdoor test CSI data array

    # Convert data to float32 (matching training data type for inference)
    x_test = x_test.astype('float32')
    # Reshape to channels_first format: [batch_size, channels, height, width] (match model input)
    x_test = np.reshape(x_test, (len(x_test), img_channels, img_height, img_width))
    # ────────────────────────  CSI Reconstruction Inference  ────────────────────── #
    # Measure total inference time for the entire test set
    tStart = time.time()
    x_hat = autoencoder.predict(x_test)  # Reconstruct CSI from test input (forward pass only)
    tEnd = time.time()
    # Calculate and print average inference time per test sample
    print ("It cost %f sec" % ((tEnd - tStart)/x_test.shape[0]))
    # ────────────────────────  Performance Evaluation (NMSE & Correlation)  ─────── #
    # Load original frequency-domain CSI data for correlation coefficient calculation
    if envir == 'indoor':
        mat = sio.loadmat('data/DATA_HtestFin_all.mat')
        X_test = mat['HF_all']# Indoor frequency-domain CSI array
    elif envir == 'outdoor':
        mat = sio.loadmat('data/DATA_HtestFout_all.mat')
        X_test = mat['HF_all']# Outdoor frequency-domain CSI array

    # Reshape frequency-domain CSI to [batch_size, img_height, 125] (original frequency bins)
    X_test = np.reshape(X_test, (len(X_test), img_height, 125))

    # Convert original CSI from [0,1] tensor to complex domain (-0.5~0.5 for real/imag parts)
    x_test_real = np.reshape(x_test[:, 0, :, :], (len(x_test), -1))
    x_test_imag = np.reshape(x_test[:, 1, :, :], (len(x_test), -1))
    x_test_C = x_test_real-0.5 + 1j*(x_test_imag-0.5)

    # Convert reconstructed CSI from [0,1] tensor to complex domain (same as original)
    x_hat_real = np.reshape(x_hat[:, 0, :, :], (len(x_hat), -1))
    x_hat_imag = np.reshape(x_hat[:, 1, :, :], (len(x_hat), -1))
    x_hat_C = x_hat_real-0.5 + 1j*(x_hat_imag-0.5)

    # Reshape complex CSI to frequency domain and perform FFT for correlation calculation
    x_hat_F = np.reshape(x_hat_C, (len(x_hat_C), img_height, img_width))
    # Zero-padding to 257 frequency bins + FFT + truncate to original 125 bins (matching X_test)
    X_hat = np.fft.fft(np.concatenate((x_hat_F, np.zeros((len(x_hat_C), img_height, 257-img_width))), axis=2), axis=2)
    X_hat = X_hat[:, :, 0:125]

    # Calculate correlation coefficient (rho) between original and reconstructed frequency-domain CSI
    n1 = np.sqrt(np.sum(np.conj(X_test)*X_test, axis=1))  # L2 norm of original frequency-domain CSI
    n1 = n1.astype('float64')
    n2 = np.sqrt(np.sum(np.conj(X_hat)*X_hat, axis=1))  # L2 norm of reconstructed frequency-domain CSI
    n2 = n2.astype('float64')
    aa = abs(np.sum(np.conj(X_test)*X_hat, axis=1))     # Cross inner product for correlation
    rho = np.mean(aa/(n1*n2), axis=1)                   # Per-sample correlation coefficient

    # Reshape for NMSE (Normalized Mean Square Error) calculation (flatten to 1D)
    X_hat = np.reshape(X_hat, (len(X_hat), -1))
    X_test = np.reshape(X_test, (len(X_test), -1))

    # Compute NMSE (in dB) for time-domain CSI reconstruction
    power = np.sum(abs(x_test_C)**2, axis=1)    # Power of original complex CSI
    mse = np.sum(abs(x_test_C-x_hat_C)**2, axis=1)  # MSE between original and reconstructed complex CSI
    # Print key performance metrics for the selected environment and compression rate
    print("In "+envir+" environment")
    print("When dimension is", encoded_dim)
    print("NMSE is ", 10*math.log10(np.mean(mse/power)))  # NMSE in logarithmic dB scale
    print("Correlation is ", np.mean(rho))                # Average correlation coefficient over test set
# ────────────────────────  CSI Reconstruction Visualization  ────────────────── #
import matplotlib.pyplot as plt
'''繪製前 10 筆測試樣本的原始與重建 complex CSI 絕對振幅'''
n = min(10, len(x_test))  # 要視覺化的樣本數，避免測試資料少於 10 筆時出錯
plt.figure(figsize=(20, 4))
for i in range(n):
    # 顯示原始 CSI 的絕對振幅
    ax = plt.subplot(2, n, i + 1 )
    x_testplo = abs(x_test[i, 0, :, :]-0.5 + 1j*(x_test[i, 1, :, :]-0.5))
    plt.imshow(np.max(np.max(x_testplo))-x_testplo.T)  # 反轉顯示，使圖像較容易觀察
    plt.gray()  # 使用灰階 colormap
    ax.get_xaxis().set_visible(False)  # 隱藏 x 軸
    ax.get_yaxis().set_visible(False)  # 隱藏 y 軸
    ax.invert_yaxis()  # 反轉 y 軸，使空間方向一致

    # 顯示重建 CSI 的絕對振幅
    ax = plt.subplot(2, n, i + 1 + n)
    decoded_imgsplo = abs(x_hat[i, 0, :, :]-0.5 
                          + 1j*(x_hat[i, 1, :, :]-0.5))
    plt.imshow(np.max(np.max(decoded_imgsplo))-decoded_imgsplo.T)  # 反轉顯示，使圖像較容易觀察
    plt.gray()  # 使用灰階 colormap
    ax.get_xaxis().set_visible(False)  # 隱藏 x 軸
    ax.get_yaxis().set_visible(False)  # 隱藏 y 軸
    ax.invert_yaxis()  # 反轉 y 軸，使空間方向一致
plt.show()  # 顯示視覺化結果