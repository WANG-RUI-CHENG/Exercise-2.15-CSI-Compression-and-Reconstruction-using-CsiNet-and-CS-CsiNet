"""
CsiNet Implementation for CSI Compression and Reconstruction
This script builds and trains a residual-based autoencoder (CsiNet) for Channel State Information (CSI) compression
in both indoor and outdoor wireless environments. It supports different compression rates via adjustable encoded dimensions,
evaluates performance with NMSE (Normalized Mean Square Error) and correlation coefficient,
and saves training logs, model weights, and reconstruction visualizations.
"""
import tensorflow as tf
from keras.layers import Input, Dense, BatchNormalization, Reshape, Conv2D, add, LeakyReLU
from keras.models import Model
from keras.callbacks import TensorBoard, Callback
import scipy.io as sio 
import numpy as np
import math
import time
import os
import csv
tf.reset_default_graph()
# ────────────────────────  Environment Configuration  ───────────────────────── #
envir = 'indoor' #'indoor' or 'outdoor' -> Select wireless propagation environment
# ────────────────────────  CSI Image Parameters  ────────────────────────────── #
img_height = 32        # CSI matrix height (spatial dimension)
img_width = 32         # CSI matrix width (frequency dimension)
img_channels = 2       # Real and imaginary parts of CSI (2 channels)
img_total = img_height*img_width*img_channels  # Total CSI feature dimensions
# ────────────────────────  Network Hyperparameters  ─────────────────────────── #
residual_num = 2       # Number of residual blocks in the decoder
encoded_dim = 512      # Compress rate=1/4->dim.=512, 1/16->128, 1/32->64, 1/64->32

# ────────────────────────  Exercise 2.15 多資料集設定  ───────── #
# 設為 True 時，執行 Exercise 2.15，使用超過五組 COST 2100 通道資料集。
use_cost2100_multi_dataset = True
# 'mixed_cost2100'：對應題目 (c)，將所有 COST 2100 訓練/驗證資料集合併後訓練。
# 'single_cost2100'：只使用 cost2100_datasets 中第一組資料訓練，方便做消融測試或除錯。
train_dataset_mode = 'mixed_cost2100'
# 設為 True 時，訓練完成後會逐一測試每一組 COST 2100 測試資料集。
evaluate_all_cost2100_datasets = True
# 請將 COST 2100 產生的 MAT 檔放到 data/，並維持變數名稱為 HT 與 HF_all。
# HT 的形狀應為 [num_samples, 2048]；HF_all 可選，但若提供即可計算 correlation。
cost2100_datasets = [
    {
        'name': 'user_center',
        'train': 'data/DATA_Htrainin_user_center.mat',
        'val': 'data/DATA_Hvalin_user_center.mat',
        'test': 'data/DATA_Htestin_user_center.mat',
        'test_f': 'data/DATA_HtestFin_user_center_all.mat'
    },
    {
        'name': 'user_edge',
        'train': 'data/DATA_Htrainin_user_edge.mat',
        'val': 'data/DATA_Hvalin_user_edge.mat',
        'test': 'data/DATA_Htestin_user_edge.mat',
        'test_f': 'data/DATA_HtestFin_user_edge_all.mat'
    },
    {
        'name': 'user_uniform',
        'train': 'data/DATA_Htrainin_user_uniform.mat',
        'val': 'data/DATA_Hvalin_user_uniform.mat',
        'test': 'data/DATA_Htestin_user_uniform.mat',
        'test_f': 'data/DATA_HtestFin_user_uniform_all.mat'
    },
    {
        'name': 'user_left_cluster',
        'train': 'data/DATA_Htrainin_user_left_cluster.mat',
        'val': 'data/DATA_Hvalin_user_left_cluster.mat',
        'test': 'data/DATA_Htestin_user_left_cluster.mat',
        'test_f': 'data/DATA_HtestFin_user_left_cluster_all.mat'
    },
    {
        'name': 'user_right_cluster',
        'train': 'data/DATA_Htrainin_user_right_cluster.mat',
        'val': 'data/DATA_Hvalin_user_right_cluster.mat',
        'test': 'data/DATA_Htestin_user_right_cluster.mat',
        'test_f': 'data/DATA_HtestFin_user_right_cluster_all.mat'
    },
    {
        'name': 'user_ring',
        'train': 'data/DATA_Htrainin_user_ring.mat',
        'val': 'data/DATA_Hvalin_user_ring.mat',
        'test': 'data/DATA_Htestin_user_ring.mat',
        'test_f': 'data/DATA_HtestFin_user_ring_all.mat'
    }
]
# 若先執行 CsiNet_onlytest.py，此 CSV 會用來比較題目 (b) 與題目 (c) 的結果。
baseline_result_csv = 'result/ex215_eval_pretrained_CsiNet_' + envir + '_dim' + str(encoded_dim) + '.csv'
os.makedirs('result', exist_ok=True)
os.makedirs('saved_model', exist_ok=True)
# ────────────────────────  CsiNet Autoencoder Construction  ─────────────────── #
def residual_network(x, residual_num, encoded_dim):
    """
    Build the residual-based encoder-decoder network for CsiNet.
    Encoder: Conv2D -> Flatten -> Dense (compression to encoded_dim)
    Decoder: Dense -> Reshape -> Residual Blocks -> Conv2D (reconstruction to original CSI shape)
    Args:
        x (tensor): Input CSI tensor (shape: [batch, img_channels, img_height, img_width])
        residual_num (int): Number of residual blocks in the decoder
        encoded_dim (int): Dimension of the compressed CSI feature vector
    Returns:
        tensor: Reconstructed CSI tensor with sigmoid activation (range [0,1])
    """
    def add_common_layers(y):
        """Add BatchNormalization and LeakyReLU activation (shared in residual blocks)."""
        y = BatchNormalization()(y)
        y = LeakyReLU()(y)
        return y

    def residual_block_decoded(y):
        """Residual block for decoder: 3x3 Conv2D stack with shortcut connection."""
        shortcut = y  # Shortcut for residual connection
        y = Conv2D(8, kernel_size=(3, 3), padding='same', data_format='channels_first')(y)
        y = add_common_layers(y)
        
        y = Conv2D(16, kernel_size=(3, 3), padding='same', data_format='channels_first')(y)
        y = add_common_layers(y)
        
        y = Conv2D(2, kernel_size=(3, 3), padding='same', data_format='channels_first')(y)
        y = BatchNormalization()(y)
        y = add([shortcut, y])  # Residual connection: skip + conv output
        y = LeakyReLU()(y)
        return y
    
    # Encoder part: initial convolution + flatten + dense compression
    x = Conv2D(2, (3, 3), padding='same', data_format="channels_first")(x)
    x = add_common_layers(x)
    
    x = Reshape((img_total,))(x)  # Flatten CSI tensor to 1D vector
    encoded = Dense(encoded_dim, activation='linear')(x)  # Compress to encoded_dim
    
    # Decoder part: dense decompression + reshape + residual blocks + final convolution
    x = Dense(img_total, activation='linear')(encoded)  # Decompress to original flat dimension
    x = Reshape((img_channels, img_height, img_width,))(x)  # Reshape back to 4D CSI tensor
    for i in range(residual_num):
        x = residual_block_decoded(x)  # Stack residual blocks for reconstruction
    
    x = Conv2D(2, (3, 3), activation='sigmoid', padding='same', data_format="channels_first")(x)  # Output [0,1] for real/imag parts
    return x

# Build input tensor and full autoencoder model
image_tensor = Input(shape=(img_channels, img_height, img_width))  # Input shape for CSI data
network_output = residual_network(image_tensor, residual_num, encoded_dim)  # Reconstructed CSI
autoencoder = Model(inputs=[image_tensor], outputs=[network_output])  # Define full autoencoder
autoencoder.compile(optimizer='adam', loss='mse')  # Compile with Adam optimizer and MSE loss (CSI reconstruction)
print(autoencoder.summary())  # Print network architecture and parameter count
# ────────────────────────  Data Loading and Preprocessing  ──────────────────── #
# 以下輔助函式是為 Exercise 2.15 的多資料集訓練與評估所新增。
def require_file(path):
    """讀取資料前先確認必要的 MAT 檔是否存在。"""
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Missing file: %s\n"
            "Please generate this COST 2100 dataset first, put it under data/, "
            "or update cost2100_datasets to match your file name." % path
        )

def load_ht_data(path):
    """從 MAT 檔讀取正規化後的 CSI 資料 HT。"""
    require_file(path)
    mat = sio.loadmat(path)
    if 'HT' not in mat:
        raise KeyError("%s does not contain variable 'HT'." % path)
    return mat['HT'].astype('float32')

def load_hf_data(path):
    """若檔案存在，則從 MAT 檔讀取頻域 CSI 資料 HF_all。"""
    if path is None or not os.path.exists(path):
        return None
    mat = sio.loadmat(path)
    if 'HF_all' not in mat:
        return None
    return mat['HF_all']

def reshape_to_channels_first(x):
    """將展平的 CSI 樣本轉成原始 CsiNet 使用的 channels_first 格式。"""
    return np.reshape(x.astype('float32'), (len(x), img_channels, img_height, img_width))

def load_cost2100_tensor(dataset_spec, split_name):
    """讀取某一組 COST 2100 資料集的指定 split，並轉成 CsiNet 輸入格式。"""
    return reshape_to_channels_first(load_ht_data(dataset_spec[split_name]))

if use_cost2100_multi_dataset:
    # 當 train_dataset_mode 為 'mixed_cost2100' 時，讀取超過五組 COST 2100 資料並混合訓練。
    if train_dataset_mode == 'mixed_cost2100':
        x_train = np.concatenate([load_cost2100_tensor(ds, 'train') for ds in cost2100_datasets], axis=0)
        x_val = np.concatenate([load_cost2100_tensor(ds, 'val') for ds in cost2100_datasets], axis=0)
        x_test = load_cost2100_tensor(cost2100_datasets[0], 'test')
    elif train_dataset_mode == 'single_cost2100':
        x_train = load_cost2100_tensor(cost2100_datasets[0], 'train')
        x_val = load_cost2100_tensor(cost2100_datasets[0], 'val')
        x_test = load_cost2100_tensor(cost2100_datasets[0], 'test')
    else:
        raise ValueError("train_dataset_mode must be 'mixed_cost2100' or 'single_cost2100'.")

    print("Exercise 2.15 dataset mode:", train_dataset_mode)
    print("Training samples:", x_train.shape[0], "Validation samples:", x_val.shape[0])
else:
    # ────────────────────────  Data Loading and Preprocessing  ──────────────────── #
    # Load MATLAB-formatted CSI datasets (train/val/test) for selected environment
    if envir == 'indoor':
        mat = sio.loadmat('data/DATA_Htrainin.mat') 
        x_train = mat['HT'] # Training CSI data array
        mat = sio.loadmat('data/DATA_Hvalin.mat')
        x_val = mat['HT'] # Validation CSI data array
        mat = sio.loadmat('data/DATA_Htestin.mat')
        x_test = mat['HT'] # Test CSI data array
    elif envir == 'outdoor':
        mat = sio.loadmat('data/DATA_Htrainout.mat') 
        x_train = mat['HT'] # Training CSI data array
        mat = sio.loadmat('data/DATA_Hvalout.mat')
        x_val = mat['HT'] # Validation CSI data array
        mat = sio.loadmat('data/DATA_Htestout.mat')
        x_test = mat['HT'] # Test CSI data array

    # Convert data to float32 for neural network training
    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    x_test = x_test.astype('float32')

    # Reshape data to fit channels_first format: [batch, channels, height, width]
    x_train = np.reshape(x_train, (len(x_train), img_channels, img_height, img_width))
    x_val = np.reshape(x_val, (len(x_val), img_channels, img_height, img_width))
    x_test = np.reshape(x_test, (len(x_test), img_channels, img_height, img_width))

# ────────────────────────  Exercise 2.15 評估輔助函式  ─────────────────── #
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

# ────────────────────────  Custom Loss Callback  ─────────────────────────────── #
class LossHistory(Callback):
    """Custom Keras Callback to record batch-wise training loss and epoch-wise validation loss."""
    def on_train_begin(self, logs={}):
        """Initialize empty loss lists at training start."""
        self.losses_train = []
        self.losses_val = []

    def on_batch_end(self, batch, logs={}):
        """Append training loss of current batch to list."""
        self.losses_train.append(logs.get('loss'))
        
    def on_epoch_end(self, epoch, logs={}):
        """Append validation loss of current epoch to list."""
        self.losses_val.append(logs.get('val_loss'))
        
# Initialize loss history callback
history = LossHistory()
# ────────────────────────  Model Training  ──────────────────────────────────── #
# Generate unique file name with environment, encoded dimension and current date
file = 'CsiNet_'+(envir)+'_dim'+str(encoded_dim)+time.strftime('_%m_%d')
path = 'result/TensorBoard_%s' %file  # TensorBoard log directory

# Train the autoencoder with CSI data (input = target for reconstruction task)
autoencoder.fit(x_train, x_train,
                epochs=1000,               # Total training epochs
                batch_size=200,            # Mini-batch size
                shuffle=True,              # Shuffle training data per epoch
                validation_data=(x_val, x_val),  # Validation dataset
                callbacks=[history,        # Record loss history
                           TensorBoard(log_dir = path)])  # TensorBoard visualization

# Save training and validation loss to CSV files
filename = 'result/trainloss_%s.csv'%file
loss_history = np.array(history.losses_train)
np.savetxt(filename, loss_history, delimiter=",")

filename = 'result/valloss_%s.csv'%file
loss_history = np.array(history.losses_val)
np.savetxt(filename, loss_history, delimiter=",")
# ────────────────────────  Model Inference on Test Data  ────────────────────── #
if use_cost2100_multi_dataset and evaluate_all_cost2100_datasets:
    # 對應 Exercise 2.15(c)：將混合資料訓練後的 CsiNet 逐一測試在每組 COST 2100 資料集上。
    ex215_results = []
    x_hat = None
    rho = np.array([])
    for idx, ds in enumerate(cost2100_datasets):
        x_eval = load_cost2100_tensor(ds, 'test')
        X_eval = load_hf_data(ds.get('test_f'))
        result_row, x_hat_eval, rho_eval = evaluate_csi_reconstruction(
            autoencoder, x_eval, X_eval, dataset_name=ds['name']
        )
        ex215_results.append(result_row)
        if idx == 0:
            x_test = x_eval
            x_hat = x_hat_eval
            rho = rho_eval

    filename = 'result/ex215_eval_mixed_%s.csv' % file
    with open(filename, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['dataset', 'nmse_db', 'correlation', 'sec_per_sample', 'num_samples'])
        writer.writeheader()
        writer.writerows(ex215_results)
    print("Exercise 2.15 mixed-training results saved to", filename)

    if os.path.exists(baseline_result_csv):
        # 比較題目 (c) 的混合訓練結果與題目 (b) 的 pretrained / single-domain 測試結果。
        baseline_rows = {}
        with open(baseline_result_csv, 'r', newline='') as csv_file:
            for row in csv.DictReader(csv_file):
                baseline_rows[row['dataset']] = row
        compare_rows = []
        for row in ex215_results:
            base = baseline_rows.get(row['dataset'])
            if base is not None:
                base_nmse = float(base['nmse_db'])
                mixed_nmse = float(row['nmse_db'])
                compare_rows.append({
                    'dataset': row['dataset'],
                    'pretrained_nmse_db': base_nmse,
                    'mixed_nmse_db': mixed_nmse,
                    'mixed_minus_pretrained_db': mixed_nmse - base_nmse,
                    'pretrained_correlation': base.get('correlation', 'nan'),
                    'mixed_correlation': row['correlation']
                })
        if compare_rows:
            filename = 'result/ex215_compare_%s.csv' % file
            with open(filename, 'w', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=[
                    'dataset', 'pretrained_nmse_db', 'mixed_nmse_db', 'mixed_minus_pretrained_db',
                    'pretrained_correlation', 'mixed_correlation'
                ])
                writer.writeheader()
                writer.writerows(compare_rows)
            print("Exercise 2.15 comparison results saved to", filename)
else:
    # ────────────────────────  Model Inference on Test Data  ────────────────────── #
    # Measure inference time for CSI reconstruction
    tStart = time.time()
    x_hat = autoencoder.predict(x_test)  # Reconstruct CSI from test input
    tEnd = time.time()
    # Print average inference time per test sample
    print ("It cost %f sec" % ((tEnd - tStart)/x_test.shape[0]))
    # ────────────────────────  Performance Evaluation (NMSE & Correlation)  ──────────────────── #
    # Load original frequency-domain CSI for correlation calculation
    if envir == 'indoor':
        mat = sio.loadmat('data/DATA_HtestFin_all.mat')
        X_test = mat['HF_all']# Original frequency-domain CSI array
    elif envir == 'outdoor':
        mat = sio.loadmat('data/DATA_HtestFout_all.mat')
        X_test = mat['HF_all']# Original frequency-domain CSI array

    # Reshape frequency-domain CSI to [batch, img_height, 125]
    X_test = np.reshape(X_test, (len(X_test), img_height, 125))

    # Convert reconstructed/raw CSI from [0,1] to complex domain (-0.5~0.5 for real/imag)
    x_test_real = np.reshape(x_test[:, 0, :, :], (len(x_test), -1))
    x_test_imag = np.reshape(x_test[:, 1, :, :], (len(x_test), -1))
    x_test_C = x_test_real-0.5 + 1j*(x_test_imag-0.5)  # Raw complex CSI

    x_hat_real = np.reshape(x_hat[:, 0, :, :], (len(x_hat), -1))
    x_hat_imag = np.reshape(x_hat[:, 1, :, :], (len(x_hat), -1))
    x_hat_C = x_hat_real-0.5 + 1j*(x_hat_imag-0.5)  # Reconstructed complex CSI

    # Reshape complex CSI to frequency domain and perform FFT for correlation calculation
    x_hat_F = np.reshape(x_hat_C, (len(x_hat_C), img_height, img_width))
    # Zero-padding + FFT to match original frequency-domain CSI dimension
    X_hat = np.fft.fft(np.concatenate((x_hat_F, np.zeros((len(x_hat_C), img_height, 257-img_width))), axis=2), axis=2)
    X_hat = X_hat[:, :, 0:125]  # Truncate to original 125 frequency bins

    # Calculate correlation coefficient (rho) between original and reconstructed frequency-domain CSI
    n1 = np.sqrt(np.sum(np.conj(X_test)*X_test, axis=1))  # Norm of original CSI
    n1 = n1.astype('float64')
    n2 = np.sqrt(np.sum(np.conj(X_hat)*X_hat, axis=1))  # Norm of reconstructed CSI
    n2 = n2.astype('float64')
    aa = abs(np.sum(np.conj(X_test)*X_hat, axis=1))     # Cross term for correlation
    rho = np.mean(aa/(n1*n2), axis=1)                   # Correlation coefficient per sample

    # Reshape for NMSE calculation (Normalized Mean Square Error)
    X_hat = np.reshape(X_hat, (len(X_hat), -1))
    X_test = np.reshape(X_test, (len(X_test), -1))

    # Compute NMSE (in dB) for CSI reconstruction
    power = np.sum(abs(x_test_C)**2, axis=1)    # Power of original complex CSI
    power_d = np.sum(abs(X_hat)**2, axis=1)     # Power of reconstructed frequency-domain CSI
    mse = np.sum(abs(x_test_C-x_hat_C)**2, axis=1)  # MSE between original and reconstructed complex CSI
    # Print performance metrics for the selected environment and compression rate
    print("In "+envir+" environment")
    print("When dimension is", encoded_dim)
    print("NMSE is ", 10*math.log10(np.mean(mse/power)))  # NMSE in dB
    print("Correlation is ", np.mean(rho))                # Average correlation coefficient

    # Save reconstructed CSI and correlation coefficient to CSV files
    filename = "result/decoded_%s.csv"%file
    x_hat1 = np.reshape(x_hat, (len(x_hat), -1))
    np.savetxt(filename, x_hat1, delimiter=",")

    filename = "result/rho_%s.csv"%file
    np.savetxt(filename, rho, delimiter=",")
# ────────────────────────  CSI Reconstruction Visualization  ────────────────── #
import matplotlib.pyplot as plt
'''Plot absolute value of original and reconstructed complex CSI (first 10 samples)'''
n = min(10, len(x_test))
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original CSI absolute value
    ax = plt.subplot(2, n, i + 1 )
    x_testplo = abs(x_test[i, 0, :, :]-0.5 + 1j*(x_test[i, 1, :, :]-0.5))
    plt.imshow(np.max(np.max(x_testplo))-x_testplo.T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.invert_yaxis()
    # Display reconstructed CSI absolute value
    ax = plt.subplot(2, n, i + 1 + n)
    decoded_imgsplo = abs(x_hat[i, 0, :, :]-0.5 
                          + 1j*(x_hat[i, 1, :, :]-0.5))
    plt.imshow(np.max(np.max(decoded_imgsplo))-decoded_imgsplo.T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.invert_yaxis()
plt.show()
# ────────────────────────  Model Saving  ────────────────────────────────────── #
# Serialize model architecture to JSON file
model_json = autoencoder.to_json()
outfile = "result/model_%s.json"%file
with open(outfile, "w") as json_file:
    json_file.write(model_json)
# Serialize model weights to HDF5 file
outfile = "result/model_%s.h5"%file
autoencoder.save_weights(outfile)
