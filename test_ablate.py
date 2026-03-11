import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from rmldataset2016 import load_data

# ===== 按你的实际模块路径修改 =====
from HANet.HANet import get_ap, get_fft

# 如果你的消融模型中也定义了自定义 Lambda / 函数，
# 需要把对应函数继续补到 custom_objects 里


def l2_normalize(x, axis=-1):
    y = np.sum(x ** 2, axis=axis, keepdims=True)
    return x / np.sqrt(y + 1e-12)


def to_amp_phase(X_test, nsamples=128):
    X_test_cmplx = X_test[:, 0, :] + 1j * X_test[:, 1, :]
    X_test_amp = np.abs(X_test_cmplx)
    X_test_ang = np.arctan2(X_test[:, 1, :], X_test[:, 0, :]) / np.pi
    X_test_amp = np.reshape(X_test_amp, (-1, 1, nsamples))
    X_test_ang = np.reshape(X_test_ang, (-1, 1, nsamples))
    X_test = np.concatenate((X_test_amp, X_test_ang), axis=1)
    return np.transpose(np.array(X_test), (0, 2, 1))


def norm_pad_zeros(X_train, nsamples):
    print("Pad:", X_train.shape)
    for i in range(X_train.shape[0]):
        norm_val = np.linalg.norm(X_train[i, :, 0], 2)
        if norm_val > 0:
            X_train[i, :, 0] = X_train[i, :, 0] / norm_val
    return X_train


def preprocess_data(X, model_name, nsamples=128):
    """
    根据训练时逻辑对测试数据做预处理
    这里默认 HANet 及其 4 个消融模型都使用 expand_dims(axis=3)
    如果你的某个消融模型输入不同，需要单独改这里
    """
    X_p = X.copy()

    if model_name in ['LSTM2']:
        X_p = to_amp_phase(X_p, nsamples)
        X_p = norm_pad_zeros(X_p[:, :128, :], 128)
    elif model_name in ['CLDNN']:
        X_p = np.reshape(X_p, (-1, 1, 2, 128))
    elif model_name in [
        'ResNet', 'HANet', 'CLDNN2', 'DenseNet', 'MCNET',
        'HANet_only_AP', 'HANet_only_FFT', 'HANet_only_IQ', 'HANet_without_attention'
    ]:
        X_p = np.expand_dims(X_p, axis=3)
    elif model_name in ['CNN1']:
        pass
    elif model_name in ['DAE']:
        X_p = to_amp_phase(X_p, 128)
        X_p[:, :, 0] = l2_normalize(X_p[:, :, 0])
        for i in range(X_p.shape[0]):
            phase_max = X_p[i, :, 1].max()
            phase_min = X_p[i, :, 1].min()
            if phase_max != phase_min:
                k = 2 / (phase_max - phase_min)
                X_p[i, :, 1] = -1 + k * (X_p[i, :, 1] - phase_min)

    return X_p


def safe_get_acc_list(snrs, acc_dict):
    vals = []
    for s in snrs:
        vals.append(acc_dict.get(str(s), np.nan))
    return vals


def main():
    # ===== 1. 读取测试集 =====
    with open('/root/autodl-tmp/RML2016.10a/test_idx.pkl', 'rb') as f:
        test_idx = pickle.load(f)

    (mods, snrs, lbl_all), _, _, (X_test, Y_test), _ = load_data(idx=[[], [], test_idx])
    lbl_test = [lbl_all[i] for i in test_idx]

    # ===== 2. 消融模型列表 =====
    model_list = [
        'HANet',
        'HANet_only_AP',
        'HANet_only_FFT',
        'HANet_only_IQ',
        'HANet_without_attention'
    ]

    # ===== 3. 路径设置 =====
    result_root = 'results_ablate'
    test_results_dir = 'result_test_ablate'
    os.makedirs(test_results_dir, exist_ok=True)

    all_snr_acc = {}

    # ===== 4. 逐模型测试 =====
    for model_name in model_list:
        print(f"\n正在测试模型: {model_name}")

        X_test_processed = preprocess_data(X_test, model_name)

        weight_path = os.path.join(result_root, model_name, 'weights.keras')
        if not os.path.exists(weight_path):
            print(f"警告: 未找到模型权重 {weight_path}，跳过...")
            continue

        try:
            model = load_model(
                weight_path,
                custom_objects={
                    'get_ap': get_ap,
                    'get_fft': get_fft,
                    'Lambda': tf.keras.layers.Lambda
                },
                safe_mode=False
            )
        except Exception as e:
            print(f"加载模型 {model_name} 失败: {e}")
            continue

        snr_acc = {}

        for snr in snrs:
            idx_snr = [i for i, l in enumerate(lbl_test) if l[1] == snr]
            if not idx_snr:
                continue

            X_snr = X_test_processed[idx_snr]
            Y_snr = Y_test[idx_snr]

            try:
                results = model.evaluate(X_snr, Y_snr, verbose=0)

                if isinstance(results, (list, tuple)):
                    if len(results) >= 2:
                        loss, acc = results[0], results[1]
                    else:
                        print(f"{model_name} 在 SNR={snr} 的 evaluate 返回异常: {results}")
                        continue
                else:
                    print(f"{model_name} 在 SNR={snr} 的 evaluate 返回非列表: {results}")
                    continue

                snr_acc[str(snr)] = float(acc)
                print(f"SNR {snr} dB: Accuracy = {acc:.4f}")

            except Exception as e:
                print(f"{model_name} 在 SNR={snr} 测试失败: {e}")

        all_snr_acc[model_name] = snr_acc

        # 保存每个模型的 SNR-Acc
        with open(os.path.join(test_results_dir, f'{model_name}_acc.pkl'), 'wb') as f:
            pickle.dump(snr_acc, f)

        # 同时保存 txt
        with open(os.path.join(test_results_dir, f'{model_name}_acc.txt'), 'w', encoding='utf-8') as f:
            for snr in snrs:
                if str(snr) in snr_acc:
                    f.write(f"{snr}\t{snr_acc[str(snr)]:.6f}\n")

    # ===== 5. 绘制总图 =====
    plt.figure(figsize=(12, 8))
    for name, acc_dict in all_snr_acc.items():
        y = safe_get_acc_list(snrs, acc_dict)
        plt.plot(snrs, y, marker='o', linewidth=2, label=name)

    plt.xlabel('SNR (dB)')
    plt.ylabel('Accuracy')
    plt.title('Ablation Model Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(test_results_dir, 'summary_accuracy_ablate.png'), dpi=300)
    plt.close()

    print(f"\n测试完成，结果已保存到: {test_results_dir}")


if __name__ == "__main__":
    main()
