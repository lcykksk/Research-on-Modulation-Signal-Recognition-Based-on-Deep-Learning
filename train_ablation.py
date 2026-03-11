import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pickle
import numpy as np
import tensorflow as tf
import mltools

from rmldataset2016 import load_data
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam

# ===== GPU config =====
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"成功识别 GPU: {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        print(f"GPU 配置错误: {e}")
else:
    print("未找到 GPU，使用 CPU 训练")
    print(f"TensorFlow 版本: {tf.__version__}")
    print(f"CUDA 可用: {tf.test.is_built_with_cuda()}")

# ===== 导入四个消融模型 =====
from HANet_ablation.HANet_ablate_only_IQ import HANet_only_IQ
from HANet_ablation.HANet_ablate_only_FFT import HANet_only_FFT
from HANet_ablation.HANet_ablate_without_attention import HANet_without_attention
from HANet_ablation.HANet_ablate_only_AP import HANet_only_AP


# ===== 模型字典 =====
models_dict = {
    "HANet_only_IQ": HANet_only_IQ,
    "HANet_only_AP": HANet_only_AP,
    "HANet_only_FFT": HANet_only_FFT,
    "HANet_without_attention": HANet_without_attention,
}

# ===== 读取数据索引 =====
idx = []
for idx_to_load in ["train_idx", "val_idx", "test_idx"]:
    with open(f"/root/autodl-tmp/RML2016.10a/{idx_to_load}.pkl", "rb") as f:
        idx.append(pickle.load(f))

(mods, snrs, lbl), (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), _ = load_data(idx=idx)

nb_epoch = 1000
batch_size = 512


def prepare_input_for_hanet_like_model(X_train, X_val, X_test):
    """
    HANet 及其消融模型统一输入:
    (N, 2, 128) -> (N, 2, 128, 1)
    """
    X_train_ = np.expand_dims(X_train, axis=3)
    X_val_ = np.expand_dims(X_val, axis=3)
    X_test_ = np.expand_dims(X_test, axis=3)
    return X_train_, X_val_, X_test_


def train_one_model(model_name, model_fn, X_train, Y_train, X_val, Y_val):
    print(f"\n========== 开始训练 {model_name} ==========")

    # 1) 预处理
    X_train_, X_val_, _ = prepare_input_for_hanet_like_model(X_train, X_val, X_test)

    print(f"{model_name} -> X_train shape: {X_train_.shape}")
    print(f"{model_name} -> X_val shape:   {X_val_.shape}")

    # 2) 建模
    model = model_fn()
    model.compile(
        optimizer=Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 3) 保存路径
    save_dir = os.path.join("results_ablate", model_name)
    os.makedirs(save_dir, exist_ok=True)

    # 4) 回调
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(save_dir, "weights.keras"),
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            mode="auto"
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            verbose=1,
            min_lr=1e-7
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=30,
            verbose=1,
            mode="auto",
            min_delta=1e-5
        )
    ]

    # 5) 训练
    history = model.fit(
        X_train_,
        Y_train,
        batch_size=batch_size,
        epochs=nb_epoch,
        verbose=1,
        validation_data=(X_val_, Y_val),
        callbacks=callbacks
    )

    # 6) 保存训练曲线文本
    mltools.show_history(history, save_dir)

    print(f"========== 完成训练 {model_name} ==========\n")


if __name__ == "__main__":
    for model_name, model_fn in models_dict.items():
        train_one_model(model_name, model_fn, X_train, Y_train, X_val, Y_val)
