import os
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Conv2D, Add, Concatenate, Lambda, Reshape,
    SeparableConv2D, BatchNormalization, Activation, Bidirectional, LSTM,
    LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
)
from tensorflow.keras.models import Model

# -------------------------
# Multi-modal transform
# -------------------------
@tf.keras.utils.register_keras_serializable(package="Custom", name="get_ap")
def get_ap(x):
    """
    x: (B, 2, 128, 1)
    return: (B, 2, 128, 1) -> [Amplitude, Phase]
    """
    i = x[:, 0, :, 0]
    q = x[:, 1, :, 0]
    amp = tf.sqrt(tf.square(i) + tf.square(q) + 1e-8)
    phase = tf.math.atan2(q, i + 1e-8)
    return tf.expand_dims(tf.stack([amp, phase], axis=1), axis=-1)

@tf.keras.utils.register_keras_serializable(package="Custom", name="get_fft")
def get_fft(x):
    """
    x: (B, 2, 128, 1)
    return: (B, 2, 128, 1) -> [FFT real, FFT imag]
    """
    i = tf.cast(x[:, 0, :, 0], tf.complex64)
    q = tf.cast(x[:, 1, :, 0], tf.complex64)
    complex_signal = tf.complex(tf.math.real(i), tf.math.real(q))
    fft_signal = tf.signal.fft(complex_signal)
    fft_real = tf.math.real(fft_signal)
    fft_imag = tf.math.imag(fft_signal)
    return tf.expand_dims(tf.stack([fft_real, fft_imag], axis=1), axis=-1)

# -------------------------
# Multi-channel feature extractor
# -------------------------
def feature_module(input_tensor, prefix=""):
    """
    input_tensor: (B, 2, 128, 1)
    """
    a = Lambda(lambda x: x[:, 0:1, :, :], output_shape=(1, 128, 1), name=f"{prefix}_A_slice")(input_tensor)
    b = Lambda(lambda x: x[:, 1:2, :, :], output_shape=(1, 128, 1), name=f"{prefix}_B_slice")(input_tensor)

    a_conv = Conv2D(8, (1, 50), padding="same", activation="relu", name=f"{prefix}_A_conv")(a)
    b_conv = Conv2D(8, (1, 50), padding="same", activation="relu", name=f"{prefix}_B_conv")(b)

    ab_concat = Concatenate(axis=1, name=f"{prefix}_AB_concat")([a_conv, b_conv])
    ab_fuse = Conv2D(16, (1, 50), padding="same", activation="relu", name=f"{prefix}_IQ_fuse_conv")(ab_concat)

    joint_conv = Conv2D(16, (2, 50), padding="same", activation="relu", name=f"{prefix}_IQ_joint_conv")(input_tensor)

    fusion = Add(name=f"{prefix}_fusion_add")([ab_fuse, joint_conv])
    out = Conv2D(16, (2, 50), padding="same", activation="relu", name=f"{prefix}_final_conv")(fusion)
    return out

# -------------------------
# Depthwise separable residual block
# -------------------------
def dsc_residual_module(x_input, filters=50, prefix="res"):
    """
    Closer to the paper:
    2D Conv -> depthwise separable conv -> residual fusion
    """
    shortcut = Conv2D(filters, (1, 1), padding="same", name=f"{prefix}_shortcut")(x_input)

    x = Conv2D(filters, (2, 8), padding="same", name=f"{prefix}_conv_layer1")(x_input)
    x = BatchNormalization(name=f"{prefix}_bn1")(x)
    x = Activation("relu", name=f"{prefix}_relu1")(x)

    x = SeparableConv2D(filters, (2, 8), padding="same", name=f"{prefix}_dsc")(x)
    x = BatchNormalization(name=f"{prefix}_bn2")(x)
    x = Activation("relu", name=f"{prefix}_relu2")(x)

    x = Conv2D(filters, (1, 1), padding="same", name=f"{prefix}_pointwise")(x)
    x = BatchNormalization(name=f"{prefix}_bn3")(x)

    x = Add(name=f"{prefix}_residual_add")([x, shortcut])
    x = Activation("relu", name=f"{prefix}_residual_relu")(x)
    return x

# -------------------------
# BiLSTM + Self-Attention + Multi-Head Attention
# -------------------------
def temporal_attention_module(x, lstm_units=128, num_heads=4, prefix="final"):
    """
    Input x: (B, 2, 128, C)
    Flow:
    reshape -> BiLSTM -> Self-Attention -> Multi-Head Attention -> pooling
    """
    seq_len = x.shape[2]
    channels = x.shape[1] * x.shape[3]
    if seq_len is None or channels is None:
        raise ValueError("Static shape inference failed before temporal module.")

    x = Reshape((seq_len, channels), name=f"{prefix}_reshape")(x)  # (B, 128, 2*C)

    x = Bidirectional(
        LSTM(lstm_units, return_sequences=True),
        name=f"{prefix}_bilstm"
    )(x)  # (B, 128, 256)

    # Self-attention
    self_attn = MultiHeadAttention(
        num_heads=1,
        key_dim=lstm_units,
        dropout=0.1,
        name=f"{prefix}_self_attention"
    )(x, x)
    x = Add(name=f"{prefix}_self_attn_add")([x, self_attn])
    x = LayerNormalization(name=f"{prefix}_self_attn_norm")(x)

    # Multi-head attention
    mha_out = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=lstm_units // num_heads,
        dropout=0.1,
        name=f"{prefix}_multihead_attention"
    )(x, x)
    x = Add(name=f"{prefix}_mha_add")([x, mha_out])
    x = LayerNormalization(name=f"{prefix}_mha_norm")(x)

    x = GlobalAveragePooling1D(name=f"{prefix}_gap")(x)
    return x

# -------------------------
# HANet
# -------------------------
def HANet_only_IQ(input_shape=(2, 128), classes=11, dropout_rate=0.5, **kwargs):
    input_layer = Input(shape=(input_shape[0], input_shape[1], 1), name="input")

    # ap_tensor = Lambda(get_ap, output_shape=(2, 128, 1), name="get_ap_lambda")(input_layer)
    # fft_tensor = Lambda(get_fft, output_shape=(2, 128, 1), name="get_fft_lambda")(input_layer)

    iq_output = feature_module(input_layer, prefix="IQ")
    # ap_output = feature_module(ap_tensor, prefix="AP")
    # fft_output = feature_module(fft_tensor, prefix="FFT")

    # fusion = Add(name="all_fusion_add")([iq_output, ap_output, fft_output])
    fusion = Conv2D(50, (2, 50), padding="same", activation="relu", name="fusion_conv")(iq_output)

    spatial_output = dsc_residual_module(fusion, filters=50, prefix="global_dsc_res")
    temporal_output = temporal_attention_module(
        spatial_output,
        lstm_units=128,
        num_heads=4,
        prefix="final_temporal"
    )

    x = Dense(128, activation="relu", name="classifier_dense")(temporal_output)
    x = Dropout(dropout_rate, name="classifier_dropout")(x)
    out = Dense(classes, activation="softmax", name="classifier_softmax")(x)

    return Model(inputs=input_layer, outputs=out, name="HANet")

