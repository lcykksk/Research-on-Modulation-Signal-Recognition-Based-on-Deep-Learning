import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, Add, Concatenate, Lambda, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.layers import SeparableConv2D, BatchNormalization, Activation, Bidirectional, LSTM, Permute, Multiply, RepeatVector
import tensorflow.keras.backend as K

# 1. 显式注册序列化
@tf.keras.utils.register_keras_serializable(package="Custom", name="get_ap")
def get_ap(x):
    I = x[:, 0, :, 0]
    Q = x[:, 1, :, 0]
    amp = tf.sqrt(tf.square(I) + tf.square(Q))
    phase = tf.math.atan2(Q, I + 1e-8)
    # 调整为 (Batch, 2, 128, 1)
    return tf.expand_dims(tf.stack([amp, phase], axis=1), axis=-1)

@tf.keras.utils.register_keras_serializable(package="Custom", name="get_fft")
def get_fft(x):
    I = tf.cast(x[:, 0, :, 0], tf.complex64)
    Q = tf.cast(x[:, 1, :, 0], tf.complex64)
    fft_signal = tf.signal.fft(I + 1j * Q)
    out = tf.stack([tf.math.real(fft_signal), tf.math.imag(fft_signal)], axis=1)
    return tf.expand_dims(out, axis=-1)

def feature_module(input_tensor, prefix=""):
    # 修正维度切片：确保后续卷积能处理 (Batch, 1, 128, 1)
    A = Lambda(lambda x: x[:, 0:1, :, :], output_shape=(1, 128, 1), name=f'{prefix}_A_slice')(input_tensor)
    B = Lambda(lambda x: x[:, 1:2, :, :], output_shape=(1, 128, 1), name=f'{prefix}_B_slice')(input_tensor)

    A_conv = Conv2D(8, (1, 50), padding='same', activation='relu', name=f'{prefix}_A_conv')(A)
    B_conv = Conv2D(8, (1, 50), padding='same', activation='relu', name=f'{prefix}_B_conv')(B)

    AB_concat = Concatenate(axis=1, name=f'{prefix}_AB_concat')([A_conv, B_conv])
    AB_fuse = Conv2D(16, (1, 50), padding='same', activation='relu', name=f'{prefix}_IQ_fuse_conv')(AB_concat)
    AB_conv = Conv2D(16, (2, 50), padding='same', activation='relu', name=f'{prefix}_IQ_joint_conv')(input_tensor)

    fusion = Add(name=f'{prefix}_fusion_add')([AB_fuse, AB_conv])
    return Conv2D(16, (2, 50), padding='same', activation='relu', name=f'{prefix}_final_conv')(fusion)

def dsc_residual_module(x_input, prefix="res"):
    x1 = Conv2D(50, (2, 8), padding='same', name=f'{prefix}_conv_layer1')(x_input)
    x1 = BatchNormalization()(x1); x1 = Activation('relu')(x1)

    x = SeparableConv2D(50, (2, 8), padding='same', name=f'{prefix}_dsc')(x1)
    x = BatchNormalization()(x); x = Activation('relu')(x)

    x = Conv2D(50, (2, 8), padding='same', name=f'{prefix}_conv_before_add')(x)
    x = BatchNormalization()(x); x = Add()([x, x1]) # 残差连接
    
    out = Conv2D(50, (2, 8), padding='same', name=f'{prefix}_final_out')(x)
    return Activation('relu')(BatchNormalization()(out))

@tf.keras.utils.register_keras_serializable(package="Custom", name="sum_axis_1")
def sum_axis_1(x):
    return tf.reduce_sum(x, axis=1)

def attention_bilstm_module(x, prefix="final"):
    # 获取动态形状并 reshape
    shape = K.int_shape(x) # (None, 2, 128, 50)
    x = Reshape((shape[2], shape[1] * shape[3]))(x) # (None, 128, 100)
    
    lstm_out = Bidirectional(LSTM(128, return_sequences=True))(x)
    
    # Attention
    att = Dense(1, activation='tanh')(lstm_out)
    att = Flatten()(att)
    att = Activation('softmax')(att)
    att = RepeatVector(256)(att) # 128*2
    att = Permute([2, 1])(att)
    
    sent_representation = Multiply(name=f'{prefix}_att_mul')([lstm_out, att])
    # 显式指定 output_shape
    return Lambda(sum_axis_1, output_shape=(256,), name=f'{prefix}_att_sum')(sent_representation)

def HANet(input_shape=[2, 128], classes=11, **kwargs):
    input_layer = Input(shape=(2, 128, 1), name='input')
    
    # 关键：显式提供 output_shape，避免 Lambda 自动推断失败
    ap_tensor = Lambda(get_ap, output_shape=(2, 128, 1), name='get_ap_lambda')(input_layer)
    fft_tensor = Lambda(get_fft, output_shape=(2, 128, 1), name='get_fft_lambda')(input_layer)

    iq_output = feature_module(input_layer, prefix="IQ")
    ap_output = feature_module(ap_tensor, prefix="AP")
    fft_output = feature_module(fft_tensor, prefix="FFT") 

    fusion = Add(name='all_fusion_add')([iq_output, ap_output, fft_output])
    x = Conv2D(50, (2, 50), padding='same', activation='relu')(fusion)
    
    res_output = dsc_residual_module(x, prefix="global_dsc_res")
    rnn_output = attention_bilstm_module(res_output, prefix="final_rnn")

    dense_x = Dense(128, activation='relu')(rnn_output)
    out = Dense(classes, activation='softmax')(Dropout(0.5)(dense_x))

    return Model(inputs=input_layer, outputs=out)