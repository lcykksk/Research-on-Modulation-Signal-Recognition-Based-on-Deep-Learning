import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Softmax, Dropout, Conv2D, Multiply, MaxPooling1D, BatchNormalization, GRU, Concatenate, Lambda, Add
from tensorflow.keras.models import Model

def feature_module(input_shape):
    
    # 信号分离
    A = Lambda(lambda x: x[:, 0:1, :, :], name='A_slice')(input)
    B = Lambda(lambda x: x[:, 1:2, :, :], name='B_slice')(input)

    # 通道卷积 (50,1×8)
    A_conv = Conv2D(8, (1, 50), padding='same', activation='relu', name='A_conv')(A)
    B_conv = Conv2D(8, (1, 50), padding='same', activation='relu', name='B_conv')(B)

    # A + B
    AB_add = Add(name='AB_add')([A_conv, B_conv])

    # Conv (50,1×8)
    AB_fuse = Conv2D(8, (1, 50), padding='same', activation='relu', name='IQ_fuse_conv')(AB_add)

    # 整体卷积 (50,2×8)
    AB_conv = Conv2D(16, (2, 50), padding='same', activation='relu', name='IQ_joint_conv')(input)

    # 相加
    fusion = Add(name='fusion_add')([AB_fuse, AB_conv])

    # 最后一层卷积 (50,2×8)
    x = Conv2D(16, (2, 50), padding='same', activation='relu', name='final_conv')(fusion)

    return x

# -------------------------
# Attention模块
# -------------------------
def attention_block(inputs):
    """
    Temporal Attention
    inputs: (batch, time, features)
    """
    score = Dense(1, activation='tanh')(inputs)
    weights = Softmax(axis=1)(score)
    context = Multiply()([inputs, weights])
    context = tf.reduce_sum(context, axis=1)
    return context


def get_ap(input):
    complex_signal = input[0,:,:] + 1j*input[1,:,:]
    amp = np.abs(complex_signal)
    phase = np.arctan2(input[1,:,:], input[0,:,:])
    ap_output = np.stack([amp, phase], axis=0)
    return ap_output

def get_fft(input):
    complex_signal = input[0,:,:] + 1j*input[1,:,:]
    fft_signal = np.fft.fft(complex_signal, axis=0)
    
    fft_real = np.real(fft_signal)
    fft_imag = np.imag(fft_signal)
    fft_output = np.stack([fft_real, fft_imag], axis=0)
    return fft_output

def HANet(weights=None, input_shape=[2, 128], classes=11,**kwargs):
    if weights is not None and not os.path.exists(weights):
        raise ValueError('The `weights` argument should be either '
                            '`None` (random initialization), '
                            'or the path to the weights file to be loaded.')
    dr = 0.6
    # input
    input = Input(input_shape + [1], name='input')
    print(input.shape)
    iq_input = input
    #得到AP
    ap_input = get_ap(input)
    #得到fft
    fft_input = get_fft(input)

    iq_output= feature_module(iq_input)
    ap_output = feature_module(ap_input)
    fft_output = feature_module(fft_output)
    # 特征融合
    fusion = Add(name='all_fusion_add')([iq_output, ap_output, fft_output])
    
    
    # 卷积 (50,2×8)
    x = Conv2D(16, (2, 50), padding='same', activation='relu', name='final_conv')(fusion)

    x = Dense(128, activation='relu')(fusion)
    x = Dropout(0.5)(x)

    out = Dense(classes, activation='softmax')(x)

    model = Model(inputs=[iq_input, ap_input], outputs=out)

    return model


