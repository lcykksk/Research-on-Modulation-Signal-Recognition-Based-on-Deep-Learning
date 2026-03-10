
import os
# 优先关闭 oneDNN 避免日志干扰
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定GPU序号（0为第一块）
import sys
import pickle
import numpy as np
import mltools

# 再添加 GPU 检测配置
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"成功识别 GPU: {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        print(f"GPU 配置错误: {e}")
else:
    print("未找到 GPU,使用 CPU 训练（速度较慢）")
    # 可选：打印 TF 版本和 CUDA 检测信息，辅助排查
    print(f"TensorFlow 版本: {tf.__version__}")
    print(f"CUDA 可用: {tf.test.is_built_with_cuda()}")
    print(f"GPU 设备列表: {tf.config.list_logical_devices('GPU')}")
from rmldataset2016 import load_data
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
 
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# sys.path.append(os.path.abspath(f'/kaggle/working/AMC/1DCNN-PF')) # cant import from module starting with number
# sys.path.append(os.path.abspath(f'/kaggle/working/AMC/PET-CGDNN'))
# sys.path.append(os.path.abspath(f'/kaggle/working/AMC/IC-AMCNet'))
# sys.path.append(os.path.abspath(f'/kaggle/working/AMC/TAD'))
# from DCNNPF import DLmodel
# from CGDNet.CGDNN import CGDNN
# from CLDNN.CLDNNLikeModel import CLDNNLikeModel
# import CLDNN.CLDNNLikeModel as cldnn
# import CLDNN2.CLDNNLikeModel as cldnn2
from CNN1.CNN2Model import CNN2Model
# from CNN2.CNN2 import CNN2
# from DAE.DAE import DAE
# from DenseNet.DenseNet import DenseNet
# from GRU2.GRUModel import GRUModel
# from IC-AMCNet import ICAMC
# from LSTM2.CuDNNLSTMModel import LSTMModel
# from MCLDNN.MCLDNN import MCLDNN
# from MCNET.MCNET import MCNET
# from PETCGDNN import PETCGDNN
from ResNet.ResNet import ResNet
# from TAD import MCLDNN_VGN
from HANet import HANest

models_dict = { 
                # 'CNN1': CNN2Model,
            #    '1DCNN-PF': DLmodel,
            #     'CGDNet': CGDNN,
            #     'CLDNN': cldnn.CLDNNLikeModel,
            #     'CLDNN2': cldnn2.CLDNNLikeModel,
            #     'CNN2': CNN2,
            #     'DAE': DAE,
            #     'DenseNet': DenseNet,
            #     'GRU2': GRUModel,
            #     'IC-AMCNet': ICAMC,
            #     'LSTM2': LSTMModel,
            #     'MCLDNN': MCLDNN,
            #     'MCNET': MCNET,
            #     'PET-CGDNN': PETCGDNN,
                'ResNet': ResNet,
            #     'TAD': MCLDNN_VGN.MCLDNN
            #     'HANet': HANest
                }

idx = []
for i, idx_to_load in enumerate(["train_idx", "val_idx", "test_idx"]):
    with open(f'E:/study/py/final/RML2016.10a/{idx_to_load}.pkl', 'rb') as file:
        idx.append(pickle.load(file))
(mods,snrs,lbl),(X_train,Y_train),(X_val,Y_val),(X_test,Y_test),(train_idx,val_idx,test_idx) = load_data(idx=idx)

nb_epoch = 1000
batch_size = 512

# Helper functions
def rotate_matrix(theta):
    m = np.zeros((2, 2))
    m[0, 0] = np.cos(theta)
    m[0, 1] = -np.sin(theta)
    m[1, 0] = np.sin(theta)
    m[1, 1] = np.cos(theta)
    print(m)
    return m

def Rotate_DA(x, y):
    N, L, C = np.shape(x)
    x_rotate1 = np.matmul(x, rotate_matrix(np.pi/2))
    x_rotate2 = np.matmul(x, rotate_matrix(np.pi))
    x_rotate3 = np.matmul(x, rotate_matrix(3 * np.pi / 2))

    x_DA = np.vstack((x, x_rotate1, x_rotate2, x_rotate3))

    y_DA = np.tile(y, (1, 4)).T.reshape(-1)
    return x_DA, y_DA


def norm_pad_zeros(X_train,nsamples):
    print ("Pad:", X_train.shape)
    for i in range(X_train.shape[0]):
        X_train[i,:,0] = X_train[i,:,0]/np.linalg.norm(X_train[i,:,0],2)
    return X_train

def l2_normalize(x, axis=-1):
    y = np.sum(x ** 2, axis, keepdims=True)
    return x / np.sqrt(y)
    

def to_amp_phase(X_train,X_val,X_test,nsamples):
    X_train_cmplx = X_train[:,0,:] + 1j* X_train[:,1,:]
    X_val_cmplx = X_val[:,0,:] + 1j* X_val[:,1,:]
    X_test_cmplx = X_test[:,0,:] + 1j* X_test[:,1,:]
    
    X_train_amp = np.abs(X_train_cmplx)
    X_train_ang = np.arctan2(X_train[:,1,:],X_train[:,0,:])/np.pi
    
    
    X_train_amp = np.reshape(X_train_amp,(-1,1,nsamples))
    X_train_ang = np.reshape(X_train_ang,(-1,1,nsamples))
    
    X_train = np.concatenate((X_train_amp,X_train_ang), axis=1) 
    X_train = np.transpose(np.array(X_train),(0,2,1))

    X_val_amp = np.abs(X_val_cmplx)
    X_val_ang = np.arctan2(X_val[:,1,:],X_val[:,0,:])/np.pi
    
    
    X_val_amp = np.reshape(X_val_amp,(-1,1,nsamples))
    X_val_ang = np.reshape(X_val_ang,(-1,1,nsamples))
    
    X_val = np.concatenate((X_val_amp,X_val_ang), axis=1) 
    X_val = np.transpose(np.array(X_val),(0,2,1))
    
    X_test_amp = np.abs(X_test_cmplx)
    X_test_ang = np.arctan2(X_test[:,1,:],X_test[:,0,:])/np.pi
    
    
    X_test_amp = np.reshape(X_test_amp,(-1,1,nsamples))
    X_test_ang = np.reshape(X_test_ang,(-1,1,nsamples))
    
    X_test = np.concatenate((X_test_amp,X_test_ang), axis=1) 
    X_test = np.transpose(np.array(X_test),(0,2,1))
    return (X_train,X_val,X_test)

for model_, mdl in list(models_dict.items()):
    print("in\n")
    # train_data
    X_train_, X_val_, X_test_ = X_train.copy(), X_val.copy(), X_test.copy()
    print(X_train)
    if model_ in ['GRU2']:
        X_train_ = X_train_.swapaxes(2, 1)
        X_val_ = X_val_.swapaxes(2, 1)
    elif model_ in ['LSTM2']:
        X_train_, X_val_, X_test_ = to_amp_phase(X_train_, X_val_, X_test_, 128)
        X_train_ = norm_pad_zeros(X_train_[:, :128, :], 128)
        X_val_ = norm_pad_zeros(X_val_[:, :128, :], 128)
    elif model_ in ['1DCNN-PF']:
        X_train_, X_val_, X_test_ = to_amp_phase(X_train_, X_val_, X_test_, 128)

        X_train_ = norm_pad_zeros(X_train_[:, :128, :], 128)
        X_val_ = norm_pad_zeros(X_val_[:, :128, :], 128)
        
        X1_train = X_train_[:, :, 0]
        X1_val = X_val_[:, :, 0]
        X2_train = X_train_[:, :, 1]
        X2_val = X_val_[:, :, 1]
        X_train_ = [X1_train, X2_train]
        X_val_ = [X1_val, X2_val]
        
    elif model_ in ['CGDNet']:
        X_train_ = np.expand_dims(X_train_, axis=1)
        X_val_ = np.expand_dims(X_val_, axis=1)
        
    elif model_ in ['CLDNN']:
        X_train_ = np.reshape(X_train_, (-1, 1, 2, 128))
        X_val_ = np.reshape(X_val_, (-1, 1, 2, 128))
        
    elif model_ in ['CLDNN2', 'DenseNet', 'IC-AMCNet', 'ResNet', 'TAD', 'HANet']:
        X_train_ = np.expand_dims(X_train_, axis=3)
        X_val_ = np.expand_dims(X_val_, axis=3)
                               
    elif model_ in ['DAE']:
        X_train_, X_val_, X_test_ = to_amp_phase(X_train_, X_val_, X_test_, 128)
        X_train_[:, :, 0] = l2_normalize(X_train_[:, :, 0])
        X_val_[:, :, 0] = l2_normalize(X_val_[:, :, 0])
        for i in range(X_train_.shape[0]):
            k = 2/(X_train_[i,:,1].max() - X_train_[i,:,1].min())
        X_train_[i,:,1]=-1+k*(X_train_[i,:,1]-X_train_[i,:,1].min())
        for i in range(X_val_.shape[0]):
            k = 2/(X_val_[i,:,1].max() - X_val_[i,:,1].min())
        X_val_[i,:,1]=-1+k*(X_val_[i,:,1]-X_val_[i,:,1].min())

    elif model_ in ['MCLDNN']:
        X1_train = np.expand_dims(X_train_[:, 0, :], axis=2)
        X1_val = np.expand_dims(X_val_[:, 0, :], axis=2)
        X2_train = np.expand_dims(X_train_[:, 1, :], axis=2)
        X2_val = np.expand_dims(X_val_[:, 1, :], axis=2)
        X_train_t = np.expand_dims(X_train_, axis=3)
        X_val_t = np.expand_dims(X_val_, axis=3)        
        X_train_ = [X_train_t, X1_train, X2_train]
        X_val_ = [X_val_t, X1_val, X2_val]

    elif model_ in ['MCNET']:
        X_train_ = np.expand_dims(X_train_, axis=3)
        X_val_ = np.expand_dims(X_val_, axis=3)

    elif model_ in ['PET-CGDNN']:
        X_train_ = X_train_.swapaxes(2, 1)
        X_val_ = X_val_.swapaxes(2, 1)
        X1_train = X_train_[:, :, 0]
        X2_train = X_train_[:, :, 1]
        X1_val = X_val_[:, :, 0]
        X2_val = X_val_[:, :, 1]
        X_train_ = np.expand_dims(X_train_, axis=3)
        X_val_ = np.expand_dims(X_val_, axis=3)
        X_train_ = [X_train_, X1_train, X2_train]
        X_val_ = [X_val_, X1_val, X2_val]
    else:
        X_train_ = X_train
        X_val_ = X_val

    if model_ in ['DAE']:
        Y_train_ = [Y_train, X_train_]
        Y_val_ = [Y_val, X_val_]
    else:
        Y_train_ = Y_train
        Y_val_ = Y_val
    
    filepath = f'results/{model_}'
    os.makedirs(filepath, exist_ok=True)

    # optimization params
    if model_ in ['DAE']:
        compile_params = {'optimizer':Adam(),
                          'loss':{'xc': 'categorical_crossentropy', 'xd': 'mean_squared_error'},
                          'loss_weights':{'xc': 0.1, 'xd': 0.9},
                          'metrics':['accuracy', 'mse']}
    else:
        compile_params = {'optimizer':Adam(),
                          'loss':'categorical_crossentropy',
                          'metrics':['accuracy']}
    print(X_train_.shape)  
    model = mdl()
    model.compile(**compile_params)
    callbacks=[
        ModelCheckpoint(filepath + '/weights.keras', monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1, patience=5, min_lr=0.0000001),
        EarlyStopping(monitor='val_loss', patience=30, verbose=1, mode='auto', min_delta=0.00001)
    ]
    history = model.fit(X_train_, Y_train_, batch_size=batch_size, epochs=nb_epoch, verbose=0,
                        validation_data=[X_val_, Y_val_], callbacks=callbacks)
    if model_ == 'DAE':
        history.history['accuracy'] = history.history['xc_accuracy']
        history.history['val_accuracy'] = history.history['val_xc_accuracy']
    mltools.show_history(history, filepath)
    print(f'Finished training {model_}')