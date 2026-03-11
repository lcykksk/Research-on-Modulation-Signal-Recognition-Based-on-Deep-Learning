import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf  
from tensorflow.keras.models import load_model
from rmldataset2016 import load_data
from HANet.HANet import HANet, get_ap, get_fft, sum_axis_1 

def l2_normalize(x, axis=-1):
    y = np.sum(x ** 2, axis, keepdims=True)
    return x / np.sqrt(y)

def to_amp_phase(X_test, nsamples=128):
    X_test_cmplx = X_test[:,0,:] + 1j* X_test[:,1,:]
    X_test_amp = np.abs(X_test_cmplx)
    X_test_ang = np.arctan2(X_test[:,1,:],X_test[:,0,:])/np.pi
    X_test_amp = np.reshape(X_test_amp,(-1,1,nsamples))
    X_test_ang = np.reshape(X_test_ang,(-1,1,nsamples))
    X_test = np.concatenate((X_test_amp,X_test_ang), axis=1) 

    return np.transpose(np.array(X_test),(0,2,1))

# 补充一个针对单组数据的辅助函数
def to_amp_phase_single(X, nsamples):
    X_cmplx = X[:, 0, :] + 1j * X[:, 1, :]
    X_amp = np.abs(X_cmplx)
    X_ang = np.arctan2(X[:, 1, :], X[:, 0, :]) / np.pi
    X_amp = np.reshape(X_amp, (-1, 1, nsamples))
    X_ang = np.reshape(X_ang, (-1, 1, nsamples))
    X_out = np.concatenate((X_amp, X_ang), axis=1)
    return np.transpose(np.array(X_out), (0, 2, 1))

def norm_pad_zeros(X_train,nsamples):
    print ("Pad:", X_train.shape)
    for i in range(X_train.shape[0]):
        X_train[i,:,0] = X_train[i,:,0]/np.linalg.norm(X_train[i,:,0],2)
    return X_train

def preprocess_data(X, model_name, nsamples=128):
    """根据训练时的逻辑对测试数据进行预处理"""
    X_p = X.copy()
    if model_name in ['LSTM2']: 
        # 1. 转换：to_amp_phase (输出形状为 (N, 128, 2))
        X_p = to_amp_phase(X_p, nsamples)
        # 2. 补零与归一化：取前128点进行L2标准化
        X_p = norm_pad_zeros(X_p[:, :128, :], 128)
    elif model_name in ['CLDNN']:
        X_p = np.reshape(X_p, (-1, 1, 2, 128))
    elif model_name in ['ResNet', 'HANet', 'CLDNN2', 'DenseNet', 'MCNET']:
        X_p = np.expand_dims(X_p, axis=3)
    elif model_name in [ 'CNN1']:
        pass
    elif model_name in ['DAE']:
        X_p = to_amp_phase(X_p, 128)
        X_p[:, :, 0] = l2_normalize(X_p[:, :, 0])
        for i in range(X_p.shape[0]):
            k = 2/(X_p[i,:,1].max() - X_p[i,:,1].min())
        X_p[i,:,1]=-1+k*(X_p[i,:,1]-X_p[i,:,1].min())
     
    return X_p


def main():
    # 加载测试集索引
    with open('/root/autodl-tmp/RML2016.10a/test_idx.pkl', 'rb') as f:
        test_idx = pickle.load(f)
    # 【关键修改】：load_data 返回的 Y_test 已经是对齐的，lbl 需要同步切分
    (mods, snrs, lbl_all), _, _, (X_test, Y_test), _ = load_data(idx=[[], [], test_idx])
    
    # 仅保留测试集对应的标签部分
    lbl_test = [lbl_all[i] for i in test_idx]
    
    model_list = [
                    'ResNet', 
                    'HANet', 
                    'CLDNN', 
                    'CNN1', 
                    'DAE', 
                    'DenseNet', 
                    'LSTM2', 
                    'MCNET'
                    ]
    test_results_dir = 'test_result'
    os.makedirs(test_results_dir, exist_ok=True)
    
    all_snr_acc = {}

    for model_name in model_list:
        print(f"正在测试模型: {model_name}")
        
        # 1. 预处理
        X_test_processed = preprocess_data(X_test, model_name)
        
        # 2. 加载模型
        weight_path = f'results/{model_name}/weights.keras'
        if not os.path.exists(weight_path):
            print(f"警告: 未找到模型 {model_name}，跳过...")
            continue
        
        model = load_model(
            weight_path, 
            custom_objects={
                'get_ap': get_ap, 
                'get_fft': get_fft,
                'sum_axis_1': sum_axis_1, # 必须添加这一项
                'Lambda': tf.keras.layers.Lambda
            },
            safe_mode=False
        )
        
        # 3. 分 SNR 测试
        snr_acc = {}
        for snr in snrs:
            # 【关键修改】：这里必须使用 lbl_test 而不是 lbl_all
            idx_snr = [i for i, l in enumerate(lbl_test) if l[1] == snr]
            
            # 如果 idx_snr 为空，说明该 SNR 在测试集中没有样本
            if not idx_snr:
                continue
                
            X_snr = X_test_processed[idx_snr]
            Y_snr = Y_test[idx_snr]
            
            # 评估
            if model_name == 'DAE':
                # 训练时 DAE 的 Y_train_ 结构是 [Y_train, X_train_]
                # 这里必须将 X_snr 也作为第二项标签传入
                Y_target = [Y_snr, X_snr] 
                results = model.evaluate(X_snr, Y_target, verbose=0)
                
                # evaluate 返回列表: [loss_total, loss_xc, loss_xd, acc_xc, acc_xd]
                # acc_xc 是分类准确率 (通常是列表的第 3 个或第 4 个，取决于 compile 顺序)
                # 请根据你的 train.py 中 metrics 的顺序调整下标
                acc = results[3] # 假设: loss, loss_xc, loss_xd, acc_xc, acc_xd
            else:
                # 常规分类模型
                loss, acc = model.evaluate(X_snr, Y_snr, verbose=0)
            
            snr_acc[str(snr)] = acc
            print(f"SNR {snr}dB: Accuracy = {acc:.4f}")
        
        all_snr_acc[model_name] = snr_acc
        
        # 保存该模型结果
        with open(os.path.join(test_results_dir, f'{model_name}_acc.pkl'), 'wb') as f:
            pickle.dump(snr_acc, f)

    # 4. 绘图总结
    plt.figure(figsize=(12, 8))
    for name, acc_dict in all_snr_acc.items():
        plt.plot(snrs, [acc_dict[str(s)] for s in snrs], label=name, marker='o')
    
    plt.xlabel('SNR (dB)')
    plt.ylabel('Accuracy')
    plt.title('Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(test_results_dir, 'summary_accuracy.png'))
    print("测试完成，结果已保存至 test_result 文件夹。")

if __name__ == "__main__":
    main()