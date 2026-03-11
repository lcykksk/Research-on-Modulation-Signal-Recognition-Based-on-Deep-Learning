import os
import numpy as np
import matplotlib.pyplot as plt
import csv

RESULTS_DIR = 'results'
SAVE_DIR = 'train_analysis'
os.makedirs(SAVE_DIR, exist_ok=True)


def read_metric_file(file_path):
    """读取txt中的浮点数，每行一个"""
    values = []
    if not os.path.exists(file_path):
        return values

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    values.append(float(line))
                except ValueError:
                    print(f"警告: 文件 {file_path} 中存在无法解析的内容: {line}")
    return values


def moving_average(data, window=3):
    """简单滑动平均，便于观察趋势"""
    if len(data) < window or window <= 1:
        return np.array(data)
    data = np.array(data)
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')


def collect_all_models(results_dir):
    """收集results下所有模型的acc/loss"""
    model_data = {}

    for model_name in sorted(os.listdir(results_dir)):
        model_dir = os.path.join(results_dir, model_name)
        if not os.path.isdir(model_dir):
            continue

        acc_file = os.path.join(model_dir, 'train_acc.txt')
        loss_file = os.path.join(model_dir, 'train_loss.txt')

        acc = read_metric_file(acc_file)
        loss = read_metric_file(loss_file)

        if len(acc) == 0 and len(loss) == 0:
            continue

        model_data[model_name] = {
            'acc': acc,
            'loss': loss
        }

    return model_data


def plot_all_accuracy(model_data, save_dir):
    plt.figure(figsize=(12, 8))
    has_data = False

    for model_name, metrics in model_data.items():
        acc = metrics['acc']
        if len(acc) == 0:
            continue
        epochs = np.arange(1, len(acc) + 1)
        plt.plot(epochs, acc, marker='o', linewidth=2, label=model_name)
        has_data = True

    if not has_data:
        print("没有找到可绘制的 accuracy 数据")
        return

    plt.xlabel('Epoch')
    plt.ylabel('Training Accuracy')
    plt.title('Training Accuracy Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'all_models_train_accuracy.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"已保存: {save_path}")


def plot_all_loss(model_data, save_dir):
    plt.figure(figsize=(12, 8))
    has_data = False

    for model_name, metrics in model_data.items():
        loss = metrics['loss']
        if len(loss) == 0:
            continue
        epochs = np.arange(1, len(loss) + 1)
        plt.plot(epochs, loss, marker='o', linewidth=2, label=model_name)
        has_data = True

    if not has_data:
        print("没有找到可绘制的 loss 数据")
        return

    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'all_models_train_loss.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"已保存: {save_path}")


def plot_all_accuracy_smooth(model_data, save_dir, window=3):
    plt.figure(figsize=(12, 8))
    has_data = False

    for model_name, metrics in model_data.items():
        acc = metrics['acc']
        if len(acc) == 0:
            continue
        smoothed = moving_average(acc, window=window)
        epochs = np.arange(window, window + len(smoothed))
        plt.plot(epochs, smoothed, linewidth=2.5, label=f'{model_name} (smooth)')
        has_data = True

    if not has_data:
        print("没有找到可绘制的平滑 accuracy 数据")
        return

    plt.xlabel('Epoch')
    plt.ylabel('Training Accuracy')
    plt.title(f'Smoothed Training Accuracy Comparison (window={window})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'all_models_train_accuracy_smooth.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"已保存: {save_path}")


def plot_all_loss_smooth(model_data, save_dir, window=3):
    plt.figure(figsize=(12, 8))
    has_data = False

    for model_name, metrics in model_data.items():
        loss = metrics['loss']
        if len(loss) == 0:
            continue
        smoothed = moving_average(loss, window=window)
        epochs = np.arange(window, window + len(smoothed))
        plt.plot(epochs, smoothed, linewidth=2.5, label=f'{model_name} (smooth)')
        has_data = True

    if not has_data:
        print("没有找到可绘制的平滑 loss 数据")
        return

    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title(f'Smoothed Training Loss Comparison (window={window})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'all_models_train_loss_smooth.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"已保存: {save_path}")


def plot_single_model(model_name, metrics, save_dir):
    acc = metrics['acc']
    loss = metrics['loss']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if len(acc) > 0:
        epochs_acc = np.arange(1, len(acc) + 1)
        axes[0].plot(epochs_acc, acc, marker='o', linewidth=2)
        axes[0].set_title(f'{model_name} - Train Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].grid(True)
    else:
        axes[0].set_title(f'{model_name} - Train Accuracy (No Data)')

    if len(loss) > 0:
        epochs_loss = np.arange(1, len(loss) + 1)
        axes[1].plot(epochs_loss, loss, marker='o', linewidth=2)
        axes[1].set_title(f'{model_name} - Train Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].grid(True)
    else:
        axes[1].set_title(f'{model_name} - Train Loss (No Data)')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{model_name}_train_curve.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"已保存: {save_path}")


def save_summary_csv(model_data, save_dir):
    csv_path = os.path.join(save_dir, 'summary.csv')

    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Model',
            'Acc Epochs',
            'Final Acc',
            'Best Acc',
            'Best Acc Epoch',
            'Loss Epochs',
            'Final Loss',
            'Min Loss',
            'Min Loss Epoch'
        ])

        for model_name, metrics in model_data.items():
            acc = metrics['acc']
            loss = metrics['loss']

            acc_epochs = len(acc)
            loss_epochs = len(loss)

            final_acc = acc[-1] if len(acc) > 0 else ''
            best_acc = max(acc) if len(acc) > 0 else ''
            best_acc_epoch = (np.argmax(acc) + 1) if len(acc) > 0 else ''

            final_loss = loss[-1] if len(loss) > 0 else ''
            min_loss = min(loss) if len(loss) > 0 else ''
            min_loss_epoch = (np.argmin(loss) + 1) if len(loss) > 0 else ''

            writer.writerow([
                model_name,
                acc_epochs,
                final_acc,
                best_acc,
                best_acc_epoch,
                loss_epochs,
                final_loss,
                min_loss,
                min_loss_epoch
            ])

    print(f"已保存: {csv_path}")


def print_simple_analysis(model_data):
    print("\n========== 训练情况简要分析 ==========")
    for model_name, metrics in model_data.items():
        acc = metrics['acc']
        loss = metrics['loss']

        print(f"\n模型: {model_name}")

        if len(acc) > 0:
            print(f"  最终训练准确率: {acc[-1]:.4f}")
            print(f"  最高训练准确率: {max(acc):.4f} (Epoch {np.argmax(acc)+1})")
        else:
            print("  无 accuracy 数据")

        if len(loss) > 0:
            print(f"  最终训练损失: {loss[-1]:.4f}")
            print(f"  最低训练损失: {min(loss):.4f} (Epoch {np.argmin(loss)+1})")
        else:
            print("  无 loss 数据")

        if len(acc) >= 2:
            if acc[-1] - acc[0] > 0.2:
                print("  分析: 准确率提升明显，模型学习有效。")
            elif acc[-1] - acc[0] > 0.05:
                print("  分析: 准确率有一定提升，但可能还有优化空间。")
            else:
                print("  分析: 准确率提升不明显，可能存在欠拟合或模型设计问题。")

        if len(loss) >= 2:
            if loss[-1] < loss[0] * 0.7:
                print("  分析: loss 下降明显，优化过程较稳定。")
            elif loss[-1] < loss[0] * 0.9:
                print("  分析: loss 有所下降，但下降幅度一般。")
            else:
                print("  分析: loss 下降不明显，建议检查学习率、结构或数据预处理。")


def main():
    model_data = collect_all_models(RESULTS_DIR)

    if len(model_data) == 0:
        print("未在 results 文件夹中找到有效的模型训练记录。")
        return

    print(f"共找到 {len(model_data)} 个模型:")
    for model_name in model_data.keys():
        print(" -", model_name)

    plot_all_accuracy(model_data, SAVE_DIR)
    plot_all_loss(model_data, SAVE_DIR)
    plot_all_accuracy_smooth(model_data, SAVE_DIR, window=3)
    plot_all_loss_smooth(model_data, SAVE_DIR, window=3)

    for model_name, metrics in model_data.items():
        plot_single_model(model_name, metrics, SAVE_DIR)

    save_summary_csv(model_data, SAVE_DIR)
    print_simple_analysis(model_data)

    print(f"\n全部完成，结果保存在: {SAVE_DIR}")


if __name__ == '__main__':
    main()
