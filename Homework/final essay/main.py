# -*- coding: utf-8 -*-
"""
时间序列预测完整流程实现（部分特征归一化版本）
特征：位置、速度和加速度特征，预测多个输出目标
模型：优化后的LSTM神经网络，支持自定义激活函数
功能：支持多步预测、自动超参数调优、防数据泄露
"""

# ==================== 依赖库导入 ====================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from typing import List, Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass, field
import datetime
import json

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.layers import Activation
import keras.backend as K


# 添加自定义激活函数
def mish(x):
    """Mish激活函数: x * tanh(softplus(x))"""
    return x * K.tanh(K.softplus(x))

# 注册自定义激活函数
tf.keras.utils.get_custom_objects().update({'mish': Activation(mish)})


@dataclass
class Config:
    """配置类，存储所有可调整的参数"""
    # 数据处理参数
    input_columns: List[str] = field(default_factory=lambda: ['x1', 'y1', 'x2', 'y2'])
    output_columns: List[str] = field(default_factory=lambda: ['x1', 'y1', 'x2', 'y2'])
    # 速度和加速度特征（需要归一化处理，因此单独分开）
    velocity_columns: List[str] = field(default_factory=lambda: ['xx1', 'yy1', 'xx2', 'yy2'])
    acceleration_columns: List[str] = field(default_factory=lambda: ['xxx1', 'yyy1', 'xxx2', 'yyy2'])
    # 是否使用速度和加速度特征
    use_velocity: bool = False
    use_acceleration: bool = False
    
    data_path: str = '/Users/chenjingxu/Desktop/20250417-save/2016/200-50/*.csv'
    # 采样率（从原始数据中每几个数据点采样一次）和训练测试集划分比例
    sampling_rate: int = 10
    train_test_split: float = 0.9
    
    # 序列生成参数
    time_steps: int = 200
    pred_steps: int = 50
    
    # 模型参数
    lstm_units: int = 256
    dropout_rate: float = 0.1
    learning_rate: float = 5e-4
    lstm_layers: int = 2
    lstm_activation: str = 'tanh'  # 新增LSTM激活函数参数，默认为tanh
    
    # 训练参数
    epochs: int = 400
    batch_size: int = 256
    validation_split: float = 0.2
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.2
    
    # 输出参数
    model_save_path: str = 'optimized_lstm_model.keras'
    training_plot_path: str = 'training_process.png'
    prediction_plot_path: str = 'actual_vs_predicted.png'
    
    # 特征归一化参数
    feature_scalers: Dict[str, Any] = field(default_factory=dict)


def get_all_feature_columns(config: Config) -> List[str]:
    """获取所有使用的特征列名称"""
    all_columns = config.input_columns.copy()
    if config.use_velocity:
        all_columns.extend(config.velocity_columns)
    if config.use_acceleration:
        all_columns.extend(config.acceleration_columns)
    return all_columns


def load_and_preprocess_data(config: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    加载和预处理数据
    
    Args:
        config: 配置对象
        
    Returns:
        训练数据和测试数据
    """
    print("开始加载和预处理数据...")
    
    # 加载原始数据
    csv_files = glob.glob(config.data_path)
    all_dfs = []

    for file in sorted(csv_files):
        print(f"处理文件: {file}")
        df = pd.read_csv(file).iloc[::config.sampling_rate, :]
        df['file_id'] = os.path.splitext(os.path.basename(file))[0]
        all_dfs.append(df)

    raw_df = pd.concat(all_dfs).reset_index(drop=True)
    
    # 按文件分割训练测试集
    train_dfs, test_dfs = [], []
    for file_id, group in raw_df.groupby('file_id'):
        split_idx = int(len(group) * config.train_test_split)
        train_part = group.iloc[:split_idx].copy()
        test_part = group.iloc[split_idx:].copy()
        
        train_part.reset_index(drop=True, inplace=True)
        test_part.reset_index(drop=True, inplace=True)
        
        train_dfs.append(train_part)
        test_dfs.append(test_part)
    
    train_data = pd.concat(train_dfs, ignore_index=True)
    test_data = pd.concat(test_dfs, ignore_index=True)
    
    # 对速度和加速度特征进行归一化处理
    if config.use_velocity:
        train_data, test_data = normalize_features(
            train_data, test_data, config.velocity_columns, config)
    
    if config.use_acceleration:
        train_data, test_data = normalize_features(
            train_data, test_data, config.acceleration_columns, config)
    
    print(f"预处理完成。训练集大小: {len(train_data)}，测试集大小: {len(test_data)}")
    return train_data, test_data


def normalize_features(train_data: pd.DataFrame, test_data: pd.DataFrame, 
                      feature_columns: List[str], config: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    对指定特征列进行归一化处理
    
    Args:
        train_data: 训练数据
        test_data: 测试数据
        feature_columns: 需要归一化的特征列
        config: 配置对象
        
    Returns:
        归一化后的训练数据和测试数据
    """
    train_data_copy = train_data.copy()
    test_data_copy = test_data.copy()
    
    for col in feature_columns:
        if col in train_data.columns:
            # 使用MinMaxScaler将特征缩放到[-1, 1]范围
            scaler = MinMaxScaler(feature_range=(-1, 1))
            train_data_copy[col] = scaler.fit_transform(train_data[[col]])
            test_data_copy[col] = scaler.transform(test_data[[col]])
            
            # 存储缩放器以便后续使用
            config.feature_scalers[col] = scaler
            
    return train_data_copy, test_data_copy


def create_sequences(input_data: np.ndarray, output_data: np.ndarray, 
                    window_size: int, pred_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    创建时间序列数据
    
    Args:
        input_data: 输入特征数据
        output_data: 输出目标数据
        window_size: 输入窗口大小
        pred_steps: 预测步长
        
    Returns:
        处理后的特征序列和目标序列
    """
    X, y = [], []
    for i in range(len(input_data)-window_size-pred_steps+1):
        # 检查数据有效性
        if np.isfinite(input_data[i:i+window_size]).all():
            X.append(input_data[i:i+window_size])
            y.append(output_data[i+window_size:i+window_size+pred_steps])
    return np.array(X), np.array(y)


def generate_train_test_sequences(train_data: pd.DataFrame, test_data: pd.DataFrame, 
                                config: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    为训练和测试数据生成序列
    
    Args:
        train_data: 训练数据
        test_data: 测试数据
        config: 配置对象
        
    Returns:
        训练特征、训练目标、测试特征、测试目标
    """
    print("生成时序序列数据...")
    
    # 获取所有输入特征列
    all_input_columns = get_all_feature_columns(config)
    
    # 提取原始特征和目标
    raw_train_input = train_data[all_input_columns].values
    raw_train_output = train_data[config.output_columns].values
    raw_test_input = test_data[all_input_columns].values
    raw_test_output = test_data[config.output_columns].values
    
    # 生成训练数据序列
    X_train, y_train = [], []
    for file_id, group in train_data.groupby('file_id'):
        start_idx = group.index[0]
        end_idx = group.index[-1]
        
        file_inputs = raw_train_input[start_idx:end_idx+1]
        file_outputs = raw_train_output[start_idx:end_idx+1]
        
        file_X, file_y = create_sequences(file_inputs, file_outputs, 
                                         config.time_steps, config.pred_steps)
        X_train.append(file_X)
        y_train.append(file_y)
    
    # 生成测试数据序列
    X_test, y_test = [], []
    for file_id, group in test_data.groupby('file_id'):
        start_idx = group.index[0]
        end_idx = group.index[-1]
        
        file_inputs = raw_test_input[start_idx:end_idx+1]
        file_outputs = raw_test_output[start_idx:end_idx+1]
        
        file_X_test, file_y_test = create_sequences(file_inputs, file_outputs, 
                                                  config.time_steps, config.pred_steps)
        X_test.append(file_X_test)
        y_test.append(file_y_test)
    
    # 合并和调整维度
    X_train = np.concatenate(X_train) if X_train else np.array([])
    y_train = np.concatenate(y_train) if y_train else np.array([])
    X_test = np.concatenate(X_test) if X_test else np.array([])
    y_test = np.concatenate(y_test) if y_test else np.array([])
    
    # 调整目标维度
    y_train = y_train.reshape(y_train.shape[0], -1)
    y_test = y_test.reshape(y_test.shape[0], -1)
    
    print(f"序列生成完成。训练样本: {len(X_train)}，测试样本: {len(X_test)}")
    print(f"输入特征维度: {X_train.shape}")
    return X_train, y_train, X_test, y_test


def build_lstm_model(config: Config) -> Model:
    """
    构建LSTM模型
    
    Args:
        config: 配置对象
        
    Returns:
        构建好的LSTM模型
    """
    model = Sequential()
    
    # 获取输入特征维度
    input_dim = len(get_all_feature_columns(config))
    
    # 双向LSTM输入层，使用指定的激活函数
    model.add(Bidirectional(
        LSTM(config.lstm_units, return_sequences=True, activation=config.lstm_activation),
        input_shape=(config.time_steps, input_dim)
    ))
    model.add(Dropout(config.dropout_rate))
    
    # 添加多个LSTM层，使用指定的激活函数
    for _ in range(config.lstm_layers):
        model.add(LSTM(config.lstm_units, return_sequences=True, activation=config.lstm_activation))
        model.add(Dropout(config.dropout_rate))
    
    # 输出层前置处理，使用指定的激活函数
    model.add(LSTM(config.lstm_units//2, activation=config.lstm_activation))
    model.add(Dropout(config.dropout_rate))
    
    # 全连接输出层（适配多步预测）
    model.add(Dense(len(config.output_columns)*config.pred_steps))
    
    # 模型编译配置
    model.compile(
        optimizer=Adam(learning_rate=config.learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def train_model(model: Model, X_train: np.ndarray, y_train: np.ndarray, 
               config: Config) -> Tuple[Model, Dict]:
    """
    训练LSTM模型
    
    Args:
        model: LSTM模型
        X_train: 训练特征
        y_train: 训练目标
        config: 配置对象
        
    Returns:
        训练好的模型和训练历史
    """
    print("开始模型训练...")
    
    # 训练回调配置
    training_callbacks = [
        EarlyStopping(
            monitor='val_loss', 
            patience=config.early_stopping_patience, 
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss', 
            factor=config.reduce_lr_factor, 
            patience=config.reduce_lr_patience
        )
    ]
    
    # 训练模型
    training_history = model.fit(
        X_train, y_train,
        epochs=config.epochs,
        batch_size=config.batch_size,
        validation_split=config.validation_split,
        callbacks=training_callbacks,
        shuffle=False
    )
    
    print(f"模型训练完成，共经历 {len(training_history.history['loss'])} 个Epochs")
    return model, training_history.history


def evaluate_model(model: Model, X_test: np.ndarray, y_test: np.ndarray, 
                  config: Config) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, float, float]]]:
    """
    评估模型性能
    
    Args:
        model: 训练好的模型
        X_test: 测试特征
        y_test: 测试目标
        config: 配置对象
        
    Returns:
        预测结果、真实值和各个预测步长的性能指标
    """
    print("评估模型性能...")
    
    # 预测测试数据
    test_predictions = model.predict(X_test)
    
    # 重构预测结果维度
    test_predictions = test_predictions.reshape(-1, config.pred_steps, len(config.output_columns))
    y_test_reshaped = y_test.reshape(-1, config.pred_steps, len(config.output_columns))
    
    # 计算各个步长的性能指标
    performance_metrics = []
    for step in range(config.pred_steps):
        mse = mean_squared_error(y_test_reshaped[:, step], test_predictions[:, step])
        mae = mean_absolute_error(y_test_reshaped[:, step], test_predictions[:, step])
        performance_metrics.append((step+1, mse, mae))
    
    return test_predictions, y_test_reshaped, performance_metrics


def plot_training_history(history: Dict, config: Config) -> None:
    """
    可视化训练过程
    
    Args:
        history: 训练历史
        config: 配置对象
    """
    plt.figure(figsize=(12, 6))
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Training Progress')
    plt.ylabel('Loss Value')
    plt.xlabel('Training Epochs')
    plt.legend()
    plt.savefig(config.training_plot_path, dpi=300, bbox_inches='tight')
    plt.close


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, config: Config) -> None:
    """
    可视化预测结果
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        config: 配置对象
    """
    plt.figure(figsize=(18, 12))
    feature_names = [col.upper() for col in config.output_columns]
    
    for idx, feature in enumerate(feature_names):
        plt.subplot(len(feature_names), 1, idx+1)
        plt.plot(y_true[:100, 0, idx], label='Actual Values', linewidth=1.5)
        plt.plot(y_pred[:100, 0, idx], label='Predicted Values', linestyle='--')
        plt.title(f'{feature} - Actual vs Predicted Comparison')
        plt.ylabel(feature)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(config.prediction_plot_path, dpi=300, bbox_inches='tight')
    plt.close()


def print_performance_metrics(metrics: List[Tuple[int, float, float]]) -> None:
    """
    打印模型性能指标
    
    Args:
        metrics: 性能指标列表
    """
    print("\nMulti-step Prediction Performance:")
    print("Step | MSE\t\t| MAE")
    for step, mse, mae in metrics:
        print(f"T+{step} | {mse:.6f}\t| {mae:.6f}")


def get_activation_string(config: Config) -> str:
    """
    获取激活函数字符串表示，用于文件名
    
    Args:
        config: 配置对象
        
    Returns:
        激活函数的字符串表示
    """
    return config.lstm_activation


def get_features_string(config: Config) -> str:
    """
    获取使用的特征集合的字符串表示，用于文件名
    
    Args:
        config: 配置对象
        
    Returns:
        特征集合的字符串表示
    """
    features = ["pos"]  # 默认使用位置特征
    if config.use_velocity:
        features.append("vel")
    if config.use_acceleration:
        features.append("acc")
    
    return "_".join(features)

# 定义对比图绘制函数
def plot_new_comparison(config, start_idx, true_vals, preds, save_path):
    plt.figure(figsize=(12, 8))
    for i, col in enumerate(config.output_columns):
        plt.subplot(len(config.output_columns), 1, i+1)
        plt.plot(true_vals[:, i], 'b-', label='True')
        plt.plot(preds[:, i], 'r--', label='Pred')
        plt.title(f'{col} @start {start_idx}')
        plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    

def main() -> None:
    """主函数，支持多组参数配置的循环训练和保存"""
    # 定义多种参数组合，按需添加或修改
    param_sets = [
        {},
    ]

    for idx, params in enumerate(param_sets, start=1):
        print(f"\n=== Experiment #{idx} start ===")
        config = Config()
        # 更新配置
        for k, v in params.items():
            setattr(config, k, v)

        # 生成唯一后缀：编号 + 时间戳
        ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        suffix = f"{idx}_{ts}"
        # 更新输出文件名
        config.model_save_path = f"model_{suffix}.keras"
        config.training_plot_path = f"train_{suffix}.png"
        config.prediction_plot_path = f"pred_{suffix}.png"

        print(f"使用配置#{idx}: {params}")
        print(f"输出前缀: {suffix}")

        # 数据处理和序列生成调用
        train_data, test_data = load_and_preprocess_data(config)
        X_train, y_train, X_test, y_test = generate_train_test_sequences(
            train_data, test_data, config)

        # 构建、训练、保存模型
        model = build_lstm_model(config)
        model.summary()
        model, history = train_model(model, X_train, y_train, config)
        # 保存训练历史到 JSON
        history_path = f"history_{suffix}.json"
        with open(history_path, 'w') as hf:
            json.dump(history, hf)
        print(f"训练历史已保存到 {history_path}")
        model.save(config.model_save_path)
        print(f"模型已保存到 {config.model_save_path}")

        # 评估并可视化
        test_preds, y_test_r, metrics = evaluate_model(model, X_test, y_test, config)
        print_performance_metrics(metrics)
        plot_training_history(history, config)
        plot_predictions(y_test_r, test_preds, config)

        # 保存参数及评估结果到 JSON
        results_path = f"params_{suffix}.json"
        with open(results_path, 'w') as rf:
            json.dump({
                "params": params,
                "metrics": metrics
            }, rf, indent=2)
        print(f"参数及评估结果已保存到 {results_path}")

        print(f"=== Experiment #{idx} done ===\n")

if __name__ == "__main__":
    main()