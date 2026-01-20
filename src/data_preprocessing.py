# src/data_preprocessing.py

import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self, data_path='data/raw/all_stocks_5yr.csv'):
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None
        self.features = None
    def load_data(self):
        print(f"正在加载数据: {self.data_path}")
        
        try:
            self.raw_data = pd.read_csv(self.data_path)
            print(f"数据加载成功！形状: {self.raw_data.shape}")
            print(f"列名: {list(self.raw_data.columns)}")
            return self.raw_data
        except FileNotFoundError:
            print(f"文件未找到: {self.data_path}")
            print("请确保已下载数据并放置在正确路径")
            return None
    
    def explore_data(self):
        if self.raw_data is None:
            print("请先加载数据！")
            return
            
        print("=" * 50)
        print("数据基本信息:")
        print("=" * 50)
        print(f"数据形状: {self.raw_data.shape}")
        print(f"\n前5行数据:")
        print(self.raw_data.head())
        
        print(f"\n数据类型:")
        print(self.raw_data.dtypes)
        
        print(f"\n缺失值统计:")
        print(self.raw_data.isnull().sum())
        
        print(f"\n数据描述统计:")
        print(self.raw_data.describe())
        
        # 查看唯一股票数量
        unique_stocks = self.raw_data['Name'].nunique()
        print(f"\n唯一股票数量: {unique_stocks}")
        
        # 查看日期范围
        min_date = self.raw_data['date'].min()
        max_date = self.raw_data['date'].max()
        print(f"日期范围: {min_date} 到 {max_date}")
    
    def clean_data(self):
        if self.raw_data is None:
            print("请先加载数据！")
            return None
            
        print("开始数据清洗...")
        data = self.raw_data.copy()
        
        # 1. 转换日期格式
        data['date'] = pd.to_datetime(data['date'])
        
        # 2. 按日期和股票排序
        data = data.sort_values(['Name', 'date']).reset_index(drop=True)
        
        # 3. 检查并处理缺失值
        missing_before = data.isnull().sum().sum()
        print(f"清洗前缺失值总数: {missing_before}")
        
        # 对于价格数据，使用前向填充（同一股票的前一天价格）
        price_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in price_columns:
            data[col] = data.groupby('Name')[col].ffill()
            data[col] = data.groupby('Name')[col].bfill()  # 如果开头有缺失，使用后向填充
        
        missing_after = data.isnull().sum().sum()
        print(f"清洗后缺失值总数: {missing_after}")
        
        # 4. 删除仍有缺失值的行
        if missing_after > 0:
            data = data.dropna()
            print(f"删除包含缺失值的行后，数据形状: {data.shape}")
        
        # 5. 添加基础特征
        data = self._add_basic_features(data)
        
        self.processed_data = data
        print("数据清洗完成！")
        return data
    
    def _add_basic_features(self, data):
        print("添加基础特征...")
        
        # 为每组股票单独计算特征
        result_dfs = []
        
        for name, group in data.groupby('Name'):
            group = group.copy().sort_values('date')
            
            # 1. 日收益率
            group['daily_return'] = group['close'].pct_change()
            
            # 2. 对数收益率（更符合正态分布）
            group['log_return'] = np.log(group['close'] / group['close'].shift(1))
            
            # 3. 简单移动平均
            group['sma_5'] = group['close'].rolling(window=5).mean()
            group['sma_10'] = group['close'].rolling(window=10).mean()
            group['sma_20'] = group['close'].rolling(window=20).mean()
            
            # 4. 指数移动平均
            group['ema_5'] = group['close'].ewm(span=5, adjust=False).mean()
            group['ema_10'] = group['close'].ewm(span=10, adjust=False).mean()
            
            # 5. 布林带
            sma_20 = group['close'].rolling(window=20).mean()
            std_20 = group['close'].rolling(window=20).std()
            group['bb_upper'] = sma_20 + 2 * std_20
            group['bb_lower'] = sma_20 - 2 * std_20
            group['bb_width'] = (group['bb_upper'] - group['bb_lower']) / sma_20
            
            # 6. 相对强弱指数 (RSI)
            delta = group['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            group['rsi'] = 100 - (100 / (1 + rs))
            
            # 7. 移动平均收敛发散 (MACD)
            exp1 = group['close'].ewm(span=12, adjust=False).mean()
            exp2 = group['close'].ewm(span=26, adjust=False).mean()
            group['macd'] = exp1 - exp2
            group['macd_signal'] = group['macd'].ewm(span=9, adjust=False).mean()
            
            # 8. 价格变化
            group['price_change'] = group['close'] - group['close'].shift(1)
            group['price_change_pct'] = group['price_change'] / group['close'].shift(1) * 100
            
            # 9. 成交量相关特征
            group['volume_sma'] = group['volume'].rolling(window=5).mean()
            group['volume_ratio'] = group['volume'] / group['volume_sma']
            
            # 10. 波动率
            group['volatility'] = group['log_return'].rolling(window=20).std() * np.sqrt(252)  # 年化波动率
            
            result_dfs.append(group)
        
        result = pd.concat(result_dfs, ignore_index=True)
        
        # 删除因计算滚动窗口产生的缺失值
        result = result.dropna()
        
        print(f"添加特征后，数据形状: {result.shape}")
        print(f"特征数量: {len(result.columns)}")
        
        return result
    
    def prepare_features_target(self, target_column='daily_return', lookback=60, prediction_horizon=1):
        if self.processed_data is None:
            print("请先清洗数据！")
            return None, None, None
        
        print(f"准备特征和目标变量，lookback={lookback}, horizon={prediction_horizon}")
        
        # 选择特征列（排除非数值列和日期列）
        exclude_cols = ['date', 'Name']
        feature_cols = [col for col in self.processed_data.columns 
                       if col not in exclude_cols and col != target_column]
        
        # 只选择部分重要特征
        selected_features = [
            'open', 'high', 'low', 'close', 'volume',
            'daily_return', 'log_return',
            'sma_5', 'sma_10', 'sma_20',
            'bb_upper', 'bb_lower', 'bb_width',
            'rsi', 'macd', 'macd_signal',
            'volatility'
        ]
        
        # 确保选择的特征存在于数据中
        selected_features = [col for col in selected_features if col in feature_cols]
        
        print(f"选择的特征: {selected_features}")
        
        # 为每只股票创建时间序列窗口
        X_list, y_list = [], []
        
        for name, group in self.processed_data.groupby('Name'):
            group = group.sort_values('date')
            
            # 获取特征矩阵
            feature_matrix = group[selected_features].values
            
            # 创建时间序列窗口
            for i in range(len(group) - lookback - prediction_horizon):
                X_window = feature_matrix[i:i+lookback]
                y_window = group[target_column].iloc[i+lookback:i+lookback+prediction_horizon].values
                
                X_list.append(X_window)
                y_list.append(y_window[-1])  # 只取最后一个预测值
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        print(f"创建的时间序列窗口形状: X={X.shape}, y={y.shape}")
        
        self.features = selected_features
        return X, y, selected_features
    
    def save_processed_data(self, output_path='data/processed/processed_data.csv'):
        if self.processed_data is None:
            print("没有处理后的数据可保存！")
            return
            
        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        self.processed_data.to_csv(output_path, index=False)
        print(f"处理后的数据已保存到: {output_path}")
    
    def split_train_test(self, X, y, test_size=0.2, random_state=42):
        # 按时间顺序划分（不打乱时间顺序）
        split_idx = int(len(X) * (1 - test_size))
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"训练集形状: X_train={X_train.shape}, y_train={y_train.shape}")
        print(f"测试集形状: X_test={X_test.shape}, y_test={y_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def normalize_features(self, X_train, X_test):
        from sklearn.preprocessing import StandardScaler
        
        # 获取原始形状
        orig_shape_train = X_train.shape
        orig_shape_test = X_test.shape
        
        # 重塑为2D数组进行标准化
        X_train_2d = X_train.reshape(-1, X_train.shape[-1])
        X_test_2d = X_test.reshape(-1, X_test.shape[-1])
        
        # 创建并拟合标准化器
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_2d)
        X_test_scaled = scaler.transform(X_test_2d)
        
        # 重塑回原始形状
        X_train_scaled = X_train_scaled.reshape(orig_shape_train)
        X_test_scaled = X_test_scaled.reshape(orig_shape_test)
        
        print("特征标准化完成！")
        
        return X_train_scaled, X_test_scaled, scaler


def main():
    # 初始化预处理器
    preprocessor = DataPreprocessor()
    
    # 1. 加载数据
    raw_data = preprocessor.load_data()
    
    if raw_data is None:
        print("数据加载失败，退出程序。")
        return
    
    # 2. 探索数据
    preprocessor.explore_data()
    
    # 3. 清洗数据并添加特征
    processed_data = preprocessor.clean_data()
    
    # 4. 保存处理后的数据
    preprocessor.save_processed_data()
    
    # 5. 准备特征和目标变量
    X, y, feature_names = preprocessor.prepare_features_target(
        target_column='daily_return',
        lookback=60,
        prediction_horizon=1
    )
    
    if X is not None and y is not None:
        # 6. 划分训练集和测试集
        X_train, X_test, y_train, y_test = preprocessor.split_train_test(X, y, test_size=0.2)
        
        # 7. 标准化特征
        X_train_scaled, X_test_scaled, scaler = preprocessor.normalize_features(X_train, X_test)
        
        print("\n数据处理流程完成！")
        print(f"最终数据形状:")
        print(f"  X_train_scaled: {X_train_scaled.shape}")
        print(f"  X_test_scaled: {X_test_scaled.shape}")
        print(f"  y_train: {y_train.shape}")
        print(f"  y_test: {y_test.shape}")


if __name__ == "__main__":
    main()