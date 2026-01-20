import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子以确保可重复性
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class LSTMModel(nn.Module):
    """
    LSTM模型用于时间序列预测
    """
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2, output_size=1):
        """
        初始化LSTM模型
        
        Parameters:
        -----------
        input_size : int
            输入特征维度
        hidden_size : int
            LSTM隐藏层维度
        num_layers : int
            LSTM层数
        dropout : float
            Dropout概率
        output_size : int
            输出维度（预测目标数量）
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 全连接输出层
        self.fc = nn.Linear(hidden_size, output_size)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """
        初始化模型权重
        """
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0)
    
    def forward(self, x):
        """
        前向传播
        
        Parameters:
        -----------
        x : torch.Tensor
            输入张量，形状为 (batch_size, seq_length, input_size)
            
        Returns:
        --------
        torch.Tensor
            输出张量，形状为 (batch_size, output_size)
        """
        batch_size = x.size(0)
        
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # LSTM前向传播
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # 取最后一个时间步的输出
        lstm_out = lstm_out[:, -1, :]
        
        # Dropout
        lstm_out = self.dropout(lstm_out)
        
        # 全连接层
        output = self.fc(lstm_out)
        
        return output
    
    def predict(self, x):
        """
        预测方法（推理模式）
        
        Parameters:
        -----------
        x : torch.Tensor
            输入张量
            
        Returns:
        --------
        numpy.ndarray
            预测结果
        """
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)
            predictions = self.forward(x)
        return predictions.numpy()


class PositionalEncoding(nn.Module):
    """
    Transformer的位置编码
    """
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerModel(nn.Module):
    """
    Transformer模型用于时间序列预测
    """
    
    def __init__(self, input_size, d_model=256, nhead=4, num_layers=4, 
                 dim_feedforward=512, dropout=0.1, output_size=1, max_seq_len=5000):
        """
        初始化Transformer模型
        
        Parameters:
        -----------
        input_size : int
            输入特征维度
        d_model : int
            Transformer模型的维度
        nhead : int
            注意力头的数量
        num_layers : int
            Transformer编码器层数
        dim_feedforward : int
            前馈网络维度
        dropout : float
            Dropout概率
        output_size : int
            输出维度
        max_seq_len : int
            最大序列长度（用于位置编码）
        """
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        
        # 输入投影层（将输入特征映射到d_model维度）
        self.input_projection = nn.Linear(input_size, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 输出层
        self.output_layer = nn.Linear(d_model, output_size)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """
        初始化模型权重
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        """
        前向传播
        
        Parameters:
        -----------
        x : torch.Tensor
            输入张量，形状为 (batch_size, seq_length, input_size)
            
        Returns:
        --------
        torch.Tensor
            输出张量，形状为 (batch_size, output_size)
        """
        # 输入投影
        x = self.input_projection(x) * np.sqrt(self.d_model)
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码器
        transformer_out = self.transformer_encoder(x)
        
        # 取最后一个时间步
        transformer_out = transformer_out[:, -1, :]
        
        # Dropout
        transformer_out = self.dropout(transformer_out)
        
        # 输出层
        output = self.output_layer(transformer_out)
        
        return output
    
    def predict(self, x):
        """
        预测方法（推理模式）
        
        Parameters:
        -----------
        x : torch.Tensor
            输入张量
            
        Returns:
        --------
        numpy.ndarray
            预测结果
        """
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)
            predictions = self.forward(x)
        return predictions.numpy()


class ModelTrainer:
    """
    模型训练器，封装训练和评估逻辑
    """
    
    def __init__(self, model, model_name, device=None):
        """
        初始化模型训练器
        
        Parameters:
        -----------
        model : nn.Module
            PyTorch模型
        model_name : str
            模型名称（用于保存和记录）
        device : torch.device
            训练设备（CPU或GPU）
        """
        self.model = model
        self.model_name = model_name
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model.to(self.device)
        
        # 训练历史记录
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        # 创建结果目录
        import os
        os.makedirs('results/figures', exist_ok=True)
        os.makedirs('results/models', exist_ok=True)
    
    def train(self, train_loader, val_loader, criterion, optimizer, 
              num_epochs=100, patience=10, lr_scheduler=None):
        """
        训练模型
        
        Parameters:
        -----------
        train_loader : DataLoader
            训练数据加载器
        val_loader : DataLoader
            验证数据加载器
        criterion : nn.Module
            损失函数
        optimizer : torch.optim.Optimizer
            优化器
        num_epochs : int
            训练轮数
        patience : int
            早停耐心值
        lr_scheduler : torch.optim.lr_scheduler
            学习率调度器
            
        Returns:
        --------
        dict
            训练历史
        """
        print(f"开始训练 {self.model_name} 模型...")
        print(f"使用设备: {self.device}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_predictions = []
            train_targets = []
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                # 前向传播
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y.unsqueeze(1))
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪（防止梯度爆炸）
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item() * batch_x.size(0)
                train_predictions.extend(outputs.cpu().detach().numpy())
                train_targets.extend(batch_y.cpu().numpy())
            
            train_loss = train_loss / len(train_loader.dataset)
            
            # 验证阶段
            val_loss, val_metrics = self.evaluate(val_loader, criterion)
            
            # 计算训练指标
            train_predictions = np.array(train_predictions).flatten()
            train_targets = np.array(train_targets)
            train_metrics = self._calculate_metrics(train_targets, train_predictions)
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_metrics'].append(train_metrics)
            self.history['val_metrics'].append(val_metrics)
            
            # 学习率调度
            if lr_scheduler is not None:
                lr_scheduler.step(val_loss)
            
            # 早停和模型保存
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_model(f"results/models/{self.model_name}_best.pth")
                print(f"Epoch {epoch+1}: 发现更好的模型，验证损失: {val_loss:.6f}")
            else:
                patience_counter += 1
            
            # 打印训练信息
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{num_epochs}:")
                print(f"  训练损失: {train_loss:.6f}, 验证损失: {val_loss:.6f}")
                print(f"  训练RMSE: {train_metrics['rmse']:.6f}, 验证RMSE: {val_metrics['rmse']:.6f}")
                print(f"  早停计数器: {patience_counter}/{patience}")
            
            # 检查早停
            if patience_counter >= patience:
                print(f"早停触发！在 {epoch+1} 轮停止训练")
                break
        
        # 加载最佳模型
        self._load_model(f"results/models/{self.model_name}_best.pth")
        
        print(f"训练完成！最佳验证损失: {best_val_loss:.6f}")
        
        return self.history
    
    def evaluate(self, data_loader, criterion):
        """
        评估模型
        
        Parameters:
        -----------
        data_loader : DataLoader
            数据加载器
        criterion : nn.Module
            损失函数
            
        Returns:
        --------
        tuple
            (平均损失, 评估指标字典)
        """
        self.model.eval()
        total_loss = 0.0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y.unsqueeze(1))
                
                total_loss += loss.item() * batch_x.size(0)
                predictions.extend(outputs.cpu().numpy())
                targets.extend(batch_y.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader.dataset)
        predictions = np.array(predictions).flatten()
        targets = np.array(targets)
        
        metrics = self._calculate_metrics(targets, predictions)
        
        return avg_loss, metrics
    
    def _calculate_metrics(self, y_true, y_pred):
        """
        计算评估指标
        
        Parameters:
        -----------
        y_true : np.ndarray
            真实值
        y_pred : np.ndarray
            预测值
            
        Returns:
        --------
        dict
            评估指标字典
        """
        # 确保数组形状一致
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
        }
        
        return metrics
    
    def _save_model(self, path):
        """
        保存模型
        
        Parameters:
        -----------
        path : str
            保存路径
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'history': self.history
        }, path)
    
    def _load_model(self, path):
        """
        加载模型
        
        Parameters:
        -----------
        path : str
            模型路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"已加载模型: {path}")
    
    def predict(self, X):
        """
        批量预测
        
        Parameters:
        -----------
        X : np.ndarray
            输入数据
            
        Returns:
        --------
        np.ndarray
            预测结果
        """
        self.model.eval()
        
        # 转换为PyTorch张量
        if isinstance(X, np.ndarray):
            X_tensor = torch.FloatTensor(X)
        else:
            X_tensor = X
        
        # 创建数据加载器
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=1024, shuffle=False)
        
        predictions = []
        
        with torch.no_grad():
            for batch_x in dataloader:
                batch_x = batch_x[0].to(self.device)
                batch_pred = self.model(batch_x)
                predictions.append(batch_pred.cpu().numpy())
        
        predictions = np.concatenate(predictions, axis=0).flatten()
        
        return predictions
    
    def plot_training_history(self):
        """
        绘制训练历史图表
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        axes[0, 0].plot(self.history['train_loss'], label='训练损失')
        axes[0, 0].plot(self.history['val_loss'], label='验证损失')
        axes[0, 0].set_xlabel('轮次')
        axes[0, 0].set_ylabel('损失')
        axes[0, 0].set_title(f'{self.model_name} - 损失曲线')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # RMSE曲线
        train_rmse = [m['rmse'] for m in self.history['train_metrics']]
        val_rmse = [m['rmse'] for m in self.history['val_metrics']]
        
        axes[0, 1].plot(train_rmse, label='训练RMSE')
        axes[0, 1].plot(val_rmse, label='验证RMSE')
        axes[0, 1].set_xlabel('轮次')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].set_title(f'{self.model_name} - RMSE曲线')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # R²曲线
        train_r2 = [m['r2'] for m in self.history['train_metrics']]
        val_r2 = [m['r2'] for m in self.history['val_metrics']]
        
        axes[1, 0].plot(train_r2, label='训练R²')
        axes[1, 0].plot(val_r2, label='验证R²')
        axes[1, 0].set_xlabel('轮次')
        axes[1, 0].set_ylabel('R²得分')
        axes[1, 0].set_title(f'{self.model_name} - R²曲线')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # MAE曲线
        train_mae = [m['mae'] for m in self.history['train_metrics']]
        val_mae = [m['mae'] for m in self.history['val_metrics']]
        
        axes[1, 1].plot(train_mae, label='训练MAE')
        axes[1, 1].plot(val_mae, label='验证MAE')
        axes[1, 1].set_xlabel('轮次')
        axes[1, 1].set_ylabel('MAE')
        axes[1, 1].set_title(f'{self.model_name} - MAE曲线')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'results/figures/{self.model_name}_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_predictions(self, y_true, y_pred, title_suffix=""):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 预测vs真实散点图
        axes[0, 0].scatter(y_true, y_pred, alpha=0.5, s=10)
        axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                        'r--', lw=2, label='完美预测线')
        axes[0, 0].set_xlabel('真实值')
        axes[0, 0].set_ylabel('预测值')
        axes[0, 0].set_title(f'{self.model_name} - 预测vs真实{title_suffix}')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 残差图
        residuals = y_true - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.5, s=10)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('预测值')
        axes[0, 1].set_ylabel('残差')
        axes[0, 1].set_title(f'{self.model_name} - 残差图{title_suffix}')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 预测值和真实值分布
        axes[1, 0].hist(y_true, bins=50, alpha=0.5, label='真实值', density=True)
        axes[1, 0].hist(y_pred, bins=50, alpha=0.5, label='预测值', density=True)
        axes[1, 0].set_xlabel('值')
        axes[1, 0].set_ylabel('密度')
        axes[1, 0].set_title(f'{self.model_name} - 分布对比{title_suffix}')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 残差分布
        axes[1, 1].hist(residuals, bins=50, alpha=0.7, color='purple', density=True)
        axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('残差')
        axes[1, 1].set_ylabel('密度')
        axes[1, 1].set_title(f'{self.model_name} - 残差分布{title_suffix}')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'results/figures/{self.model_name}_predictions{title_suffix}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()


def create_data_loaders(X_train, y_train, X_val, y_val, batch_size=64):
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    # 创建数据集
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def train_lstm_model(X_train, y_train, X_val, y_val, input_size, **kwargs):
    # 默认参数
    params = {
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.2,
        'batch_size': 64,
        'learning_rate': 0.001,
        'num_epochs': 100,
        'patience': 15
    }
    params.update(kwargs)
    
    # 创建数据加载器
    train_loader, val_loader = create_data_loaders(
        X_train, y_train, X_val, y_val, 
        batch_size=params['batch_size']
    )
    
    # 创建模型
    model = LSTMModel(
        input_size=input_size,
        hidden_size=params['hidden_size'],
        num_layers=params['num_layers'],
        dropout=params['dropout']
    )
    
    # 创建训练器
    trainer = ModelTrainer(model, "LSTM")
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 训练模型
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=params['num_epochs'],
        patience=params['patience'],
        lr_scheduler=scheduler
    )
    
    return model, trainer, history


def train_transformer_model(X_train, y_train, X_val, y_val, input_size, **kwargs):
    
    # 默认参数
    params = {
        'd_model': 256,
        'nhead': 4,
        'num_layers': 4,
        'dim_feedforward': 512,
        'dropout': 0.1,
        'batch_size': 32,
        'learning_rate': 0.0005,
        'num_epochs': 100,
        'patience': 15
    }
    params.update(kwargs)
    
    # 创建数据加载器
    train_loader, val_loader = create_data_loaders(
        X_train, y_train, X_val, y_val, 
        batch_size=params['batch_size']
    )
    
    # 创建模型
    model = TransformerModel(
        input_size=input_size,
        d_model=params['d_model'],
        nhead=params['nhead'],
        num_layers=params['num_layers'],
        dim_feedforward=params['dim_feedforward'],
        dropout=params['dropout']
    )
    
    # 创建训练器
    trainer = ModelTrainer(model, "Transformer")
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 训练模型
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=params['num_epochs'],
        patience=params['patience'],
        lr_scheduler=scheduler
    )
    
    return model, trainer, history


def compare_models(y_true, predictions_dict):
    import pandas as pd
    
    results = []
    
    for model_name, y_pred in predictions_dict.items():
        metrics = {
            '模型': model_name,
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R²': r2_score(y_true, y_pred)
        }
        results.append(metrics)
    
    df = pd.DataFrame(results)
    df = df.sort_values('RMSE')
    
    return df


def main():
    """
    主函数，演示如何使用模型模块
    """
    print("=" * 60)
    print("模型模块演示")
    print("=" * 60)
    
    # 这里只是一个演示，实际使用时需要先加载预处理好的数据
    print("注意：此演示需要先运行数据预处理模块获取数据")
    print("请先运行 test_preprocessing.py 或 notebooks/Modeling.ipynb")


if __name__ == "__main__":
    main()