# src/backtest.py

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class BacktestEngine:
    """
    回测引擎类，用于评估交易策略表现
    """
    
    def __init__(self, initial_capital=100000.0, commission=0.001, slippage=0.0005):
        """
        初始化回测引擎
        
        Parameters:
        -----------
        initial_capital : float
            初始资金
        commission : float
            交易佣金率（百分比）
        slippage : float
            滑点率（百分比）
        """
        self.initial_capital = initial_capital
        self.commission = commission  # 0.1% 佣金
        self.slippage = slippage  # 0.05% 滑点
        
        # 回测结果
        self.results = {}
        self.trade_log = []
        self.position_log = []
        
    def run_backtest(self, prices, predictions, signals=None, strategy_type='simple'):
        """
        运行回测
        
        Parameters:
        -----------
        prices : pd.Series or np.ndarray
            价格序列（收盘价）
        predictions : np.ndarray
            模型预测的收益率
        signals : np.ndarray, optional
            交易信号（如果为None则根据预测生成）
        strategy_type : str
            策略类型：'simple'（简单阈值）或 'momentum'（动量策略）
            
        Returns:
        --------
        dict
            回测结果
        """
        print("=" * 60)
        print("开始回测")
        print("=" * 60)
        print(f"初始资金: ${self.initial_capital:,.2f}")
        print(f"数据长度: {len(prices)}")
        print(f"策略类型: {strategy_type}")
        
        # 确保价格和预测长度一致
        if len(prices) != len(predictions):
            min_len = min(len(prices), len(predictions))
            prices = prices[:min_len]
            predictions = predictions[:min_len]
            print(f"调整后数据长度: {min_len}")
        
        # 生成交易信号
        if signals is None:
            signals = self._generate_signals(predictions, strategy_type)
        
        # 运行回测
        results = self._execute_backtest(prices, signals, predictions)
        
        # 计算性能指标
        performance_metrics = self._calculate_performance_metrics(results)
        
        # 合并结果
        self.results = {**results, **performance_metrics}
        
        print("\n回测完成！")
        print(f"最终资金: ${self.results['final_capital']:,.2f}")
        print(f"总收益率: {self.results['total_return']:.2%}")
        print(f"年化收益率: {self.results['annual_return']:.2%}")
        print(f"夏普比率: {self.results['sharpe_ratio']:.3f}")
        print(f"最大回撤: {self.results['max_drawdown']:.2%}")
        
        return self.results
    
    def _generate_signals(self, predictions, strategy_type='simple'):
        """
        根据预测生成交易信号
        
        Parameters:
        -----------
        predictions : np.ndarray
            预测收益率
        strategy_type : str
            策略类型
            
        Returns:
        --------
        np.ndarray
            交易信号：1表示买入，-1表示卖出，0表示持有
        """
        signals = np.zeros(len(predictions))
        
        if strategy_type == 'simple':
            # 简单阈值策略
            buy_threshold = 0.005  # 预测收益率大于0.5%时买入
            sell_threshold = -0.003  # 预测收益率小于-0.3%时卖出
            
            for i in range(1, len(predictions)):
                if predictions[i] > buy_threshold:
                    signals[i] = 1  # 买入
                elif predictions[i] < sell_threshold:
                    signals[i] = -1  # 卖出
                else:
                    signals[i] = 0  # 持有
        
        elif strategy_type == 'momentum':
            # 动量策略：基于预测收益率的趋势
            for i in range(2, len(predictions)):
                # 计算短期和长期动量
                short_momentum = predictions[i-2:i].mean()
                long_momentum = predictions[max(0, i-5):i].mean()
                
                # 金叉买入，死叉卖出
                if short_momentum > long_momentum and predictions[i] > 0:
                    signals[i] = 1  # 买入
                elif short_momentum < long_momentum and predictions[i] < 0:
                    signals[i] = -1  # 卖出
                else:
                    signals[i] = 0  # 持有
        
        elif strategy_type == 'ml_enhanced':
            # 机器学习增强策略：结合多个阈值和条件
            for i in range(1, len(predictions)):
                # 强买入信号：高预测收益率且持续向好
                if predictions[i] > 0.01 and (i > 1 and predictions[i] > predictions[i-1]):
                    signals[i] = 1.5  # 强买入（可以加倍仓位）
                # 普通买入信号
                elif predictions[i] > 0.005:
                    signals[i] = 1  # 买入
                # 强卖出信号
                elif predictions[i] < -0.01 and (i > 1 and predictions[i] < predictions[i-1]):
                    signals[i] = -1.5  # 强卖出
                # 普通卖出信号
                elif predictions[i] < -0.003:
                    signals[i] = -1  # 卖出
                else:
                    signals[i] = 0  # 持有
        
        print(f"交易信号统计:")
        print(f"  买入信号: {np.sum(signals > 0)} 次")
        print(f"  卖出信号: {np.sum(signals < 0)} 次")
        print(f"  持有信号: {np.sum(signals == 0)} 次")
        
        return signals
    
    def _execute_backtest(self, prices, signals, predictions):
        """
        执行回测
        
        Parameters:
        -----------
        prices : np.ndarray
            价格序列
        signals : np.ndarray
            交易信号
        predictions : np.ndarray
            预测收益率
            
        Returns:
        --------
        dict
            回测结果
        """
        # 初始化变量
        capital = self.initial_capital
        position = 0  # 持仓数量
        position_value = 0  # 持仓市值
        trade_count = 0
        win_trades = 0
        total_trades = 0
        
        # 记录数组
        capital_history = [capital]
        position_history = [position]
        returns_history = [0.0]
        drawdown_history = [0.0]
        
        # 交易记录
        self.trade_log = []
        self.position_log = []
        
        # 止损止盈参数
        stop_loss = 0.02  # 2% 止损
        take_profit = 0.05  # 5% 止盈
        entry_price = 0
        
        print("\n执行回测...")
        
        for i in range(1, len(prices)):
            current_price = prices[i]
            prev_price = prices[i-1]
            signal = signals[i]
            
            # 计算日内收益率
            daily_return = (current_price - prev_price) / prev_price if prev_price > 0 else 0
            
            # 如果有持仓，计算持仓收益
            if position > 0:
                position_return = (current_price - entry_price) / entry_price if entry_price > 0 else 0
                
                # 检查止损止盈
                if position_return <= -stop_loss:
                    signal = -1  # 触发止损
                    print(f"第{i}天: 触发止损，价格从{entry_price:.2f}跌至{current_price:.2f}")
                elif position_return >= take_profit:
                    signal = 1  # 触发止盈，可以考虑卖出或持有
                    # 这里我们选择部分止盈
                    if position_return > take_profit * 1.5:  # 如果盈利超过止盈点的1.5倍，卖出
                        signal = -1
                        print(f"第{i}天: 触发止盈，盈利{position_return:.2%}")
            
            # 执行交易
            if signal > 0 and capital > 0:  # 买入信号
                # 计算可买入数量（全部买入）
                buy_amount = capital * 0.95  # 保留5%现金
                commission_cost = buy_amount * self.commission
                slippage_cost = buy_amount * self.slippage
                total_cost = commission_cost + slippage_cost
                
                actual_buy_amount = buy_amount - total_cost
                shares_to_buy = int(actual_buy_amount / current_price)
                
                if shares_to_buy > 0:
                    # 更新持仓
                    position += shares_to_buy
                    entry_price = current_price  # 更新入场价格
                    capital -= shares_to_buy * current_price + total_cost
                    
                    # 记录交易
                    trade_record = {
                        'date': i,
                        'type': 'BUY',
                        'price': current_price,
                        'shares': shares_to_buy,
                        'value': shares_to_buy * current_price,
                        'commission': commission_cost,
                        'slippage': slippage_cost,
                        'capital_after': capital,
                        'position': position
                    }
                    self.trade_log.append(trade_record)
                    
                    trade_count += 1
                    print(f"第{i}天: 买入 {shares_to_buy} 股 @ ${current_price:.2f}")
            
            elif signal < 0 and position > 0:  # 卖出信号
                # 卖出全部持仓
                sell_value = position * current_price
                commission_cost = sell_value * self.commission
                slippage_cost = sell_value * self.slippage
                total_cost = commission_cost + slippage_cost
                
                actual_sell_value = sell_value - total_cost
                
                # 计算交易盈亏
                trade_return = (actual_sell_value - (position * entry_price)) / (position * entry_price) if entry_price > 0 else 0
                
                # 更新资本和持仓
                capital += actual_sell_value
                
                # 记录交易结果
                if trade_return > 0:
                    win_trades += 1
                
                total_trades += 1
                
                # 记录交易
                trade_record = {
                    'date': i,
                    'type': 'SELL',
                    'price': current_price,
                    'shares': position,
                    'value': sell_value,
                    'commission': commission_cost,
                    'slippage': slippage_cost,
                    'return': trade_return,
                    'capital_after': capital,
                    'position': 0
                }
                self.trade_log.append(trade_record)
                
                print(f"第{i}天: 卖出 {position} 股 @ ${current_price:.2f}, 收益率: {trade_return:.2%}")
                
                # 重置持仓
                position = 0
                entry_price = 0
            
            # 计算当前持仓价值
            position_value = position * current_price if position > 0 else 0
            
            # 计算总资产
            total_assets = capital + position_value
            
            # 计算每日收益率
            daily_portfolio_return = (total_assets - capital_history[-1]) / capital_history[-1] if capital_history[-1] > 0 else 0
            
            # 记录历史
            capital_history.append(total_assets)
            position_history.append(position)
            returns_history.append(daily_portfolio_return)
            
            # 记录持仓
            position_record = {
                'date': i,
                'price': current_price,
                'position': position,
                'position_value': position_value,
                'capital': capital,
                'total_assets': total_assets,
                'daily_return': daily_portfolio_return
            }
            self.position_log.append(position_record)
        
        # 计算最终结果
        final_capital = capital_history[-1]
        total_return = (final_capital - self.initial_capital) / self.initial_capital
        
        # 计算胜率
        win_rate = win_trades / total_trades if total_trades > 0 else 0
        
        # 准备结果字典
        results = {
            'initial_capital': self.initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'capital_history': np.array(capital_history),
            'returns_history': np.array(returns_history),
            'position_history': np.array(position_history),
            'trade_count': trade_count,
            'win_trades': win_trades,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'prices': prices,
            'signals': signals,
            'predictions': predictions
        }
        
        print(f"\n回测统计:")
        print(f"  交易次数: {trade_count}")
        print(f"  总交易: {total_trades}")
        print(f"  盈利交易: {win_trades}")
        print(f"  胜率: {win_rate:.2%}")
        
        return results
    
    def _calculate_performance_metrics(self, results):
        """
        计算性能指标
        
        Parameters:
        -----------
        results : dict
            回测结果
            
        Returns:
        --------
        dict
            性能指标
        """
        capital_history = results['capital_history']
        returns_history = results['returns_history']
        
        # 计算年化收益率（假设252个交易日）
        total_return = results['total_return']
        n_years = len(capital_history) / 252
        annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        
        # 计算年化波动率
        annual_volatility = np.std(returns_history) * np.sqrt(252)
        
        # 计算夏普比率（假设无风险利率为2%）
        risk_free_rate = 0.02
        excess_returns = returns_history - risk_free_rate/252
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0
        
        # 计算最大回撤
        peak = capital_history[0]
        max_drawdown = 0
        drawdown_history = []
        
        for value in capital_history:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            drawdown_history.append(drawdown)
            max_drawdown = max(max_drawdown, drawdown)
        
        # 计算索提诺比率（只考虑下行风险）
        downside_returns = returns_history[returns_history < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
        sortino_ratio = (annual_return - risk_free_rate) / (downside_std * np.sqrt(252)) if downside_std > 0 else 0
        
        # 计算卡尔玛比率
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
        
        # 计算beta（相对于市场的敏感性，这里简化计算）
        # 注意：这里需要市场收益率数据，我们简化处理
        market_returns = np.random.normal(0.0003, 0.01, len(returns_history))  # 模拟市场收益率
        covariance = np.cov(returns_history, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        beta = covariance / market_variance if market_variance > 0 else 1
        
        # 计算alpha
        alpha = annual_return - (risk_free_rate + beta * (np.mean(market_returns) * 252 - risk_free_rate))
        
        # 计算信息比率
        tracking_error = np.std(returns_history - market_returns) * np.sqrt(252)
        information_ratio = (annual_return - np.mean(market_returns) * 252) / tracking_error if tracking_error > 0 else 0
        
        # 计算风险调整收益
        risk_adjusted_return = total_return / max_drawdown if max_drawdown > 0 else 0
        
        # 计算交易成本影响
        total_commission = sum([trade['commission'] + trade['slippage'] for trade in self.trade_log]) if self.trade_log else 0
        cost_impact = total_commission / self.initial_capital
        
        metrics = {
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'beta': beta,
            'alpha': alpha,
            'information_ratio': information_ratio,
            'risk_adjusted_return': risk_adjusted_return,
            'cost_impact': cost_impact,
            'drawdown_history': np.array(drawdown_history)
        }
        
        return metrics
    
    def generate_report(self, model_name="策略回测报告"):
        """
        生成回测报告
        
        Parameters:
        -----------
        model_name : str
            模型名称
            
        Returns:
        --------
        dict
            报告数据
        """
        if not self.results:
            print("请先运行回测！")
            return None
        
        report = {
            'summary': {
                '模型名称': model_name,
                '初始资金': f"${self.results['initial_capital']:,.2f}",
                '最终资金': f"${self.results['final_capital']:,.2f}",
                '总收益率': f"{self.results['total_return']:.2%}",
                '年化收益率': f"{self.results['annual_return']:.2%}",
                '年化波动率': f"{self.results['annual_volatility']:.2%}",
                '夏普比率': f"{self.results['sharpe_ratio']:.3f}",
                '最大回撤': f"{self.results['max_drawdown']:.2%}",
                '索提诺比率': f"{self.results['sortino_ratio']:.3f}",
                '卡尔玛比率': f"{self.results['calmar_ratio']:.3f}",
                '交易次数': self.results['trade_count'],
                '胜率': f"{self.results['win_rate']:.2%}",
                'Alpha': f"{self.results['alpha']:.4f}",
                'Beta': f"{self.results['beta']:.3f}",
                '信息比率': f"{self.results['information_ratio']:.3f}"
            },
            'trades': self.trade_log[:10],  # 只显示前10笔交易
            'positions': self.position_log[-10:],  # 只显示最后10个持仓记录
            'results': self.results
        }
        
        return report
    
    def plot_results(self, save_path='results/figures/backtest_results.png'):
        """
        绘制回测结果图表
        
        Parameters:
        -----------
        save_path : str
            图表保存路径
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
        if not self.results:
            print("请先运行回测！")
            return
        
        # 创建图表
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # 1. 资金曲线
        axes[0, 0].plot(self.results['capital_history'], linewidth=2)
        axes[0, 0].axhline(y=self.initial_capital, color='r', linestyle='--', alpha=0.7, label='初始资金')
        axes[0, 0].set_title('资金曲线', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('交易日')
        axes[0, 0].set_ylabel('资金 ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 累计收益率
        cumulative_returns = (self.results['capital_history'] - self.initial_capital) / self.initial_capital
        axes[0, 1].plot(cumulative_returns * 100, linewidth=2, color='green')
        axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[0, 1].set_title('累计收益率', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('交易日')
        axes[0, 1].set_ylabel('收益率 (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 每日收益率分布
        axes[1, 0].hist(self.results['returns_history'] * 100, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1, 0].set_title('每日收益率分布', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('日收益率 (%)')
        axes[1, 0].set_ylabel('频率')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 回撤曲线
        axes[1, 1].fill_between(range(len(self.results['drawdown_history'])), 
                                self.results['drawdown_history'] * 100, 0,
                                alpha=0.3, color='red')
        axes[1, 1].plot(self.results['drawdown_history'] * 100, color='red', linewidth=2)
        axes[1, 1].set_title('回撤曲线', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('交易日')
        axes[1, 1].set_ylabel('回撤 (%)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 5. 持仓变化
        axes[2, 0].plot(self.results['position_history'], linewidth=2, color='purple')
        axes[2, 0].set_title('持仓变化', fontsize=14, fontweight='bold')
        axes[2, 0].set_xlabel('交易日')
        axes[2, 0].set_ylabel('持仓数量')
        axes[2, 0].grid(True, alpha=0.3)
        
        # 6. 预测收益率 vs 实际信号
        axes[2, 1].plot(self.results['predictions'][:100] * 100, label='预测收益率', alpha=0.7)
        axes[2, 1].scatter(range(100), 
                          self.results['signals'][:100] * 50,  # 放大信号便于观察
                          color='red', s=20, label='交易信号', alpha=0.5)
        axes[2, 1].set_title('预测收益率与交易信号（前100天）', fontsize=14, fontweight='bold')
        axes[2, 1].set_xlabel('交易日')
        axes[2, 1].set_ylabel('值')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"图表已保存到: {save_path}")
    
    def save_results(self, filename='results/backtest_results.csv'):
        """
        保存回测结果到CSV文件
        
        Parameters:
        -----------
        filename : str
            文件名
        """
        import pandas as pd
        
        if not self.results:
            print("请先运行回测！")
            return
        
        # 创建结果DataFrame
        results_df = pd.DataFrame({
            'capital': self.results['capital_history'],
            'daily_return': self.results['returns_history'],
            'drawdown': self.results['drawdown_history'],
            'position': self.results['position_history']
        })
        
        # 保存到CSV
        results_df.to_csv(filename, index=False)
        print(f"回测结果已保存到: {filename}")
        
        # 保存交易记录
        if self.trade_log:
            trades_df = pd.DataFrame(self.trade_log)
            trades_df.to_csv('results/trade_log.csv', index=False)
            print(f"交易记录已保存到: results/trade_log.csv")


class PortfolioBacktest(BacktestEngine):
    """
    投资组合回测引擎（多资产）
    """
    
    def __init__(self, initial_capital=100000.0, commission=0.001, slippage=0.0005):
        super().__init__(initial_capital, commission, slippage)
        self.stock_prices = None
        self.stock_predictions = None
        
    def run_portfolio_backtest(self, stock_data, stock_predictions, top_n=10, rebalance_days=20):
        """
        运行投资组合回测
        
        Parameters:
        -----------
        stock_data : dict
            股票数据，格式：{股票代码: 价格序列}
        stock_predictions : dict
            股票预测，格式：{股票代码: 预测收益率序列}
        top_n : int
            每期选择的前N只股票
        rebalance_days : int
            再平衡周期（天）
            
        Returns:
        --------
        dict
            回测结果
        """
        print("=" * 60)
        print("开始投资组合回测")
        print("=" * 60)
        print(f"初始资金: ${self.initial_capital:,.2f}")
        print(f"股票数量: {len(stock_data)}")
        print(f"每期选择前{top_n}只股票")
        print(f"再平衡周期: {rebalance_days}天")
        
        # 存储数据
        self.stock_prices = stock_data
        self.stock_predictions = stock_predictions
        
        # 获取所有股票的共同日期范围
        common_dates = self._get_common_dates(stock_data)
        
        # 初始化投资组合
        portfolio = self._initialize_portfolio(stock_data.keys())
        
        # 运行回测
        results = self._execute_portfolio_backtest(portfolio, common_dates, top_n, rebalance_days)
        
        return results
    
    def _get_common_dates(self, stock_data):
        """
        获取所有股票的共同日期范围
        """
        # 简化处理：假设所有股票数据长度相同
        first_stock = list(stock_data.keys())[0]
        return range(len(stock_data[first_stock]))
    
    def _initialize_portfolio(self, stock_symbols):
        """
        初始化投资组合
        """
        portfolio = {
            'cash': self.initial_capital,
            'positions': {symbol: 0 for symbol in stock_symbols},
            'weights': {symbol: 0 for symbol in stock_symbols}
        }
        return portfolio
    
    def _execute_portfolio_backtest(self, portfolio, dates, top_n, rebalance_days):
        """
        执行投资组合回测
        """
        capital_history = [self.initial_capital]
        returns_history = [0.0]
        
        for i, date in enumerate(dates):
            # 每rebalance_days天进行一次再平衡
            if i % rebalance_days == 0 and i > 0:
                # 选择前top_n只预测收益率最高的股票
                current_predictions = {}
                for symbol in self.stock_predictions.keys():
                    if i < len(self.stock_predictions[symbol]):
                        current_predictions[symbol] = self.stock_predictions[symbol][i]
                
                # 按预测收益率排序
                sorted_stocks = sorted(current_predictions.items(), key=lambda x: x[1], reverse=True)
                selected_stocks = sorted_stocks[:top_n]
                
                # 计算等权重
                weight = 1.0 / top_n
                
                # 更新投资组合权重
                for symbol in portfolio['weights'].keys():
                    portfolio['weights'][symbol] = weight if symbol in dict(selected_stocks) else 0
                
                # 执行再平衡
                portfolio = self._rebalance_portfolio(portfolio, i)
            
            # 计算当日投资组合价值
            portfolio_value = self._calculate_portfolio_value(portfolio, i)
            capital_history.append(portfolio_value)
            
            # 计算当日收益率
            if i > 0:
                daily_return = (portfolio_value - capital_history[-2]) / capital_history[-2]
                returns_history.append(daily_return)
        
        # 计算最终结果
        results = {
            'capital_history': np.array(capital_history),
            'returns_history': np.array(returns_history),
            'final_capital': capital_history[-1],
            'total_return': (capital_history[-1] - self.initial_capital) / self.initial_capital
        }
        
        return results
    
    def _rebalance_portfolio(self, portfolio, date_idx):
        """
        再平衡投资组合
        """
        # 简化实现：清空所有持仓，然后按新权重买入
        total_value = self._calculate_portfolio_value(portfolio, date_idx)
        
        # 卖出所有持仓
        for symbol in portfolio['positions'].keys():
            if portfolio['positions'][symbol] > 0:
                price = self.stock_prices[symbol][date_idx] if date_idx < len(self.stock_prices[symbol]) else 0
                if price > 0:
                    portfolio['cash'] += portfolio['positions'][symbol] * price
                    portfolio['positions'][symbol] = 0
        
        # 按新权重买入
        for symbol, weight in portfolio['weights'].items():
            if weight > 0:
                price = self.stock_prices[symbol][date_idx] if date_idx < len(self.stock_prices[symbol]) else 0
                if price > 0:
                    amount_to_invest = total_value * weight
                    shares_to_buy = int(amount_to_invest / price)
                    
                    if shares_to_buy > 0:
                        portfolio['positions'][symbol] = shares_to_buy
                        portfolio['cash'] -= shares_to_buy * price
        
        return portfolio
    
    def _calculate_portfolio_value(self, portfolio, date_idx):
        """
        计算投资组合价值
        """
        total_value = portfolio['cash']
        
        for symbol, shares in portfolio['positions'].items():
            if shares > 0 and date_idx < len(self.stock_prices[symbol]):
                price = self.stock_prices[symbol][date_idx]
                total_value += shares * price
        
        return total_value


def main():
    """
    主函数，演示如何使用回测引擎
    """
    print("回测引擎模块演示")
    print("=" * 60)
    
    # 生成示例数据
    np.random.seed(42)
    n_days = 252 * 2  # 2年的数据
    prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, n_days)))
    predictions = np.random.normal(0.001, 0.01, n_days)
    
    # 创建回测引擎
    engine = BacktestEngine(initial_capital=100000.0)
    
    # 运行回测
    results = engine.run_backtest(prices, predictions, strategy_type='simple')
    
    # 生成报告
    report = engine.generate_report("示例策略")
    
    # 绘制结果
    engine.plot_results()
    
    # 保存结果
    engine.save_results()
    
    return results


if __name__ == "__main__":
    results = main()