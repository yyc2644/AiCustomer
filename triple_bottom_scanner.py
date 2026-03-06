"""
A股三重底选股工具
识别同时满足季线三重底和月线三重底的股票
"""
import os
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 尝试导入akshare，如果失败则使用备用方案
try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False
    print("警告: akshare未安装，将使用模拟数据进行演示")

# 配置参数
CONFIG = {
    'price_tolerance': 0.05,      # 三重底价格容忍度(5%)
    'min_kline_interval': 3,       # 最低K线间隔
    'min_history_years': 3,       # 最小历史数据年份
    'output_file': 'triple_bottom_stocks.csv'
}

def get_stock_data_simulated(stock_code, years=3):
    """
    模拟股票数据（当没有akshare时使用）
    模拟一个典型的三重底形态
    """
    # 计算需要的交易日数量
    trading_days = years * 250
    
    # 创建日期索引
    end_date = datetime.now()
    dates = pd.date_range(end=end_date, periods=trading_days, freq='B')
    
    # 生成模拟数据：模拟一个三重底形态
    np.random.seed(hash(stock_code) % 10000)
    
    # 创建价格数据：经历三次探底
    base_price = 10 + np.random.random() * 40
    prices = []
    
    # 第一阶段：下跌到第一个底
    for i in range(50):
        prices.append(base_price * (1 - 0.3 * i / 50 + np.random.random() * 0.02))
    
    # 第一次反弹
    first_bottom = prices[-1]
    for i in range(30):
        prices.append(first_bottom * (1 + 0.15 * i / 30 + np.random.random() * 0.02))
    
    # 第二阶段：下跌到第二个底（与第一个底相近）
    peak1 = prices[-1]
    for i in range(40):
        prices.append(peak1 * (1 - 0.25 * i / 40 + np.random.random() * 0.02))
    
    second_bottom = prices[-1]
    # 确保第二个底与第一个底相近（在5%范围内）
    prices[-1] = first_bottom * (1 + (np.random.random() - 0.5) * 0.05)
    second_bottom = prices[-1]
    
    # 第二次反弹
    for i in range(25):
        prices.append(second_bottom * (1 + 0.12 * i / 25 + np.random.random() * 0.02))
    
    peak2 = prices[-1]
    
    # 第三阶段：下跌到第三个底（与前两个底相近）
    for i in range(35):
        prices.append(peak2 * (1 - 0.2 * i / 35 + np.random.random() * 0.02))
    
    third_bottom = prices[-1]
    # 确保第三个底与前两个底相近
    avg_bottom = (first_bottom + second_bottom) / 2
    prices[-1] = avg_bottom * (1 + (np.random.random() - 0.5) * 0.05)
    third_bottom = prices[-1]
    
    # 最后的上涨
    for i in range(trading_days - len(prices) + 20):
        prices.append(prices[-1] * (1 + 0.05 * i / 20 + np.random.random() * 0.02))
    
    # 确保数据长度正确
    prices = prices[:trading_days]
    
    # 创建DataFrame
    df = pd.DataFrame({
        '日期': dates,
        '开盘': prices,
        '收盘': [p * (1 + np.random.random() * 0.02 - 0.01) for p in prices],
        '最高': [p * (1 + np.random.random() * 0.03) for p in prices],
        '最低': [p * (1 - np.random.random() * 0.03) for p in prices],
        '成交量': [np.random.randint(1000000, 10000000) for _ in prices]
    })
    
    df.set_index('日期', inplace=True)
    return df


def get_akshare_stock_data(stock_code, years=3):
    """
    使用akshare获取A股股票数据
    """
    try:
        # 转换为akshare格式
        if stock_code.startswith('6'):
            symbol = f"{stock_code}"
        elif stock_code.startswith('0') or stock_code.startswith('3'):
            symbol = f"{stock_code}"
        else:
            return None
        
        # 获取日线数据
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", 
                                start_date=(datetime.now() - timedelta(days=years*365)).strftime('%Y%m%d'),
                                end_date=datetime.now().strftime('%Y%m%d'), adjust="qfq")
        
        if df is not None and len(df) > 0:
            df.columns = ['日期', '开盘', '收盘', '最高', '最低', '成交量', '振幅', '涨跌幅', '涨跌额', '换手率']
            df.set_index('日期', inplace=True)
            df.index = pd.to_datetime(df.index)
            return df
    except Exception as e:
        print(f"获取股票 {stock_code} 数据失败: {e}")
    
    return None


def get_stock_list():
    """
    获取A股股票列表
    """
    if AKSHARE_AVAILABLE:
        try:
            # 获取A股股票列表
            df = ak.stock_info_a_code_name()
            if df is not None:
                return df[['code', 'name']].values.tolist()
        except Exception as e:
            print(f"获取股票列表失败: {e}")
    
    # 返回模拟股票列表
    return [
        ('600519', '贵州茅台'),
        ('000858', '五粮液'),
        ('601318', '中国平安'),
        ('600036', '招商银行'),
        ('000333', '美的集团'),
        ('002475', '立讯精密'),
        ('300750', '宁德时代'),
        ('002594', '比亚迪'),
        ('600276', '恒瑞医药'),
        ('000001', '平安银行')
    ]


def resample_to_period(df, period='M'):
    """
    将日线数据转换为周期数据
    period: 'M' for Monthly, 'Q' for Quarterly
    """
    if df is None or len(df) == 0:
        return None
    
    try:
        # 转换周期
        ohlc = {
            '开盘': 'first',
            '最高': 'max',
            '最低': 'min',
            '收盘': 'last',
            '成交量': 'sum'
        }
        
        resampled = df.resample(period).agg(ohlc)
        resampled = resampled.dropna()
        
        return resampled
    except Exception as e:
        print(f"周期转换失败: {e}")
        return None


def find_local_minima(prices, window=5):
    """
    寻找局部最低点
    """
    if len(prices) < window * 2 + 1:
        return []
    
    minima = []
    for i in range(window, len(prices) - window):
        if all(prices[i] <= prices[i-j] for j in range(1, window+1)) and \
           all(prices[i] <= prices[i+j] for j in range(1, window+1)):
            minima.append(i)
    
    return minima


def identify_triple_bottom(df, tolerance=0.05, min_interval=3):
    """
    识别三重底形态
    
    参数:
    - df: 包含OHLC数据的DataFrame
    - tolerance: 价格容忍度（默认5%）
    - min_interval: 最低间隔K线数
    
    返回:
    - dict: 包含识别结果或None
    """
    if df is None or len(df) < 20:
        return None
    
    try:
        prices = df['最低'].values
        closes = df['收盘'].values
        
        # 寻找局部最低点
        minima = find_local_minima(prices, window=3)
        
        if len(minima) < 3:
            return None
        
        # 查找三个相近的低点
        for i in range(len(minima) - 2):
            for j in range(i + 1, len(minima) - 1):
                for k in range(j + 1, len(minima)):
                    low1 = prices[minima[i]]
                    low2 = prices[minima[j]]
                    low3 = prices[minima[k]]
                    
                    # 检查时间间隔
                    if (minima[j] - minima[i] < min_interval or 
                        minima[k] - minima[j] < min_interval):
                        continue
                    
                    # 检查价格差异
                    avg_low = (low1 + low2 + low3) / 3
                    max_diff = max(abs(low1 - avg_low), abs(low2 - avg_low), abs(low3 - avg_low)) / avg_low
                    
                    if max_diff > tolerance:
                        continue
                    
                    # 找到三个相近的低点，检查是否有两个反弹高点
                    # 找到最高的两个高点
                    highs = []
                    for idx in range(minima[i], minima[k] + 1):
                        highs.append((idx, df['最高'].iloc[idx]))
                    
                    if len(highs) < 4:
                        continue
                    
                    # 排序找最高点
                    highs.sort(key=lambda x: x[1], reverse=True)
                    
                    # 获取颈线（两个较高的高点）
                    neckline = (highs[0][1] + highs[1][1]) / 2
                    
                    # 当前价格位置
                    current_price = closes[-1]
                    current_idx = len(closes) - 1
                    
                    # 检查是否已经突破颈线或接近颈线
                    distance_to_neckline = (neckline - current_price) / neckline
                    
                    # 判断形态状态
                    if current_price >= neckline:
                        status = "已突破"
                    elif current_price >= neckline * 0.95:
                        status = "接近突破"
                    else:
                        status = "形成中"
                    
                    return {
                        'lows': [low1, low2, low3],
                        'neckline': neckline,
                        'current_price': current_price,
                        'distance_to_neckline': distance_to_neckline,
                        'status': status,
                        'low_indices': [minima[i], minima[j], minima[k]]
                    }
        
        return None
        
    except Exception as e:
        print(f"三重底识别错误: {e}")
        return None


def analyze_stock(stock_code, stock_name):
    """
    分析单只股票
    """
    print(f"分析股票: {stock_code} {stock_name}")
    
    # 获取数据
    if AKSHARE_AVAILABLE:
        df = get_akshare_stock_data(stock_code, CONFIG['min_history_years'])
    else:
        df = get_stock_data_simulated(stock_code, CONFIG['min_history_years'])
    
    if df is None or len(df) < 60:
        return None
    
    # 转换为月线和季线
    monthly_df = resample_to_period(df, 'M')
    quarterly_df = resample_to_period(df, 'Q')
    
    # 识别三重底
    monthly_result = identify_triple_bottom(monthly_df, 
                                              CONFIG['price_tolerance'], 
                                              CONFIG['min_kline_interval'])
    
    quarterly_result = identify_triple_bottom(quarterly_df, 
                                               CONFIG['price_tolerance'], 
                                               CONFIG['min_kline_interval'])
    
    # 返回结果
    if monthly_result and quarterly_result:
        return {
            'stock_code': stock_code,
            'stock_name': stock_name,
            'monthly': monthly_result,
            'quarterly': quarterly_result
        }
    
    return None


def main():
    """
    主函数
    """
    print("=" * 60)
    print("A股三重底选股工具")
    print("=" * 60)
    
    # 获取股票列表
    print("\n获取A股股票列表...")
    stock_list = get_stock_list()
    print(f"共获取到 {len(stock_list)} 只股票")
    
    # 分析每只股票
    results = []
    total = len(stock_list)
    
    print(f"\n开始分析股票 (共{total}只)...")
    print("-" * 60)
    
    for i, (code, name) in enumerate(stock_list):
        print(f"进度: {i+1}/{total}", end='\r')
        
        result = analyze_stock(code, name)
        if result:
            results.append(result)
    
    print("\n" + "=" * 60)
    print(f"分析完成！找到 {len(results)} 只符合三重底形态的股票")
    print("=" * 60)
    
    # 输出结果
    if results:
        print("\n符合条件的股票:")
        print("-" * 60)
        
        for result in results:
            print(f"\n股票代码: {result['stock_code']}")
            print(f"股票名称: {result['stock_name']}")
            print(f"月线三重底:")
            print(f"  - 三个低点: {result['monthly']['lows']}")
            print(f"  - 颈线: {result['monthly']['neckline']:.2f}")
            print(f"  - 当前价格: {result['monthly']['current_price']:.2f}")
            print(f"  - 状态: {result['monthly']['status']}")
            print(f"季线三重底:")
            print(f"  - 三个低点: {result['quarterly']['lows']}")
            print(f"  - 颈线: {result['quarterly']['neckline']:.2f}")
            print(f"  - 当前价格: {result['quarterly']['current_price']:.2f}")
            print(f"  - 状态: {result['quarterly']['status']}")
            print("-" * 60)
        
        # 保存到CSV
        output_data = []
        for result in results:
            output_data.append({
                '股票代码': result['stock_code'],
                '股票名称': result['stock_name'],
                '月线低点1': result['monthly']['lows'][0],
                '月线低点2': result['monthly']['lows'][1],
                '月线低点3': result['monthly']['lows'][2],
                '月线颈线': result['monthly']['neckline'],
                '月线状态': result['monthly']['status'],
                '季线低点1': result['quarterly']['lows'][0],
                '季线低点2': result['quarterly']['lows'][1],
                '季线低点3': result['quarterly']['lows'][2],
                '季线颈线': result['quarterly']['neckline'],
                '季线状态': result['quarterly']['status']
            })
        
        df_output = pd.DataFrame(output_data)
        df_output.to_csv(CONFIG['output_file'], index=False, encoding='utf-8-sig')
        print(f"\n结果已保存到: {CONFIG['output_file']}")
        
    else:
        print("\n未找到符合条件的股票")
        print("\n可能原因:")
        print("1. 当前市场没有符合三重底形态的股票")
        print("2. 数据获取失败")
        print("3. 参数设置过于严格")


if __name__ == "__main__":
    main()
