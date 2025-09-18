import yfinance as yf
import os
import time

# ===== 配置部分 =====
# 要下载的股票代码列表（可以修改）
tickers = ["AAPL", "MSFT", "TSLA", "GOOG"]

# 下载的时间范围
start_date = "2020-01-01"
end_date = "2025-01-01"

# 保存路径
save_path = r"F:\\HelloWorld\\csc108\\csv"
os.makedirs(save_path, exist_ok=True)

# ===== 下载函数 =====
def download_stock(ticker, retries=3, delay=5):
    for i in range(retries):
        try:
            print(f"\n正在下载 {ticker} 的数据... 尝试 {i+1}/{retries}")
            data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
            if data.empty:
                print(f"⚠️ {ticker} 数据为空，可能是代码无效或被限流。")
                return None
            file_path = os.path.join(save_path, f"{ticker}.csv")
            data.to_csv(file_path, encoding="utf-8-sig")
            print(f"✅ 已保存 {ticker} 数据到 {file_path}")
            return file_path
        except Exception as e:
            print(f"下载 {ticker} 失败: {e}")
            if i < retries - 1:
                print(f"等待 {delay} 秒后重试...")
                time.sleep(delay)
            else:
                print(f"❌ {ticker} 下载失败，已放弃。")
    return None

# ===== 主程序 =====
if __name__ == "__main__":
    for ticker in tickers:
        download_stock(ticker)
