import subprocess
import sys
import time
import webbrowser
from threading import Timer

def open_browser():
    """延迟打开浏览器"""
    time.sleep(2)  # 等待服务器启动
    webbrowser.open('http://localhost:5000')

def main():
    print("=" * 60)
    print("量化交易策略对比分析平台")
    print("=" * 60)
    print()
    print("功能特点:")
    print("- 6种交易策略实时对比")
    print("- 多维度性能分析图表")
    print("- 交互式策略选择器")
    print("- 数据导出功能")
    print()
    print("正在启动服务器...")

    # 在新线程中打开浏览器
    Timer(1, open_browser).start()

    try:
        # 启动Flask应用
        subprocess.run([sys.executable, 'app.py'])
    except KeyboardInterrupt:
        print("\n服务器已停止")

if __name__ == "__main__":
    main()