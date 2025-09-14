#!/usr/bin/env python3
"""
量化交易策略对比分析平台 - 快速启动器
从项目根目录直接启动webapp
"""

import os
import subprocess
import sys
import time
import webbrowser
from threading import Timer

def open_browser():
    """延迟打开浏览器"""
    time.sleep(3)  # 等待服务器启动
    webbrowser.open('http://localhost:5000')

def main():
    print("=" * 60)
    print("量化交易策略对比分析平台")
    print("=" * 60)
    print()
    print("正在启动策略分析网页...")
    print("文件位置: webapp/")
    print("访问地址: http://localhost:5000")
    print()
    print("功能特点:")
    print("- 6种交易策略实时对比")
    print("- 多维度性能分析图表")
    print("- 交互式策略选择器")
    print("- 数据导出功能")
    print()
    print("按 Ctrl+C 停止服务器")
    print("-" * 60)

    # 切换到webapp目录
    webapp_dir = os.path.join(os.path.dirname(__file__), 'webapp')

    if not os.path.exists(webapp_dir):
        print(f"错误: 找不到webapp目录: {webapp_dir}")
        return

    # 在新线程中打开浏览器
    Timer(2, open_browser).start()

    try:
        # 切换到webapp目录并启动应用
        os.chdir(webapp_dir)
        print("切换到webapp目录")
        print("启动Flask服务器...")

        # 启动Flask应用
        subprocess.run([sys.executable, 'app.py'])

    except KeyboardInterrupt:
        print("\n\n服务器已停止")
    except Exception as e:
        print(f"\n启动失败: {e}")

if __name__ == "__main__":
    main()