import requests
import json
import time

def test_api():
    """测试API功能"""
    base_url = "http://localhost:5000"

    print("开始测试API...")

    # 测试获取所有策略数据
    try:
        print("1. 测试获取所有策略数据...")
        response = requests.get(f"{base_url}/api/strategies", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"   成功获取 {len(data)} 个策略的数据")
            for strategy_name, strategy_data in data.items():
                print(f"     - {strategy_name}: 夏普比率 {strategy_data['sharpe_ratio']:.4f}")
        else:
            print(f"   获取策略数据失败: {response.status_code}")
    except Exception as e:
        print(f"   API测试失败: {e}")

    # 测试策略对比
    try:
        print("\n2. 测试策略对比...")
        response = requests.get(f"{base_url}/api/compare", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("   成功获取策略对比数据")
            print(f"     - 策略数量: {len(data['names'])}")
            print(f"     - 平均夏普比率: {sum(data['sharpe_ratios'])/len(data['sharpe_ratios']):.4f}")
        else:
            print(f"   获取对比数据失败: {response.status_code}")
    except Exception as e:
        print(f"   对比API测试失败: {e}")

    print("\nAPI测试完成!")

if __name__ == "__main__":
    # 等待服务器启动
    print("等待服务器启动...")
    time.sleep(3)
    test_api()