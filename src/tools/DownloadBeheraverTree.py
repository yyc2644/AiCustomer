import requests
import os
from datetime import datetime

import argparse
EXP_path  =  r"C:\Users\ZhanYi\PycharmProjects\AiCustomer\BehaviorTree\base"

def export_behavior_tree(platform: str, language: str, authorization: str, save_path: str = "behavior_tree.xlsx",output:str = ""):
    """
    导出系统配置为 Excel 文件。

    参数:
        platform (str): 平台名称，比如 "ComeIndia"
        language (str): 语言，比如 "en"
        authorization (str): Bearer Token 授权
        output (str): Output excel file name, e.g. config.xlsx

    返回:
        bool: 导出成功返回 True，失败返回 False
    """
    url = f"https://test-api.zhizhi168.com/admin-api/system/config/export?platform={platform}&language={language}"

    headers = {
        'accept': 'application/json, text/plain, */*',
        'accept-language': 'zh-CN,zh;q=0.9',
        'authorization': f'Bearer {authorization}',
        'cache-control': 'no-cache',
        'pragma': 'no-cache',
        'priority': 'u=1, i',
        'referer': 'https://test-api.zhizhi168.com/',
        'sec-ch-ua': '"Google Chrome";v="143", "Chromium";v="143", "Not A(Brand";v="24"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36'
    }
    print("Requesting:", url)

    try:
        resp  = requests.get(url, headers=headers, timeout=30)
        resp .raise_for_status()  # 如果请求失败，会抛出异常

        # 保存文件
        os.makedirs(EXP_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.basename(output)
        final_path = os.path.join(EXP_path, f"{timestamp}_{filename}")

        with open(final_path, "wb") as f:
            f.write(resp.content)

        print(f"行为树Excel 文件已成功导出到: {output}")
        return True

    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        return False

# 示例调用
if __name__ == "__main__":
    export_behavior_tree(
        platform="ComeIndia",
        language="en",
        authorization="029588a5ca424735831ff0068dd36248",
        output="BehaviorTree/base/jsonl/config_export.xlsx"
    )
