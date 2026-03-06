import os
import json
from typing import Dict, Optional

# 建议使用更加健壮的路径处理
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FILE_PATH = os.path.join(BASE_DIR, '../Key.json')


def load_api_keys(file_path: str = DEFAULT_FILE_PATH) -> Dict[str, str]:
    keys = {}

    if not os.path.exists(file_path):
        print(f"警告: API密钥文件 '{file_path}' 不存在")
        return keys

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            keys = json.load(file)
            # print("成功加载密钥文件")

        # 统一处理环境变量设置
        for key_name, value in keys.items():
            if isinstance(value, str):
                # 统一转为大写存入环境变量，确保一致性
                os.environ[key_name.upper()] = value.strip()

        print(f"已设置环境变量: {', '.join(keys.keys())}")

    except json.JSONDecodeError as e:
        print(f"错误: JSON文件格式不正确 - {e}")
    except Exception as e:
        print(f"错误: 读取API密钥文件时出错 - {e}")

    return keys


def get_api_key(provider: str, file_path: str = DEFAULT_FILE_PATH) -> Optional[str]:
    # 1. 尝试从环境变量获取 (标准化键名)
    env_var_name = f"{provider.upper()}_API_KEY"
    key_from_env = os.getenv(env_var_name)
    if key_from_env:
        return key_from_env

    # 2. 如果环境变量没有，加载文件
    keys = load_api_keys(file_path)

    # 3. 尝试多种可能的匹配方式
    possible_keys = [env_var_name, f"{provider.upper()}_API", provider.upper()]
    for k in possible_keys:
        if k in keys:
            return keys[k]

    return None


if __name__ == "__main__":
    # 加载并设置
    load_api_keys()

    # 安全地获取并打印
    deepseek_key = get_api_key("DEEPSEEK")

    # 使用 .get() 避免程序因缺少 key 而崩溃
    env_val = os.environ.get('DEEPSEEK_API_KEY', '未设置')
    env_val = os.environ.get('DASHSCOPE_API_KEY', '未设置')

    print(f"获取到的 Key: {deepseek_key[:7]}")
    print(f"环境变量中的值: {env_val[:7]}")