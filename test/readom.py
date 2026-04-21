import random
import string

def generate_upper_string():
    """生成全大写、AS开头、后6位为字母数字的8位字符串"""
    prefix = "AS"
    chars = string.ascii_uppercase + string.digits
    suffix = ''.join(random.choices(chars, k=6))
    return prefix + suffix

if __name__ == "__main__":
    # 生成10个字符串，可根据需要修改数量
    count = 100
    results = [generate_upper_string() for _ in range(count)]
    print(','.join(results))