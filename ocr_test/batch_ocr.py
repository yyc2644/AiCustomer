import os
import json
import requests

# API 地址
url = "https://test-api.zhizhi168.com/ocr-process"

# 图片目录路径（请根据实际情况调整）
img_dir = r"C:\Users\ZhanYi\PycharmProjects\AiCustomer\ocr_test\img"

# 支持的图片扩展名
SUPPORTED_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')

headers = {}
payload = {}

# 遍历目录下的所有文件
for filename in os.listdir(img_dir):
    # 检查是否为图片文件
    if filename.lower().endswith(SUPPORTED_EXT):
        file_path = os.path.join(img_dir, filename)

        try:
            with open(file_path, 'rb') as f:
                files = [('file', (filename, f, 'image/jpeg'))]
                response = requests.post(url, headers=headers, data=payload, files=files)

            # 解析响应 JSON
            # try:
            #     result = response.json()
            #     is_valid = result.get('data', {}).get('is_valid')
            #     if is_valid is True:
            #         print(f"文件名: {filename} - ✅ OCR结果有效")
            #     elif is_valid is False:
            #         print(f"文件名: {filename} - ⚠️ OCR结果无效 (is_valid=false)")
            #         print(f"详细信息: {json.dumps(result, ensure_ascii=False, indent=2)}")
            #     else:
            #         # is_valid 缺失或不是布尔值
            #         print(f"文件名: {filename} - ⚠️ 响应中缺少 is_valid 字段或格式不正确")
            #         print(f"详细信息: {json.dumps(result, ensure_ascii=False, indent=2)}")
            try:
                result = response.json()
                data = result.get('data', {})

                # 提取关键字段（缺失时默认为空字符串）
                utr = data.get('utr', '')
                order_time = data.get('order_time', '')
                money = data.get('money', '')

                # 判断是否三个字段都有非空值
                if utr and order_time and money:
                    print(f"文件名: {filename} - ✅ 识别成功")
                    # 如需查看完整数据可取消下一行注释
                    # print(f"详细信息: {json.dumps(result, ensure_ascii=False, indent=2)}")
                else:
                    print(f"文件名: {filename} - ⚠️ 识别失败（字段缺失或为空）")
                    print(f"详细信息: {json.dumps(result, ensure_ascii=False, indent=2)}")
                    # 红色警告（终端支持 ANSI 颜色时显示）
                    print("\033[91m⚠️ 请检查该图片的 OCR 结果！\033[0m")

            except json.JSONDecodeError:
                print(f"文件名: {filename}")
                print(f"响应非 JSON 格式: {response.text}")
                print("⚠️ 无法解析响应，跳过有效性检查")

        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")

        print("-" * 60)  # 分隔线，便于区分不同文件的结果