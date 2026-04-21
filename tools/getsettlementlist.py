import requests
import json
import os
from urllib.parse import urlparse

def get_img_urls():
    err_url = ("https://ufadgghjkpvbnmtyuifghiu.newt1chatadmf4g3osbkdb8s20ndksfk2ak34.com/admin-api/system/work-order/page?"
           "pageNo=1&pageSize=100&createTimeType=1&createTime%5B0%5D=1774627200000&createTime%5B1%5D=1774886399999&orderStatusInt=&orderType=1&"
           "status=2&operateCounts%5B0%5D=&operateCounts%5B1%5D=&orderAmount%5B0%5D=&orderAmount%5B1%5D=&localTimezone=Asia%2FShanghai&timezone=Asia%2FShanghai")
    people_success_url = ("https://ufadgghjkpvbnmtyuifghiu.newt1chatadmf4g3osbkdb8s20ndksfk2ak34.com/admin-api/system/work-order/page?pageNo=1&pageSize=100&createTimeType=1&createTime%5B0%5D=1774627200000&createTime%5B1%5D=1774886399999&orderStatusInt=&orderType=1&status=1&operateCounts%5B0%5D=1&operateCounts%5B1%5D=10&orderAmount%5B0%5D=&orderAmount%5B1%5D=&localTimezone=Asia%2FShanghai&timezone=Asia%2FShanghai")
    success_url = ("https://ufadgghjkpvbnmtyuifghiu.newt1chatadmf4g3osbkdb8s20ndksfk2ak34.com/admin-api/system/work-order/page?pageNo=1&pageSize=100&createTimeType=1&createTime%5B0%5D=1774627200000&createTime%5B1%5D=1774886399999&orderStatusInt=&orderType=1&status=1&operateCounts%5B0%5D=1&operateCounts%5B1%5D=10&orderAmount%5B0%5D=&orderAmount%5B1%5D=&localTimezone=Asia%2FShanghai&timezone=Asia%2FShanghai")
    url = success_url
    payload = {}
    headers = {
        'accept': 'application/json, text/plain, */*',
        'accept-language': 'zh-CN,zh;q=0.9',
        'authorization': 'Bearer 2f6bb52e28594841b464ed5a04393466',
        'cache-control': 'no-cache',
        'pragma': 'no-cache',
        'priority': 'u=1, i',
        'referer': 'https://ufadgghjkpvbnmtyuifghiu.newt1chatadmf4g3osbkdb8s20ndksfk2ak34.com/',
        'sec-ch-ua': '"Chromium";v="146", "Not-A.Brand";v="24", "Google Chrome";v="146"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36'
    }

    response = requests.request("GET", url, headers=headers, data=payload)
    # print(response.text)
    data = json.loads(response.text)

    image_urls = [item['voucherImage'] for item in data['data']['list'] if item.get('voucherImage')]
    return image_urls
    # for url in image_urls:
    #     print(url)


def download_images(urls, save_dir=r'C:\Users\ZhanYi\PycharmProjects\AiCustomer\ocr_test\success_img', timeout=10):
    """
    批量下载图片
    :param urls: 图片URL列表
    :param save_dir: 保存目录，默认 'images'
    :param timeout: 请求超时时间（秒）
    """
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for idx, url in enumerate(urls, start=1):
        try:
            # 发送请求
            response = requests.get(url, timeout=timeout, stream=True)
            response.raise_for_status()  # 检查请求是否成功

            # 从URL或Content-Type获取文件扩展名
            content_type = response.headers.get('content-type', '')
            if 'image' not in content_type:
                print(f'跳过非图片URL: {url}')
                continue

            # 尝试从URL获取扩展名
            parsed = urlparse(url)
            ext = os.path.splitext(parsed.path)[1]
            if not ext or ext.lower() not in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
                # 默认使用.jpg
                ext = '.jpg'

            filename = f'image_{idx}{ext}'
            filepath = os.path.join(save_dir, filename)

            # 写入文件
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f'下载成功: {filename}')

        except Exception as e:
            print(f'下载失败 {url}: {e}')

if __name__ == '__main__':
    download_images(get_img_urls())