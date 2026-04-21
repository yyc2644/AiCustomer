import requests

def ocr_request():
    url = "https://test-api.zhizhi168.com/ocr-process"

    payload = {}
    files = [
        ('file', ('eaf6aff9e7034c76da13484f7c9eb74f783e724b375d9bc99c62b1f5dd8424cd.jpg', open(
            '/C:/Users/ZhanYi/PycharmProjects/AiCustomer/ocr_test/img/eaf6aff9e7034c76da13484f7c9eb74f783e724b375d9bc99c62b1f5dd8424cd.jpg',
            'rb'), 'image/jpeg'))
    ]
    headers = {}

    response = requests.request("POST", url, headers=headers, data=payload, files=files)

    print(response.text)

    return response.text