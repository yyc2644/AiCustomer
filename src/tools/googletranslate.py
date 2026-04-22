from google.cloud import translate_v2 as translate
import os


# 设置环境变量指向你的服务账号密钥文件
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\Users\ZhanYi\PycharmProjects\AiCustomer\googlekey.json'


def translate_text(text, target_language='en', source_language=None):
    """
    使用Google Cloud Translation API翻译文本

    Args:
        text: 要翻译的文本（字符串或列表）
        target_language: 目标语言代码，默认中文
        source_language: 源语言代码（可选，自动检测）

    Returns:
        翻译结果
    """
    # 创建客户端
    translate_client = translate.Client()

    # 进行翻译
    if isinstance(text, list):
        # 批量翻译
        result = translate_client.translate(
            text,
            target_language=target_language,
            source_language=source_language
        )
    else:
        # 单个文本翻译
        result = translate_client.translate(
            text,
            target_language=target_language,
            source_language=source_language
        )

    return result


# 使用示例
if __name__ == "__main__":
    # 示例1：翻译单个文本（自动检测源语言）
    text_to_translate = "Hello, world!"
    translation = translate_text(text_to_translate, target_language='zh-CN')

    print(f"原始文本: {text_to_translate}")
    print(f"翻译结果: {translation['translatedText']}")
    print(f"检测到的源语言: {translation.get('detectedSourceLanguage', 'N/A')}")
    print("-" * 50)

    # 示例2：指定源语言
    translation2 = translate_text(
        "Bonjour le monde",
        target_language='en',
        source_language='fr'  # 指定源语言为法语
    )
    print(f"法语翻译: {translation2['translatedText']}")
    print("-" * 50)

    # 示例3：批量翻译
    texts = ["Good morning", "How are you?", "Thank you"]
    batch_result = translate_text(texts, target_language='zh-CN')

    print("批量翻译结果:")
    for i, item in enumerate(batch_result):
        print(f"{texts[i]} -> {item['translatedText']}")