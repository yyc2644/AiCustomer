# import os
# os.environ['AWQ_NO_KERNELS'] = '1'  # 禁用AWQ内核警告

from tools.simple_websocket import get_assistant_text, quest
# from tools.similarity import similarity
# from tools.question import *
# from tools.googletranslate import translate_text
from tools.quest_pt import *

WEBSOCKET_URL = "wss://test-api.zhizhi168.com/infra/ws?token=a21e89debe5843f982bd97b2ed7b6393&type=0&language=pt&platform_tag=ComeIndia_CHAT&connect=1"

list1 = question_Relatar_um_problema

if __name__ == "__main__":
    for i in list1:
        payload = quest(text=i)
        modelrequest = get_assistant_text(url=WEBSOCKET_URL, payload=payload)
        # if similarity(translate_text(modelrequest)['translatedText'], translate_text(list1[-1])['translatedText']) < 0.7 :
        #     print("=" * 20,
        #           "原始问题：", list1[0], "问题翻译：", translate_text(list1[0])['translatedText'], "=" * 20,"\n",
        #           "拓展后问题：", i, "问题翻译：", translate_text(i)['translatedText'], "\n",
        #           # "模型识别语义相似度：", similarity(list1[0], i), ",\n",
        #           "模型回答：", modelrequest, "\n",
        #           "标准回答", list1[-1], "\n",
        #           "模型回答翻译后：", translate_text(modelrequest)['translatedText'], "\n",
        #           "标准回答翻译后:", translate_text(list1[-1])['translatedText'], "\n"
        #           "回答语义相似度：", similarity(translate_text(modelrequest)['translatedText'], translate_text(list1[-1])['translatedText'])
        #           )


        # print("=" * 20,
        #       "原始问题：", list1[0], "问题翻译：", translate_text(list1[0])['translatedText'], "=" * 20,"\n",
        #       "拓展后问题：", i, "问题翻译：", translate_text(i)['translatedText'], "\n",
        #       # "模型识别语义相似度：", similarity(list1[0], i), ",\n",
        #       "模型回答：", modelrequest, "\n",
        #       "标准回答", list1[-1], "\n",
        #       "模型回答翻译后：", translate_text(modelrequest)['translatedText'], "\n",
        #       "标准回答翻译后:", translate_text(list1[-1])['translatedText'], "\n"
        #       "回答语义相似度：", similarity(translate_text(modelrequest)['translatedText'], translate_text(list1[-1])['translatedText'])
        #       )
        if modelrequest != list1[-1]:
            print("问题:",i,"回答：",modelrequest)