import json

import websocket

# --- 配置部分 ---
WEBSOCKET_URL ="wss://test-api.zhizhi168.com/infra/ws?token=a21e89debe5843f982bd97b2ed7b6393&type=0&language=pt&platform_tag=ComeIndia_CHAT&connect=1"


def quest(
    text: str,
    *,
    timestamp: str = "2025-12-24:21:16.373Z",
    message_id: str = "273e5dd1-e2e4-4f05-8527-cb258a1841e1",
    session_id: str = "94b5d3f9-5a77-32c6-69d3-8cd0ae26a10d",
) -> dict:
    """
    根据入参 text（以及可选时间/ID）构建固定格式的请求体。
    """
    return {
        "role": "user",
        "content": [
            {
                "type": "text",
                "data": {"text": text},
            }
        ],
        "timestamp": timestamp,
        "messageId": message_id,
        "sessionId": session_id,
    }


def get_assistant_text(url, payload):
    """
    发送请求并阻塞等待，直到获取到 assistant 的 text 回复后返回字符串。
    """
    ws = None
    try:
        # 1. 建立同步连接
        ws = websocket.create_connection(url, timeout=10)  # 设置10秒超时，防止死等

        # 2. 发送数据
        ws.send(json.dumps(payload))
        # print("请求已发送，正在等待回复...")

        # 3. 循环接收消息
        while True:
            result = ws.recv()  # 这里会阻塞，直到收到一条消息

            try:
                data = json.loads(result)

                # --- 核心过滤逻辑 ---
                # 检查 role 是否为 assistant
                if data.get("role") == "assistant":
                    content_list = data.get("content", [])

                    # 检查 content 列表是否存在且 type 为 text
                    if content_list and isinstance(content_list, list):
                        item = content_list[0]
                        if item.get("type") == "text":
                            # 提取目标文本
                            target_text = item.get("data", {}).get("text", "")

                            # 找到结果，关闭连接并返回
                            ws.close()
                            return target_text

            except json.JSONDecodeError:
                continue  # 忽略非 JSON 消息

    except Exception as e:
        print(f"发生错误: {e}")
        if ws:
            ws.close()
        return None


if __name__ == "__main__":
    # question =  [
    #     'ही गेम माझ्या सिस्टमवर किती वेळ लागते? प्रोसेसिंग समय कमी कशासाठी आणि कसे कमी केला जाऊ शकतो?',
    #     'प्रॉसेसिंग समय क्यों इतना अधिक लग रहा है? क्या इसकी वजह से गेम धीमा हो रहा है?',
    #     'मेरा गेम बहुत ज्यादा प्रोसेसिंग समय ले रहा है, क्या इसका कारण बता सकते हैं?',
    #     'प्रॉसेसिंग समय क्या है और मुझे इसे कैसे कम किया जा सकता है?',
    #     'मेरा गेम बहुत ज्यादा प्रोसेसिंग समय ले रहा है, क्या इसका कोई त्वरित समाधान है?',
    #     'मेरा गेम लोड होने में बहुत अधिक समय लग रहा है, क्या करूँ?',
    #     'प्रॉसेसिंग समय क्या है और इसे कैसे घटाया जा सकता है?',
    #     'मेरा गेम खेलने में प्रॉसेसिंग समय बहुत अधिक लग रहा है, इसका क्या कारण हो सकता है?',
    #     'हे, मजकुर प्रॉसेसिंग समय मध्ये काय गोंद आला आहे? मला सहाय्य करू शकता नाही?',
    #     ' प्रोसेसिंग समय क्या है और इसे कैसे घटाया जा सकता है?'
    # ]
    questions=[
        "Como posso levantar a minha criptomoeda para a minha conta bancária?",
        "Qual é o processo para sacar moeda digital numa corretora?",
        "Quanto tempo demora um levantamento de criptomoeda?",
        "Quais são as taxas associadas ao saque de moeda digital?",
        "Existe um valor mínimo para levantar criptomoeda?",
        "Posso levantar a minha moeda digital diretamente para um cartão?",
        "Quantas confirmações são necessárias para um levantamento de criptomoeda?",
        "Como faço para levantar Bitcoin (BTC) para uma carteira externa?",
        "Porque é que o meu levantamento de moeda digital está pendente há tanto tempo?",
        "É possível sacar criptomoeda em euros (EUR) ou outra moeda fiduciária?",
        "Qual é o limite diário para levantamentos de moeda digital?",
        "Como adiciono um novo endereço de carteira para sacar criptomoeda?",
        "O saque de criptomoeda para uma conta bancária é considerado uma venda para efeitos fiscais?",
        "Que documentos preciso verificar para levantar moeda digital?",
        "Posso cancelar um levantamento de criptomoeda após o ter solicitado?",
        "Há um período de espera obrigatório antes de poder levantar moeda digital depositada?",
        "Quais são os métodos de saque disponíveis para moeda digital nesta plataforma?",
        "Como posso rastrear o estado do meu levantamento de criptomoeda?",
        "O que é a 'rede' e como a escolho ao levantar criptomoeda?",
        "Porque falhou o meu pedido de saque de moeda digital?",
        "Posso levantar criptomoeda para o PayPal ou outro serviço de pagamento online?",
        "Quais são os riscos de segurança ao sacar moeda digital para uma carteira?",
        "Como funciona o levantamento instantâneo de criptomoeda?",
        "O saque de criptomoeda é anónimo ou rastreável?",
        "Para que endereço devo enviar para levantar a minha moeda digital de forma segura?"
    ]

    for i in questions:
        # print(i)
        payload = quest(text=i)
        reply_text = get_assistant_text(WEBSOCKET_URL, payload)
        # print(reply_text)
        print("问题：",i,"回答：",reply_text)

        #
        # if reply_text:
        #     print("问题：",i,"回答：",question)
        #     # print("最终获取到的文本内容:")
        #     # print(question)
        #     print("-" * 30)
        # else:
        #     print("未能获取到有效回复。")