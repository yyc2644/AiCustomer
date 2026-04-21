import json
from swift.llm import inference
from IPython.display import Latex, display, Markdown

sum, score = 0, 0
for line in open("./resources/2_7/test.jsonl"):
    # 读取测试集中的问题
    math_question = json.loads(line)
    query = math_question["messages"][1]["content"]
    # 使用基准模型推理
    response, _ = inference(llm, template, query)
    # 获取正确答案
    ans = math_question["messages"][2]["content"]
    pos = ans.find("ans")
    end_pos = ans[pos:].find('}}')
    ans = ans[pos - 2: end_pos + pos + 2]
    # 整理输出
    print(("========================================================================================"))
    print(query.split("#数学题#\n")[1])
    print("问题答案是：" + ans)
    print("-----------模型回答----------------")
    display(Latex(response))
    print("-----------回答结束----------------")
    # 计算模型得分
    if ans in response or ans[6:-2] in response:
        score += 1
        print("模型回答正确")
    else: print("模型回答错误")
    sum += 1
# 总结
display(Markdown("模型在考试中得分：**" + str(int(100*score/sum)) + "** 分"))