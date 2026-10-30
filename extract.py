import json
from openai import OpenAI
from config import API_KEY

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

resume = """
张宇,通信工程专业,本科在读。
-excel：数据透视表、函数统计与图表
-调研：问卷设计、描述统计、基础差异检验
-AI 应用：ChatGPT、豆包、Codex；调研提纲、内容初稿、资料整理、会议纪要
-课程研究：6G通感一体化认知与偏好调查
-带领班级获'先进班集体'，个人获'优秀学生干部''优秀团员'
-英语 CET-4 已通过。
求职意向:人才发展实习生。
"""

response = client.chat.completions.create(
    model="deepseek-chat",
    response_format={"type": "json_object"},
    messages=[
        {
            "role": "system",
            "content": (
                "你是资深技术招聘官。评估这份简历与'人才发展生'岗位的匹配度。"
                "只输出 JSON,字段:score(0-100的整数)、strengths(数组,2-3条)、"
                "weaknesses(数组,1-3条)、suggestion(一句具体可执行的改进建议)。"
            ),
        },
        {"role": "user", "content": resume},
    ],
)

result = json.loads(response.choices[0].message.content)

print(f"匹配度:{result['score']}")
print("\n优势:")
for s in result["strengths"]:
    print(" -", s)
print("\n短板:")
for w in result["weaknesses"]:
    print(" -", w)
print("\n建议:", result["suggestion"])