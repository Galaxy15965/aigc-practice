from openai import OpenAI
from config import API_KEY

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

r = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "把用户的问题改写成3个不同说法。要求:每句至少换掉2个关键词,可以用同义词、上位词,或从相反角度提问。只输出3行,不要编号,不要解释。"},
        {"role": "user", "content": "没有调查就没有发言权的出处是哪篇文章?"},
    ],
    temperature=0.7,
)
print(r.choices[0].message.content)