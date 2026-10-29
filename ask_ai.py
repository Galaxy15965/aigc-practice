from openai import OpenAI
from config import API_KEY

print("key 长度:", len(API_KEY))

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com",
)

question = input("问点什么:")
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": question}],
)
print(response.choices[0].message.content)