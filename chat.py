from openai import OpenAI
from config import API_KEY

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

messages = [
    {"role": "system", "content": "你是一个耐心的编程入门老师,回答简短,多用类比。"}
]

print("输入 exit 退出\n")

while True:
    question = input("你:")
    if question.strip() == "exit":
        break

    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
    )
    answer = response.choices[0].message.content
    print(f"AI:{answer}")
    print(f"[累计对话 {len(messages)} 条,token {response.usage.total_tokens}]\n")

    messages.append({"role": "assistant", "content": answer})