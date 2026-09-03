import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
)

model = os.getenv("QWEN_TEXT_MODEL", "qwen-plus")

response = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "user",
            "content": "请回复：Qwen API 调用成功"
        }
    ]
)

print(response.choices[0].message.content)