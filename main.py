"""最简 LangChain 问答示例：提示词 + 模型。"""

import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()  # 读取 .env（OPENAI_API_KEY 等）

# DeepSeek 兼容 OpenAI 接口：用 ChatOpenAI，改 base_url 和模型名即可
llm = ChatOpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1"),
    model=os.getenv("LLM_MODEL", "deepseek-chat"),  # 可在 .env 中覆盖
    temperature=0.7,  # 随机性（0 严谨，1 发散）
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个乐于助人的中文助手。"),
    ("human", "{question}"),
])

chain = prompt | llm

if __name__ == "__main__":
    question = input("请输入你的问题：")
    response = chain.invoke({"question": question})
    print(f"\n回答：{response.content}")
