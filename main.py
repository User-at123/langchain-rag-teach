"""LangChain 教学示例：一个最简的"提示词 + 模型"调用流程。"""

import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 1. 加载 .env 文件中的环境变量（如 OPENAI_API_KEY）
load_dotenv()

# 2. 创建模型（LLM）
# DeepSeek 兼容 OpenAI 接口，所以用 ChatOpenAI，只需改 base_url 和模型名
llm = ChatOpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1"),
    model=os.getenv("LLM_MODEL", "deepseek-chat"),  # 模型名称，可在 .env 中覆盖
    temperature=0.7,                                # 回答的随机性（0 最严谨，1 最发散）
)

# 3. 定义提示词模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个乐于助人的中文助手。"),
    ("human", "{question}"),
])

# 4. 用管道符把"提示词 + 模型"组合成一条链（LangChain 核心概念）
chain = prompt | llm

# 5. 运行链
if __name__ == "__main__":
    question = input("请输入你的问题：")
    response = chain.invoke({"question": question})
    print(f"\n回答：{response.content}")
