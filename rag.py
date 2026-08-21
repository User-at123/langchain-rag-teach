"""LangChain RAG 教学示例：检索增强生成。

与 main.py（普通问答）相比，RAG 多了一个"知识库"环节：
    加载文档 → 切分 → 嵌入 → 检索 → 生成
"""

import os

from dotenv import load_dotenv

# 必须在导入 huggingface 相关库之前加载 .env，
# 因为下面的 HF 环境变量只在 import 时被读取一次
load_dotenv()

# 国内镜像（可选）：HF_ENDPOINT=https://hf-mirror.com 可加速/修复模型下载
if os.getenv("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = os.getenv("HF_ENDPOINT")

# 若报 SSL 证书验证失败（Windows/公司代理常见），在 .env 设置 USE_INSECURE_SSL=1 跳过验证
# 注意：仅教学演示用，生产环境应修复证书而非禁用验证
if os.getenv("USE_INSECURE_SSL") == "1":
    os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ========== 1. 加载文档 ==========
# 读取本地知识库文件（教学用，实际项目中可能是 PDF、网页等）
with open("knowledge_base.txt", encoding="utf-8") as f:
    text = f.read()
print(f"已加载知识库，共 {len(text)} 字")

# ========== 2. 切分（Chunking） ==========
# 长文本切小块，便于精确检索；chunk_overlap 让相邻块有重叠，避免语义被切断
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
chunks = splitter.split_text(text)
print(f"切分成 {len(chunks)} 个片段")

# ========== 3. 嵌入 + 存储（Chroma 持久化） ==========
# 把每个片段转成向量，存入 Chroma 向量库，索引会保存到磁盘（./chroma_db 目录）
# 注意：DeepSeek 没有 Embedding API，所以用本地开源模型（首次运行会自动下载，约 100MB）
embeddings = HuggingFaceEmbeddings(
    model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
)

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")

# 第二次及以后运行：索引已在磁盘上，直接加载，跳过重新嵌入（几秒搞定）
if os.path.exists(CHROMA_DIR):
    print("检测到已有向量索引，直接加载（无需重新嵌入）")
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
# 第一次运行：创建索引并保存到磁盘
else:
    print("首次运行：创建向量索引并保存到磁盘 ...")
    vector_store = Chroma.from_texts(
        chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )

# ========== 4. 创建检索器 ==========
# 根据问题找出最相关的片段，k=2 表示取最相关的 2 个
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# ========== 5. 模型 + 提示词 ==========
# 把检索到的内容作为"参考资料"塞进提示词，让模型据此回答
llm = ChatOpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1"),
    model=os.getenv("LLM_MODEL", "deepseek-chat"),
    temperature=0.3,
)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个助手。请优先根据【参考资料】回答问题；"
     "如果参考资料里没有答案，请如实说明。\n\n"
     "【参考资料】\n{context}"),
    ("human", "{question}"),
])

# ========== 6. 组合成链 ==========
def format_docs(docs):
    """把检索到的文档列表拼成一段干净文本，作为参考资料。"""
    return "\n\n".join(doc.page_content for doc in docs)

# 并行分支：
#   - context：先从输入里取出 question 字符串（检索器只接受字符串），交给 retriever 检索，
#              再用 format_docs 把结果拼成文本
#   - question：原样透传问题给提示词模板
chain = (
    {
        "context": (lambda x: x["question"]) | retriever | format_docs,
        "question": lambda x: x["question"],
    }
    | prompt
    | llm
)

# ========== 7. 运行 ==========
if __name__ == "__main__":
    question = input("请输入你的问题（关于知识库内容）：")
    response = chain.invoke({"question": question})
    print(f"\n回答：{response.content}")
