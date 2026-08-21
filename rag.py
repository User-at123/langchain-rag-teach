"""LangChain RAG 教学示例：检索增强生成（第一阶段：支持多格式知识库）。

流程：加载文档（txt/md/pdf/docx/csv）→ 切分 → 嵌入 → Chroma 检索 → 生成
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
from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ========== 1. 加载文档（支持多格式） ==========
# 规则：docs/ 目录里有文件 → 按扩展名路由到对应 Loader 加载全部；
#       docs/ 不存在或为空 → 回退加载单个 knowledge_base.txt（兼容原教学流程）
# 注意：新版 langchain-community（0.4+）移除了 DirectoryLoader 的 loader_map 参数，
#       所以这里改为手动遍历目录 + 按扩展名路由，逻辑更直白，也更好教学
DOCS_DIR = os.getenv("DOCS_DIR", "./docs")

# 扩展名 → (Loader 类, 传给它的参数) 的路由表（清晰直观）
# 为什么传 encoding="utf-8"：Windows 默认用 GBK 打开文本文件，
# 而知识库文件一般保存为 UTF-8（含中文），不指定就会报 UnicodeDecodeError
LOADER_MAP = {
    ".txt": (TextLoader, {"encoding": "utf-8"}),      # 纯文本
    ".md": (TextLoader, {"encoding": "utf-8"}),       # Markdown
    ".pdf": (PyPDFLoader, {}),                        # PDF（二进制，无编码问题）
    ".docx": (Docx2txtLoader, {}),                    # Word（二进制，无编码问题）
    ".csv": (CSVLoader, {"encoding": "utf-8"}),       # CSV（Excel 请另存为 CSV）
}


def load_documents():
    """加载知识库文档，返回 Document 列表（每个带 page_content 和 metadata）。"""
    if os.path.isdir(DOCS_DIR) and any(os.listdir(DOCS_DIR)):
        docs = []
        # 递归遍历 docs/ 下所有文件，按扩展名挑选对应 Loader 逐个加载
        for root, _, files in os.walk(DOCS_DIR):
            for name in files:
                ext = os.path.splitext(name)[1].lower()  # 取出扩展名，如 ".pdf"
                entry = LOADER_MAP.get(ext)
                if entry is None:
                    print(f"跳过不支持的文件类型: {name}")
                    continue
                loader_cls, loader_kwargs = entry  # 拆出 Loader 类和它的参数
                path = os.path.join(root, name)
                # 每个文件加载出一个或多个 Document（自带 metadata={"source": 路径}）
                docs.extend(loader_cls(path, **loader_kwargs).load())
        return docs
    else:
        # 回退路径：兼容旧版单文件教学
        with open("knowledge_base.txt", encoding="utf-8") as f:
            return [Document(page_content=f.read(), metadata={"source": "knowledge_base.txt"})]


documents = load_documents()
print(f"加载了 {len(documents)} 个文档")

# ========== 2. 切分（Chunking） ==========
# 长文本切小块，便于精确检索；chunk_overlap 让相邻块有重叠，避免语义被切断
# 第一阶段统一用 RecursiveCharacterTextSplitter（通用切分器）
# 注意：用 split_documents 而不是 split_text —— 这样切分后的块会保留来源 metadata
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
chunks = splitter.split_documents(documents)
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
    vector_store = Chroma.from_documents(
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
    """把检索到的文档列表拼成一段干净文本，并标注来源文件名。"""
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "未知来源")
        parts.append(f"[来源: {source}]\n{doc.page_content}")
    return "\n\n".join(parts)

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
