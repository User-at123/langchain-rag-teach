"""LangChain RAG 教学示例：检索增强生成（第二阶段：多格式加载 + 按格式差异化切分）。

流程：加载文档（txt/md/pdf/docx/csv/xlsx）→ 按格式切分 → 嵌入 → Chroma 检索 → 生成
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
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# ========== 1. 加载文档（支持多格式） ==========
# 规则：docs/ 目录里有文件 → 按扩展名路由到对应 Loader 加载全部；
#       docs/ 不存在或为空 → 回退加载单个 knowledge_base.txt（兼容原教学流程）
# 注意：新版 langchain-community（0.4+）移除了 DirectoryLoader 的 loader_map 参数，
#       所以这里改为手动遍历目录 + 按扩展名路由，逻辑更直白，也更好教学
DOCS_DIR = os.getenv("DOCS_DIR", "./docs")

# ===== 各格式的加载函数 =====
# 统一签名：fn(path, **kwargs) -> list[Document]，每个 Document 带 metadata={"source": ...}
# 为什么文本类要指定 encoding="utf-8"：Windows 默认用 GBK 打开文本文件，
# 而知识库文件一般保存为 UTF-8（含中文），不指定就会报 UnicodeDecodeError
def load_text(path, encoding):
    """加载 .txt / .md：LangChain 的 TextLoader，读入整个文件。"""
    return TextLoader(path, encoding=encoding).load()


def load_pdf(path):
    """加载 .pdf：PyPDFLoader 按页切出多个 Document（二进制解析，无编码问题）。"""
    return PyPDFLoader(path).load()


def load_docx(path):
    """加载 .docx：Docx2txtLoader 提取 Word 中的文本（二进制解析，无编码问题）。"""
    return Docx2txtLoader(path).load()


def load_csv(path, encoding):
    """加载 .csv：CSVLoader 默认按行切，每行一个 Document（表格语义在"行"）。"""
    return CSVLoader(path, encoding=encoding).load()


def load_xlsx(path):
    """加载 .xlsx（2.5 步新增，原生 Excel 支持；2.6 步改进表头拼接）。
    用 openpyxl 按行读取：
    - 第 1 行视为表头（字段名），数据行拼接为 "字段名: 值 | 字段名: 值"，
      让每一行自带字段名，解决"字段名被单独切块、检索命中表头却拿不到值"的问题；
    - 跳过全空行、跳过与表头完全重复的行；
    - 表头全空（无表头的表）时退化为原行为：整行单元格用 | 连接。
    为什么不用 LangChain 的 Excel Loader：UnstructuredExcelLoader 需要重量级的
    unstructured 依赖，教学项目用 openpyxl 自己写更轻、更透明。"""
    from openpyxl import load_workbook
    wb = load_workbook(path, read_only=True, data_only=True)
    docs = []
    for sheet in wb.worksheets:
        rows = list(sheet.iter_rows(values_only=True))
        if not rows:
            continue
        header = ["" if c is None else str(c).strip() for c in rows[0]]
        if any(header):  # 有表头：字段名拼进每行数据
            for row_no, row in enumerate(rows[1:], start=2):
                cells = ["" if c is None else str(c).strip() for c in row]
                if not any(cells):
                    continue  # 跳过全空行
                if cells == header:
                    continue  # 跳过与表头重复的行（有些表会重复打印表头）
                pairs = [f"{h}: {v}" for h, v in zip(header, cells) if h and v]
                if not pairs:
                    continue
                docs.append(Document(
                    page_content=" | ".join(pairs),
                    metadata={"source": f"{path} [工作表: {sheet.title} 第{row_no}行]"},
                ))
        else:  # 无表头：整行单元格直接 | 连接（原行为）
            for row_no, row in enumerate(rows, start=1):
                cells = ["" if c is None else str(c) for c in row]
                if not any(cells):
                    continue
                docs.append(Document(
                    page_content=" | ".join(cells),
                    metadata={"source": f"{path} [工作表: {sheet.title} 第{row_no}行]"},
                ))
    return docs


# 扩展名 → (加载函数, 传给它的参数) 的路由表（清晰直观）
LOADER_MAP = {
    ".txt": (load_text, {"encoding": "utf-8"}),   # 纯文本
    ".md": (load_text, {"encoding": "utf-8"}),    # Markdown
    ".pdf": (load_pdf, {}),                       # PDF
    ".docx": (load_docx, {}),                     # Word
    ".csv": (load_csv, {"encoding": "utf-8"}),    # CSV
    ".xlsx": (load_xlsx, {}),                     # Excel（原生支持）
}


def load_documents():
    """加载知识库文档，返回 Document 列表（每个带 page_content 和 metadata）。"""
    if os.path.isdir(DOCS_DIR) and any(os.listdir(DOCS_DIR)):
        docs = []
        # 递归遍历 docs/ 下所有文件，按扩展名挑选对应加载函数逐个加载
        for root, _, files in os.walk(DOCS_DIR):
            for name in files:
                ext = os.path.splitext(name)[1].lower()  # 取出扩展名，如 ".pdf"
                entry = LOADER_MAP.get(ext)
                if entry is None:
                    print(f"跳过不支持的文件类型: {name}")
                    continue
                load_fn, load_kwargs = entry  # 拆出加载函数和它的参数
                path = os.path.join(root, name)
                # 每个文件加载出一个或多个 Document（自带 metadata={"source": 路径}）
                docs.extend(load_fn(path, **load_kwargs))
        return docs
    else:
        # 回退路径：兼容旧版单文件教学
        with open("knowledge_base.txt", encoding="utf-8") as f:
            return [Document(page_content=f.read(), metadata={"source": "knowledge_base.txt"})]


documents = load_documents()
print(f"加载了 {len(documents)} 个文档")

# ========== 2. 切分（Chunking）：按格式选择切分器 ==========
# 2.5 步升级：不同格式的"语义单元"不同，切分策略也要不同：
#   - .md        → MarkdownHeaderTextSplitter：按标题层级切，章节不拆散
#   - .csv/.xlsx → 表格语义在"行"：加载时已经按行拆成一个个 Document，无需再切
#   - 其他格式   → RecursiveCharacterTextSplitter：通用切分器（长文本切小块，
#                  chunk_overlap 让相邻块有重叠，避免语义被切断）
def split_documents_by_format(documents):
    """按文件扩展名路由切分器，返回切分后的 chunks。"""
    chunks = []
    # 先把所有文档按扩展名分组：{".md": [...], ".csv": [...], ...}
    by_ext = {}
    for doc in documents:
        ext = os.path.splitext(doc.metadata.get("source", ""))[1].lower()
        by_ext.setdefault(ext, []).append(doc)

    for ext, docs in by_ext.items():
        if ext == ".md":
            # Markdown：按标题层级切分，标题会写进每个块的 metadata，方便溯源
            md_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=[("#", "一级标题"), ("##", "二级标题"), ("###", "三级标题")],
                strip_headers=False,  # 标题文字保留在正文里，检索时能对上章节
            )
            for doc in docs:
                # MarkdownHeaderTextSplitter 输入是字符串（不是 Document），
                # 切出的块没有 source，需要手动补回来源文件名
                for piece in md_splitter.split_text(doc.page_content):
                    piece.metadata["source"] = doc.metadata.get("source", "未知来源")
                    chunks.append(piece)
        elif ext in (".csv", ".xlsx"):
            # 表格：一行 = 一条记录，加载时已按行拆好，直接入库，不再切分
            chunks.extend(docs)
        else:
            # 通用切分器：.txt / .pdf / .docx / knowledge_base.txt 等
            # 用 split_documents 而不是 split_text —— 切分后的块会保留来源 metadata
            splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
            chunks.extend(splitter.split_documents(docs))
    return chunks


chunks = split_documents_by_format(documents)
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
# 根据问题找出最相关的片段：
#   - k=6：返回最相关的 6 个片段。k 是"回答的视野"——k 太小（如 2），
#     问"所有客户的行业有哪些"只召回 1 条案例行，模型只能答出 1 个客户；
#     k 调大后同类记录都能进候选，模型才能汇总出完整列表
#   - 曾经尝试 MMR（最大边际相关性）检索：它在"相关 + 多样"间平衡，但 3 条客户案例行
#     结构相似会被当成"重复"而只留 1 条，反而丢信息；普通相似度检索在小知识库更直观有效
retriever = vector_store.as_retriever(search_kwargs={"k": 6})

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
