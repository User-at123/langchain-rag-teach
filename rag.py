"""LangChain RAG 教学示例：检索增强生成（第三阶段：混合检索 + 重排序）。

流程：加载文档（txt/md/pdf/docx/csv/xlsx）→ 按格式切分 → 嵌入 → 混合检索（BM25 关键词 + 向量语义）→ 重排序（bge-reranker）→ 生成
"""

import json
import os

import jieba

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

from typing import Any, List

from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from pydantic import ConfigDict, Field, PrivateAttr

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


# ===== PDF 加载（含扫描件自动 OCR）=====
# 判定阈值：一页提取出的文字少于该数量 → 视为扫描页（无文字层，需 OCR）
OCR_MIN_TEXT_LEN = 20
# OCR 结果缓存目录：识别出的文本落盘，下次构建索引直接读缓存，不再重跑 OCR
# 注意：必须在 docs/ 之外（如项目根），否则会被 os.walk 遍历当成文档重复加载
OCR_CACHE_DIR = os.getenv("OCR_CACHE_DIR", "./ocr_cache")


def _ocr_cache_path(path):
    """由 PDF 路径推导缓存文件路径：ocr_cache/<PDF 文件名去扩展名>.txt。"""
    os.makedirs(OCR_CACHE_DIR, exist_ok=True)
    return os.path.join(OCR_CACHE_DIR, os.path.splitext(os.path.basename(path))[0] + ".txt")


def _write_ocr_cache(cache_path, page_texts):
    """把 {页索引: 文本} 写入缓存文件（分隔行格式，正文可含换行）。
    读写都指定 newline="\\n"，避免 Windows 下 \\n 被写成 \\r\\n 导致解析错位。"""
    with open(cache_path, "w", encoding="utf-8", newline="\n") as f:
        for idx in sorted(page_texts):
            f.write(f"<<<PAGE:{idx}>>>\n{page_texts[idx]}\n")


def _read_ocr_cache(cache_path):
    """读缓存文件，返回 {页索引: 文本}。"""
    with open(cache_path, encoding="utf-8", newline="\n") as f:
        content = f.read()
    cached = {}
    for part in content.split("<<<PAGE:")[1:]:
        idx_str, _, text = part.partition(">>>\n")
        cached[int(idx_str)] = text.rstrip("\n")
    return cached


def load_pdf(path):
    """加载 .pdf：文字版直接提取；扫描页自动走 OCR（离线中文识别）。

    教学点：PDF 分两种——"文字版"（可复制，PyPDFLoader 直接读）和
    "扫描版"（每页是一张图，提取出来是空字符串）。
    这里逐页判定：文字层够的页直接用，不够的页用 PyMuPDF 渲染成图片
    + RapidOCR（包内自带模型，完全离线）识别，识别结果补回原 Document。
    """
    docs = PyPDFLoader(path).load()
    # 第 1 步：找出"扫描页"（提取文字过少）
    ocr_idx = [i for i, d in enumerate(docs)
               if len(d.page_content.strip()) < OCR_MIN_TEXT_LEN]
    if not ocr_idx:
        print(f"  {path}: 文字版 PDF，{len(docs)} 页直接提取（无需 OCR）")
        return docs

    # —— OCR 结果缓存：识别过的页落盘，下次直接读，不再重跑 ——
    cache_path = _ocr_cache_path(path)
    if os.path.exists(cache_path) and os.path.getmtime(cache_path) >= os.path.getmtime(path):
        cached = _read_ocr_cache(cache_path)
        hit = 0
        for i in ocr_idx:
            if i in cached:
                docs[i] = Document(
                    page_content=cached[i],
                    metadata={**docs[i].metadata, "ocr": True},
                )
                hit += 1
        print(f"  {path}: 命中 OCR 缓存（{hit}/{len(ocr_idx)} 页），跳过识别，直接读文本")
        return docs

    # 第 2 步：扫描页走 OCR（模型懒加载：只有遇到扫描件才 import/初始化，
    #         普通文字版 PDF 不受影响，启动不被拖慢）
    print(f"  {path}: {len(ocr_idx)}/{len(docs)} 页是扫描件，开始 OCR（较慢，请耐心）...")
    import pymupdf as fitz  # PyMuPDF：渲染 PDF 页为位图（新版包名 pymupdf，兼容旧名 fitz）
    import numpy as np
    from PIL import Image
    from rapidocr_onnxruntime import RapidOCR  # 离线中文 OCR，模型内置于包内
    from tqdm import tqdm  # 进度条：当前页/总页数、速度、预计剩余时间一目了然

    ocr = RapidOCR()
    pdf = fitz.open(path)
    recognized = {}  # 页索引 -> OCR 文本（跑完一次性写入缓存）
    for i in tqdm(ocr_idx, desc=f"OCR {os.path.basename(path)}", unit="页"):
        pix = pdf[i].get_pixmap(dpi=200)      # 页面渲染成位图（dpi 越高越清晰也越慢）
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        result, _ = ocr(np.array(img))        # result: [[四角坐标, 文字, 置信度], ...]
        if not result:                        # 纯图片页 / 识别不出 → 记空文本（也缓存，避免下次重跑）
            recognized[i] = ""
            continue
        result.sort(key=lambda r: r[0][0][1])  # 按行从上到下排序，保持阅读顺序
        recognized[i] = "\n".join(r[1] for r in result)
        docs[i] = Document(
            page_content=recognized[i],
            metadata={**docs[i].metadata, "ocr": True},  # 标记 OCR 来源，方便溯源
        )
    pdf.close()
    _write_ocr_cache(cache_path, recognized)
    print(f"  {path}: OCR 完成，共识别 {len(recognized)} 页；结果已缓存到 {cache_path}")
    print(f"          下次构建索引直接读缓存，这本 PDF 不再重跑 OCR")
    return docs


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


def load_single_file(path):
    """加载单个文件，返回 Document 列表（按扩展名路由到对应加载函数）。

    从 load_documents 的循环体抽出（V8.1 增量索引）：
    增量索引的最小处理单元是"一个文件"——新增/变更哪个文件就只加载哪个，
    这个函数就是那一步的入口。
    """
    ext = os.path.splitext(path)[1].lower()  # 取出扩展名，如 ".pdf"
    entry = LOADER_MAP.get(ext)
    if entry is None:
        print(f"跳过不支持的文件类型: {os.path.basename(path)}")
        return []
    load_fn, load_kwargs = entry  # 拆出加载函数和它的参数
    # 每个文件加载出一个或多个 Document（自带 metadata={"source": 路径}）
    return load_fn(path, **load_kwargs)


def load_documents():
    """加载知识库文档，返回 Document 列表（每个带 page_content 和 metadata）。"""
    if os.path.isdir(DOCS_DIR) and any(os.listdir(DOCS_DIR)):
        docs = []
        # 递归遍历 docs/ 下所有文件，逐个加载（复用一个文件的加载逻辑）
        for root, _, files in os.walk(DOCS_DIR):
            for name in files:
                docs.extend(load_single_file(os.path.join(root, name)))
        return docs
    else:
        # 回退路径：兼容旧版单文件教学
        with open("knowledge_base.txt", encoding="utf-8") as f:
            return [Document(page_content=f.read(), metadata={"source": "knowledge_base.txt"})]


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


# ========== 3. 嵌入 + 存储（Chroma 持久化 + 增量索引 V8.1） ==========
# 把每个片段转成向量，存入 Chroma 向量库，索引会保存到磁盘（./chroma_db 目录）
# 注意：DeepSeek 没有 Embedding API，所以用本地开源模型（首次运行会自动下载，约 100MB）
embeddings = HuggingFaceEmbeddings(
    model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
)

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")

# ===== 增量索引（V8.1）：文件指纹状态管理 =====
# 为什么需要：之前是"chroma_db 存在直接加载 / 不存在全量重建"，中间没有
# "哪些文件是新的"判断——往 docs/ 加新文件必须删库重建、所有文件重新嵌入，
# 扫描 PDF 每次全量重 OCR（最贵一环反复支付）。
# 现在用 .index_state.json 记录每个文件的指纹（mtime + size），启动时三向比对：
#   新增 → 只嵌入新文件；变更 → 只重建该文件；删除 → 只删该文件；未变 → 跳过。
# 指纹用 mtime+size 而非 md5：快（446 页 PDF 算 md5 有成本），教学够用；
# 代价是"内容变了但 mtime/size 恰好没变"会漏检（生产环境可换 md5）。
INDEX_STATE_FILE = os.getenv("INDEX_STATE_FILE", "./.index_state.json")


def file_fingerprint(path):
    """文件指纹：修改时间 + 大小（快，教学够用）。"""
    st = os.stat(path)
    return f"{st.st_mtime:.3f}-{st.st_size}"


def scan_docs_files():
    """扫描 docs/ 下所有支持的文件，返回 {路径: 指纹}（LOADER_MAP 之外的类型不参与）。"""
    scanned = {}
    if os.path.isdir(DOCS_DIR):
        for root, _, files in os.walk(DOCS_DIR):
            for name in files:
                path = os.path.join(root, name)
                if os.path.splitext(name)[1].lower() in LOADER_MAP:
                    scanned[path] = file_fingerprint(path)
    return scanned


def load_index_state():
    """读 .index_state.json，返回 {路径: 指纹}；文件不存在/解析失败返回 {}。"""
    if os.path.exists(INDEX_STATE_FILE):
        try:
            with open(INDEX_STATE_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"  警告：状态文件解析失败（{e}），按无状态处理（全量校准）")
    return {}


def save_index_state(state):
    """把 {路径: 指纹} 写回 .index_state.json（作为下次增量的基线）。"""
    with open(INDEX_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# ===== 三向分支：首次全量 / 旧索引校准 / 增量比对 =====
scan = scan_docs_files()

if not os.path.exists(CHROMA_DIR):
    # 分支 1：首次运行——全量加载 + 切分 + 嵌入
    print("首次运行：加载全部文档 ...")
    documents = load_documents()
    print(f"加载了 {len(documents)} 个文档")
    chunks = split_documents_by_format(documents)
    print(f"切分成 {len(chunks)} 个片段")
    print("创建向量索引并保存到磁盘 ...")
    vector_store = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    save_index_state(scan)  # 全部文件记为"已入库"
    print(f"  已记录 {len(scan)} 个文件的指纹（{INDEX_STATE_FILE}），下次启动开始走增量")
else:
    # 索引已存在：直接加载，不重新嵌入
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    state = load_index_state()
    if not state:
        # 分支 2：旧索引升级（有索引但无状态文件）——全量校准：只记指纹，不重新嵌入
        # 教学假设：旧索引内容与磁盘一致（删库重建时代的产物）
        print("检测到已有向量索引但无指纹状态（旧索引）→ 全量校准：记录指纹，不重新嵌入")
        save_index_state(scan)
        print(f"  已记录 {len(scan)} 个文件的指纹，下次启动开始走增量")
    else:
        # 分支 3：增量比对——只处理有变化的文件，其余零成本跳过
        added = [p for p in scan if p not in state]
        changed = [p for p in scan if p in state and scan[p] != state[p]]
        removed = [p for p in state if p not in scan]
        untouched = [p for p in scan if p in state and scan[p] == state[p]]
        print(f"增量比对：新增 {len(added)} 变更 {len(changed)} 删除 {len(removed)} 跳过 {len(untouched)}")
        for p in removed:
            print(f"  [删除] {p}")
            vector_store.delete(where={"source": p})
        for p in added + changed:
            action = "新增" if p in added else "变更"
            print(f"  [{action}] {p}（单文件处理，其余跳过）")
            docs = load_single_file(p)
            new_chunks = split_documents_by_format(docs)
            if p in changed:
                # 变更：先按 source 删掉旧片段，再入库新的（避免残留重复）
                vector_store.delete(where={"source": p})
            vector_store.add_documents(new_chunks)  # 只嵌入这 1 个文件的片段
        save_index_state(scan)  # 更新基线
        print("  指纹基线已更新")

# ========== 4. 创建检索器（第三阶段：混合检索 + 重排序） ==========
# 思路：两路召回（BM25 关键词 + 向量语义）取并集 → 广召回 Top 50 →
#       bge-reranker 精排 → 只留 top_n 条进提示词
# 为什么不再用单一向量检索：向量找"意思相近"，对专有名词/精确词不敏感；
#   BM25 按词频命中关键词，两者互补。数据量变大后（千级~万级片段），
#   靠调大 k 会噪声爆炸（见 mindmap 核心概念 4/6），混合检索 + 精排才是"又全又准"的正解。
#   k 的职责变化（V7.7 → V7.9）：k 从"回答视野"降级为"粗筛漏斗"，精排后才定最终视野。
# 注：检索参数变化不需要重建 chroma_db（只影响查询，不影响索引内容）。

# ===== 3.5 模型（提前定义） =====
# 提前到检索器之前：V8.2 的 MultiQueryRetriever 需要 LLM 把问题拆成多路子查询
#（见 4.5），主问答链（第 5 节）复用同一个 llm 实例。
llm = ChatOpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1"),
    model=os.getenv("LLM_MODEL", "deepseek-chat"),
    temperature=0.3,
)

# ===== 手写工具类：RRF 融合 + Reranker 精排 =====
# 为什么手写：官方 EnsembleRetriever / CrossEncoderReranker 在独立包 langchain-retrievers 中，
# 但该包国内镜像（清华/阿里）未同步、PyPI 无法安装；且手写实现（各约 20 行）能把
# 这两个"黑盒"讲透，教学价值更高。依赖只需 rank_bm25 + jieba（均已安装）。


class RRFRetriever(BaseRetriever):
    """互惠排名融合（Reciprocal Rank Fusion）——手写版 EnsembleRetriever。

    多路检索器各自给出 Top-k 排名，融合公式：score(d) = Σ 1/(k + rank_i(d))
    - 只比"排名"不比"分数"：BM25 的得分和向量余弦相似度量纲不同，直接相加无意义；
    - 排名越靠前贡献越大，两路都命中的片段自然排最前（"互补"的数学体现）。
    """

    retrievers: List[BaseRetriever] = Field(description="多路检索器（如 BM25 + 向量）")
    rrf_k: int = Field(default=60, description="RRF 常数（经验值 60）")
    top_n: int = Field(default=50, description="融合后保留的候选条数")

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None):
        scores: dict = {}    # page_content -> 累计融合分
        docs_map: dict = {}  # page_content -> Document（保留 metadata）
        for retriever in self.retrievers:
            for rank, doc in enumerate(retriever.invoke(query)):
                key = doc.page_content
                docs_map.setdefault(key, doc)
                scores[key] = scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank)
        ranked = sorted(
            docs_map.values(),
            key=lambda d: scores[d.page_content],
            reverse=True,
        )
        return ranked[: self.top_n]


class RerankerRetriever(BaseRetriever):
    """bge-reranker 精排检索器——手写版官方"重排序器 + 压缩检索器"组合。

    流程：base_retriever 粗召回 Top 50 → 交叉编码器把"问题+文档"拼成一段算相关度 →
          按分数重排，只留 top_n 条。
    为什么用交叉编码器：双塔嵌入（向量检索）先各自编码再算相似度，快但精度有限；
    交叉编码器同时看问题与文档全文，精度高但逐条计算慢，所以只对少量候选精排。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    base_retriever: BaseRetriever = Field(description="粗召回检索器（混合检索）")
    model_name: str = Field(
        default="BAAI/bge-reranker-base",
        description="交叉编码器模型名（首次运行自动下载）",
    )
    top_n: int = Field(
        default=6,
        description="精排后保留的条数（最终进提示词的视野，承接 V7.7 的 k=6 经验）",
    )

    _cross_encoder: Any = PrivateAttr(default=None)

    def _get_cross_encoder(self):
        """懒加载交叉编码器：首次调用才下载模型，避免 import 阶段联网卡住"""
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder

            self._cross_encoder = CrossEncoder(self.model_name)
        return self._cross_encoder

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None):
        candidates = self.base_retriever.invoke(query)     # 粗召回 Top 50
        pairs = [(query, d.page_content) for d in candidates]
        scores = self._get_cross_encoder().predict(pairs)  # 逐条相关度，越大越相关
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [d for d, _ in ranked[: self.top_n]]


class MultiQueryRetriever(BaseRetriever):
    """多路子查询检索器——手写版官方 MultiQueryRetriever（V8.2 新增）。

    解决的问题：枚举/聚合类问题（"有几个/都是谁"）单路检索会漏"全集"。
    原因：一个问法只能匹配一种"说法"，而同一事实在原文有多种表达
    （田雨=妻子，秀芹=婆娘/娶媳妇/新婚妻子）。问题里只带"妻子"一词，
    BM25/向量就只命中田雨（书里明确写"你的妻子田雨"），秀芹全漏。

    流程（三步）：
      1. LLM 一次调用把问题拆成多路子查询（覆盖不同说法/人名/角度）
      2. 每路子查询各自粗召回（RRF top50），全部候选【合并成一个 batch】
         一次性喂给 reranker 精排——每对用自己的子查询打分
      3. 每路取精排前 per_query_top_n → 按 rank 融合分排序（不同路的分数
         量纲不可比，只比排名）→ 合并去重 → 最终 top_n 进 LLM 视野

    并行要点（教学点，见 mindmap 核心概念 10）：
      - 拆分子查询是一次 LLM 调用，没有并行空间（多线程调用只会翻倍成本）
      - 提速靠 reranker 的 batch 推理：全部候选对合并成一个大 batch 一次
        predict，模型只加载一次、单次推理，比逐路 predict 快得多；
        也比多线程更有效（Python GIL 下 CPU 推理多线程会串行化）
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    base_retriever: BaseRetriever = Field(description="粗召回检索器（RRF 混合检索）")
    llm: Any = Field(description="生成子查询的 LLM（一次调用生成多路）")
    sub_query_count: int = Field(default=5, description="拆成几路子查询")
    per_query_top_n: int = Field(default=4, description="每路子查询精排后保留条数（给弱相关但关键的信息留空间）")
    recall_top_n: int = Field(
        default=20,
        description="每路子查询粗召回条数。精排只取前 per_query_top_n 条进最终视野，"
        "召回 50 条纯属浪费——reranker 是本地 CPU 交叉编码器，50 条/路 × 5 路 = 250 对"
        "要算 ~25 秒（实测），砍到 20 条/路 = 100 对 ≈ 11 秒，质量不变（V8.2 性能优化）",
    )
    top_n: int = Field(default=6, description="最终进 LLM 视野的条数")
    model_name: str = Field(
        default="BAAI/bge-reranker-base",
        description="交叉编码器模型名（与 RerankerRetriever 同款）",
    )

    _cross_encoder: Any = PrivateAttr(default=None)

    def _get_cross_encoder(self):
        """懒加载交叉编码器：首次调用才下载模型（与 RerankerRetriever 同款）"""
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder

            self._cross_encoder = CrossEncoder(self.model_name)
        return self._cross_encoder

    def _generate_sub_queries(self, query):
        """一次 LLM 调用生成子查询列表（要求 JSON 数组，容错解析）。

        解析三级降级：json.loads → 正则抓 [...] → 退化返回原问题本身，
        保证 LLM 输出不规范时检索链路不崩。
        """
        import json
        import re

        prompt = (
            f"你是检索辅助器。请把用户问题改写成 {self.sub_query_count} 个不同的子查询，"
            "用于分别检索知识库。\n"
            "要求：\n"
            "1. 如果问题在问一个\"集合\"（如\"有几个/都有谁/分别是\"），请为每个可能的成员"
            "单独生成一个子查询，并直接带上该成员的名字或专有名词"
            "（如\"李云龙第一任妻子 杨秀芹\"、\"李云龙第二任妻子 田雨\"）；\n"
            "2. 每个子查询只聚焦一个对象/一个说法，不要在一个子查询里混多个名字；\n"
            "3. 可补充\"关系式\"问法（如\"杨秀芹和李云龙是什么关系\"），绕开原问题里的关键词；\n"
            "4. 每个子查询包含足够上下文词，能独立完成检索；\n"
            "5. 只返回 JSON 数组字符串（如 [\"子查询1\", \"子查询2\", ...]），不要任何其他文字。\n\n"
            f"用户问题：{query}"
        )
        resp = self.llm.invoke(prompt)
        text = getattr(resp, "content", str(resp)).strip()

        def _try_parse(t):
            arr = json.loads(t)
            if isinstance(arr, list):
                return [s.strip() for s in arr if isinstance(s, str) and s.strip()]
            return None

        try:
            subs = _try_parse(text)
            if subs:
                return subs[: self.sub_query_count]
        except Exception:
            pass
        m = re.search(r"\[.*\]", text, re.S)
        if m:
            try:
                subs = _try_parse(m.group(0))
                if subs:
                    return subs[: self.sub_query_count]
            except Exception:
                pass
        return [query]  # 解析失败：退化用原问题本身

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None):
        sub_queries = self._generate_sub_queries(query)
        # 第 2 步：每路粗召回，收集 (子查询, Document) 对
        # V8.3 跨路去重：5 路子查询是同一问题的不同问法，命中的语料高度重叠——
        # 实测 100 对候选按 page_content 去重后只剩 53 个唯一片段（重复率 47%），
        # reranker 对相同内容算了不止一遍。同一片段只保留第一次出现，打分对数直接减半。
        collected = []
        seen = set()  # 跨路去重：已见过的 page_content 不再重复打分
        raw_count = 0
        for sq in sub_queries:
            for doc in self.base_retriever.invoke(sq)[: self.recall_top_n]:
                raw_count += 1
                if doc.page_content in seen:
                    continue  # 被其他子查询召回过 → 跳过（同一片段取一次就够）
                seen.add(doc.page_content)
                collected.append((sq, doc))
        print(f"  跨路去重：{raw_count} 对候选 → {len(collected)} 对唯一片段（节省 {raw_count - len(collected)}）")
        # 关键提速点：全部候选对合并成一个 batch 一次 predict（批量并行），
        # 每对用自己的子查询打分——比逐路 predict 少加载模型、单次推理更快
        # 明确标注输入输出类型，避免类型检查把 list 推断成 tuple（[List, Document] 元组序）
        pairs: List[List[str]] = [[sq, d.page_content] for sq, d in collected]
        scores = self._get_cross_encoder().predict(pairs, batch_size=64)
        # 第 3 步：按子查询分组，每路取精排前 per_query_top_n，rank 融合排序
        by_route = {}
        for (sq, doc), s in zip(collected, scores):
            by_route.setdefault(sq, []).append((doc, s))
        fusion = {}  # page_content -> [Document, 融合分]
        for sq, items in by_route.items():
            for rank, (doc, s) in enumerate(
                sorted(items, key=lambda x: x[1], reverse=True)
            ):
                if rank >= self.per_query_top_n:
                    break  # 每路只保留精排前 per_query_top_n 条
                key = doc.page_content
                if key not in fusion:
                    fusion[key] = [doc, 0.0]
                # 只比排名不比分数：不同子查询打出的分数量纲不可比（实验已验证）
                fusion[key][1] += 1.0 / (60 + rank)
        ranked = sorted(fusion.values(), key=lambda x: x[1], reverse=True)
        return [d for d, _ in ranked[: self.top_n]]


# 4.1 BM25 关键词检索（内存索引，无状态：每次启动从向量库全量取回文本现场重建）
#     为什么不再用内存 chunks（V8.1 增量索引后的关键联动）：
#     增量模式下不再有"全量 chunks"（只有变化文件的局部片段），而 BM25 必须拿到
#     全量文本——向量库恰好存着全量文本，取回即重建。
#     教学点：向量索引持久化（落盘、可增量维护）/ BM25 无状态（每次现造，不落盘）。
#     坑：rank_bm25 默认按空格分词，中文整句会被当成一个 token，BM25 直接失效；
#         必须传入 jieba 分词器（中文场景的关键踩坑点）。
#     参数名注意：本项目 langchain-community 版本用 preprocess_func（新版独立包才叫 tokenizer）
_all_docs = vector_store.get(include=["documents", "metadatas"])
bm25_texts = _all_docs["documents"]
bm25_metadatas = _all_docs["metadatas"]
print(f"BM25 从向量库取回 {len(bm25_texts)} 个片段重建（无状态索引）")
bm25_retriever = BM25Retriever.from_texts(
    bm25_texts,
    k=50,
    preprocess_func=lambda t: list(jieba.cut(t)),
    metadatas=bm25_metadatas,  # 保留 source 等元数据，方便追溯
)

# 4.2 向量语义检索（现有 Chroma，召回放宽到 50，让 reranker 来精排）
vector_retriever = vector_store.as_retriever(search_kwargs={"k": 50})

# 4.3 融合两路召回（手写 RRF：只比排名不比分数，两路都命中的片段自然最靠前）
ensemble_retriever = RRFRetriever(
    retrievers=[bm25_retriever, vector_retriever],
)

# 4.4 重排序（bge-reranker 交叉编码器：候选逐条与问题算相关度，精排后只留 top_n 条）
# 注意：以下实例是【教学保留件】——定义了但没有进当前管道（chain 里用的是 4.5 的
#       MultiQuery，它内部自带"批量精排 + rank 融合"，把本类的活包进去了）。
#       保留它只为单独演示"粗召回 → 精排"两段式，或临时替换管道对比单路 vs 多路效果。
#       删掉它不影响运行（链路照常），想删就删。
reranker_retriever = RerankerRetriever(
    base_retriever=ensemble_retriever,
    model_name=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base"),
    top_n=6,
)

# 4.5 MultiQuery 多路检索（V8.2 新增，手写版）：
#     LLM 拆分子查询覆盖"不同说法" → 每路独立召回精排 → rank 融合合并。
#     解决"枚举/聚合类问题"（几个/都有谁）单路检索漏全集——典型案例如
#     "李云龙妻子有几个"：书里秀芹从不用"妻子"称呼，单路只命中田雨。
#     RerankerRetriever 保留为单路精排组件（reranker_retriever，教学保留件，未进管道，
#     见 4.4 标注），最外层 retriever 换 MultiQuery，管道 `| retriever |` 一行不用改。
retriever = MultiQueryRetriever(
    base_retriever=ensemble_retriever,
    llm=llm,
    sub_query_count=5,
    per_query_top_n=3,
    top_n=6,
)

# ========== 5. 提示词 ==========
# 把检索到的内容作为"参考资料"塞进提示词，让模型据此回答
#（llm 已在 3.5 节定义，MultiQuery 拆分子查询与主问答链共用同一个实例）
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
