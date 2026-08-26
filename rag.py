"""RAG 知识库问答：加载文档 → 切分 → 嵌入 → 混合检索（BM25+向量）→ 重排序 → 生成。

支持 txt/md/pdf(含扫描件 OCR)/docx/csv/xlsx；增量索引（.index_state.json）；
含 Text-to-SQL 路由（sql_db.py）。Web 服务入口见 app.py。
"""

import json
import os

import jieba

from dotenv import load_dotenv

# 必须在 import huggingface 相关库之前加载 .env（HF 环境变量只在 import 时读一次）
load_dotenv()

# 国内镜像（可选）：HF_ENDPOINT=https://hf-mirror.com 可加速/修复模型下载
if os.getenv("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = os.getenv("HF_ENDPOINT")

# SSL 证书验证失败时（Windows/公司代理常见），可在 .env 设 USE_INSECURE_SSL=1 跳过验证
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
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from pydantic import ConfigDict, Field, PrivateAttr

from sql_db import SQLiteDb

# ========== 1. 加载文档（支持多格式） ==========
# docs/ 有文件 → 按扩展名路由加载全部；docs/ 不存在或为空 → 回退加载 knowledge_base.txt
DOCS_DIR = os.getenv("DOCS_DIR", "./docs")

# ===== 各格式的加载函数 =====
# 统一签名：fn(path, **kwargs) -> list[Document]，metadata 带 "source" 来源路径
# 文本类必须指定 encoding="utf-8"：Windows 默认 GBK 打开，知识库是 UTF-8，
# 不指定会报 UnicodeDecodeError
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
    """加载 .pdf：文字版直接提取；扫描页（提取文字过少）用 PyMuPDF 渲染 + RapidOCR 离线识别。"""
    docs = PyPDFLoader(path).load()
    # 找出"扫描页"（提取文字过少）
    ocr_idx = [i for i, d in enumerate(docs)
               if len(d.page_content.strip()) < OCR_MIN_TEXT_LEN]
    if not ocr_idx:
        print(f"  {path}: 文字版 PDF，{len(docs)} 页直接提取（无需 OCR）")
        return docs

    # —— OCR 缓存：识别过的页落盘，命中则直接读，不再重跑 ——
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

    # 懒加载：只有遇到扫描件才 import/初始化，普通文字版 PDF 不受影响
    print(f"  {path}: {len(ocr_idx)}/{len(docs)} 页是扫描件，开始 OCR（较慢，请耐心）...")
    import pymupdf as fitz  # 渲染 PDF 页为位图（新版包名 pymupdf，兼容旧名 fitz）
    import numpy as np
    from PIL import Image
    from rapidocr_onnxruntime import RapidOCR  # 离线中文 OCR，模型内置
    from tqdm import tqdm  # OCR 进度条

    ocr = RapidOCR()
    pdf = fitz.open(path)
    recognized = {}  # 页索引 -> OCR 文本（跑完一次性写入缓存）
    for i in tqdm(ocr_idx, desc=f"OCR {os.path.basename(path)}", unit="页"):
        pix = pdf[i].get_pixmap(dpi=200)      # 渲染位图（dpi 越高越清晰也越慢）
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        result, _ = ocr(np.array(img))        # [[四角坐标, 文字, 置信度], ...]
        if not result:                        # 识别不出也记空文本并缓存，避免下次重跑
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
    """加载 .xlsx：第 1 行视为表头，数据行拼成 "字段名: 值 | ..."（行自带字段名）；
    跳过全空行/重复表头行；无表头时整行用 | 连接。"""
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


# 扩展名 → (加载函数, 参数) 路由表
LOADER_MAP = {
    ".txt": (load_text, {"encoding": "utf-8"}),
    ".md": (load_text, {"encoding": "utf-8"}),
    ".pdf": (load_pdf, {}),
    ".docx": (load_docx, {}),
    ".csv": (load_csv, {"encoding": "utf-8"}),
    ".xlsx": (load_xlsx, {}),
}


def load_single_file(path):
    """加载单个文件（按扩展名路由）。增量索引的最小单元是一个文件：只加载有变化的那个。"""
    ext = os.path.splitext(path)[1].lower()
    entry = LOADER_MAP.get(ext)
    if entry is None:
        print(f"跳过不支持的文件类型: {os.path.basename(path)}")
        return []
    load_fn, load_kwargs = entry
    return load_fn(path, **load_kwargs)


def load_documents():
    """加载知识库文档，返回 Document 列表（每个带 page_content 和 metadata）。"""
    if os.path.isdir(DOCS_DIR) and any(os.listdir(DOCS_DIR)):
        docs = []
        # 递归遍历 docs/ 下所有文件，逐个加载
        for root, _, files in os.walk(DOCS_DIR):
            for name in files:
                docs.extend(load_single_file(os.path.join(root, name)))
        return docs
    else:
        # 回退：docs/ 为空时加载单文件知识库
        with open("knowledge_base.txt", encoding="utf-8") as f:
            return [Document(page_content=f.read(), metadata={"source": "knowledge_base.txt"})]


# ========== 2. 切分（按格式选择切分器） ==========
# - .md        → 按标题层级切（章节不拆散）
# - .csv/.xlsx → 加载时已按行拆好，不再切分
# - 其他格式   → RecursiveCharacterTextSplitter（chunk_overlap 让相邻块重叠，避免语义切断）
def split_documents_by_format(documents):
    """按文件扩展名路由切分器，返回切分后的 chunks。"""
    chunks = []
    by_ext = {}
    for doc in documents:
        ext = os.path.splitext(doc.metadata.get("source", ""))[1].lower()
        by_ext.setdefault(ext, []).append(doc)

    for ext, docs in by_ext.items():
        if ext == ".md":
            # Markdown：按标题层级切分
            md_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=[("#", "一级标题"), ("##", "二级标题"), ("###", "三级标题")],
                strip_headers=False,  # 标题文字保留在正文里，检索时能对上章节
            )
            for doc in docs:
                # 该切分器输入是字符串，切出的块没有 source，需手动补回来源
                for piece in md_splitter.split_text(doc.page_content):
                    piece.metadata["source"] = doc.metadata.get("source", "未知来源")
                    chunks.append(piece)
        elif ext in (".csv", ".xlsx"):
            # 表格：加载时已按行拆好，直接入库，不再切分
            chunks.extend(docs)
        else:
            # 通用切分器；用 split_documents 以保留来源 metadata
            splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
            chunks.extend(splitter.split_documents(docs))
    return chunks


# ========== 3. 嵌入 + 存储（Chroma 持久化 + 增量索引） ==========
# 片段转向量存 ./chroma_db（首次运行自动下载本地模型，约 100MB）
# 注意：DeepSeek 无 Embedding API，故用本地开源模型
embeddings = HuggingFaceEmbeddings(
    model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
)

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")

# ===== 增量索引：文件指纹状态管理（.index_state.json） =====
# 记录每个文件的指纹（mtime+size），启动时三向比对：
#   新增 → 只嵌入新文件；变更 → 只重建该文件；删除 → 只删该文件；未变 → 跳过
# 用 mtime+size 而非 md5：快；代价是"内容变了但 mtime/size 恰好没变"会漏检
INDEX_STATE_FILE = os.getenv("INDEX_STATE_FILE", "./.index_state.json")


def file_fingerprint(path):
    """文件指纹：修改时间 + 大小。"""
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
        # 分支 2：有索引但无状态文件——只记指纹，不重新嵌入
        print("检测到已有向量索引但无指纹状态（旧索引）→ 全量校准：记录指纹，不重新嵌入")
        save_index_state(scan)
        print(f"  已记录 {len(scan)} 个文件的指纹，下次启动开始走增量")
    else:
        # 分支 3：增量比对——只处理有变化的文件
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
                # 变更：先按 source 删旧片段，再入库新的（避免残留重复）
                vector_store.delete(where={"source": p})
            vector_store.add_documents(new_chunks)
        save_index_state(scan)  # 更新基线
        print("  指纹基线已更新")

# ========== 4. 创建检索器（混合检索 + 重排序） ==========
# 两路召回（BM25 关键词 + 向量语义）取并集 → 广召回 Top 50 → bge-reranker 精排 → 留 top_n 条
# 注：检索参数变化无需重建 chroma_db（只影响查询，不影响索引内容）

# ===== 3.5 模型（提前定义：MultiQuery 拆子查询与主问答链共用） =====
llm = ChatOpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1"),
    model=os.getenv("LLM_MODEL", "deepseek-chat"),
    temperature=0.3,
)

# ===== 手写工具类：RRF 融合 + Reranker 精排 =====
# 官方实现在独立包 langchain-retrievers（国内镜像未同步无法安装），故手写；
# 依赖只需 rank_bm25 + jieba（均已安装）


class RRFRetriever(BaseRetriever):
    """互惠排名融合（RRF）：多路 Top-k 排名按 1/(k+rank) 融合，只比排名不比分数。"""

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
    """bge-reranker 精排：粗召回候选 → 交叉编码器逐条算相关度 → 重排留 top_n 条。"""

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
    """多路子查询检索：LLM 把问题拆成多个子查询（覆盖同一事实的不同说法，
    解决"枚举/聚合"类问题单路检索漏全集）。

    三步：拆子查询 → 每路粗召回后合并成一个大 batch 交给 reranker 精排 →
    每路取精排前 per_query_top_n，按 rank 融合分排序去重 → 最终 top_n。
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
        """一次 LLM 调用生成子查询列表（JSON 数组）。
        解析失败逐级降级：json.loads → 正则抓 [...] → 返回原问题本身。"""
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
        # 每路粗召回，收集 (子查询, Document) 对；跨路去重：同一 page_content 只保留一次
        collected = []
        seen = set()
        raw_count = 0
        for sq in sub_queries:
            for doc in self.base_retriever.invoke(sq)[: self.recall_top_n]:
                raw_count += 1
                if doc.page_content in seen:
                    continue  # 已被其他子查询召回，跳过
                seen.add(doc.page_content)
                collected.append((sq, doc))
        print(f"  跨路去重：{raw_count} 对候选 → {len(collected)} 对唯一片段（节省 {raw_count - len(collected)}）")
        # 提速：全部候选对合并成一个 batch 一次 predict（模型只加载一次）
        # 显式标注类型，避免类型检查把 list 推断成 tuple
        pairs: List[List[str]] = [[sq, d.page_content] for sq, d in collected]
        scores = self._get_cross_encoder().predict(pairs, batch_size=64)
        # 按子查询分组，每路取精排前 per_query_top_n，rank 融合排序
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


# 4.1 BM25 关键词检索（无状态：每次启动从向量库全量取回文本现场重建）
#     坑：rank_bm25 默认按空格分词，中文整句会被当成一个 token，BM25 失效；
#         必须传 jieba 分词器（preprocess_func；新版独立包叫 tokenizer）
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

# 4.2 向量语义检索（Chroma，召回放宽到 50，交由 reranker 精排）
vector_retriever = vector_store.as_retriever(search_kwargs={"k": 50})

# 4.3 融合两路召回（手写 RRF）
ensemble_retriever = RRFRetriever(
    retrievers=[bm25_retriever, vector_retriever],
)

# 4.4 重排序（交叉编码器：候选逐条算相关度，精排后只留 top_n 条）
# 注意：以下实例未进当前管道（管道用 4.5 的 MultiQuery，内部自带批量精排），
#       保留作单路对比演示；删掉不影响运行
reranker_retriever = RerankerRetriever(
    base_retriever=ensemble_retriever,
    model_name=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base"),
    top_n=6,
)

# 4.5 MultiQuery 多路检索：LLM 拆子查询覆盖"不同说法"→ 每路独立召回精排 → rank 融合。
#     解决"枚举/聚合"类问题（几个/都有谁）单路检索漏全集。
#     管道 `| retriever |` 不用改，换的只是 retriever 实现。
retriever = MultiQueryRetriever(
    base_retriever=ensemble_retriever,
    llm=llm,
    sub_query_count=5,
    per_query_top_n=3,
    top_n=6,
)

# ========== 5. 提示词 ==========
# 检索结果作为"参考资料"进提示词，让模型据此回答
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

# 并行分支：context = 检索 question 并格式化成文本；question = 原样透传
chain = (
    {
        "context": (lambda x: x["question"]) | retriever | format_docs,
        "question": lambda x: x["question"],
    }
    | prompt
    | llm
)

# ========== 6.5 Text-to-SQL + 路由器 ==========
# 链路：路由器判断类型 → 事实查询走向量链（chain）→ 列举/统计走 SQL 链（sql_db.py）；
#       SQL 执行失败 → 报错回传 LLM 重写 1 次；空结果/仍失败 → 降级向量链
# 术语翻译靠 LLM 常识（schema 只给真实表头+样例），捏造列名由 sql_db 列名校验拦截
db = SQLiteDb()

# 6.5.1 路由器：判断问题走 vector（文档检索）还是 sql（数据库查询）
router_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是问题路由器。判断用户问题应该走哪条查询路径，只返回一个词：vector 或 sql。\n\n"
     "- vector：答案在叙述性文档里（问属性/关系/定义/背景），读几段原文就能答；\n"
     "  例如\"华信银行的行业是什么\"、\"李云龙的妻子是谁\"。\n"
     "- sql：需要对表格数据做列举/统计/聚合/排名，必须查数据库才能答；\n"
     "  例如\"所有客户的行业有哪些\"、\"客户总数是多少\"、\"各行业客户数量排名\"。\n\n"
     "判断规则：\n"
     "1. 出现统计/聚合词（多少/几个/总数/合计/平均/最大/最小/排行/排名/占比）→ sql\n"
     "2. 出现全集列举词（所有/全部/每个/各自/名单/明细/清单）→ sql\n"
     "3. 其余（是什么/怎么样/为什么/谁/关系/约定/条款）→ vector\n"
     "4. 不确定时默认 vector（文档检索覆盖面更广，SQL 只覆盖表格数据）"),
    ("human", "{question}"),
])
router_chain = router_prompt | llm | StrOutputParser()

# 6.5.2 Text-to-SQL：问题 + schema（真实表头/样例）→ LLM 生成 SELECT
sql_gen_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是 SQL 生成器。根据表结构把用户问题转成一条 SQLite SELECT 语句。\n\n"
     "表结构（列名可能与用户说法不同，请按语义匹配最合适的列，\n"
     "比如用户说\"诞生时间/成立时间\"，表里列名可能是\"创立日期\"）：\n{schema}\n\n"
     "要求：\n"
     "- 只输出 SQL 语句本身，不要任何解释\n"
     "- 只允许 SELECT，禁止 INSERT/UPDATE/DELETE/DROP\n"
     "- 中文值要加单引号，表名/列名不要加引号\n"
     "- 如果问题与表结构完全无关，只输出：NULL\n\n"
     "示例：\n"
     "问：所有客户的行业有哪些？\n"
     "答：SELECT DISTINCT 行业 FROM customers\n"
     "问：客户总数是多少？\n"
     "答：SELECT COUNT(*) FROM customers"),
    ("human", "{question}"),
])

# 6.5.3 SQL 执行失败时的重写提示词（报错回传，最多重试 1 次）
sql_retry_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你生成的 SQL 执行失败。请根据错误信息和表结构重新生成一条 SQLite SELECT 语句。\n\n"
     "表结构：\n{schema}\n"
     "你上次的 SQL：{sql}\n"
     "错误信息：{error}\n\n"
     "只输出修正后的 SQL 语句本身，不要解释。"),
    ("human", "{question}"),
])


def format_sql_result(cols, rows):
    """SQL 结果格式化成文本（纯拼接，不额外调 LLM）。
    注意：rows 是 dict 列表，取值必须 r[c]（zip 迭代 dict 取到的是键不是值）。"""
    parts = [f"[数据库查询] 共 {len(rows)} 条结果："]
    for r in rows:
        parts.append("  " + " | ".join(f"{c}: {r[c]}" for c in cols))
    return "\n".join(parts)


def sql_answer(question):
    """SQL 链：生成 SQL → 执行 → 报错重写 1 次；空结果/失败返回 None 由门面降级。"""
    schema = db.schema_text()
    sql = (sql_gen_prompt | llm | StrOutputParser()).invoke(
        {"question": question, "schema": schema}
    ).strip()
    print(f"  [SQL 链] 生成的 SQL：{sql}")
    if sql.upper() == "NULL":
        return None  # LLM 判断与表无关
    try:
        cols, rows = db.query(sql)
    except Exception as e:
        retry = (sql_retry_prompt | llm | StrOutputParser()).invoke(
            {"question": question, "schema": schema, "sql": sql, "error": str(e)}
        ).strip()
        print(f"  [SQL 链] 重写：{retry}")
        try:
            cols, rows = db.query(retry)
        except Exception as e2:
            print(f"  [SQL 链] 重写后仍失败，降级向量检索：{e2}")
            return None
    if not rows:
        return None  # 空结果：数据可能不在表里 → 交给向量链
    return format_sql_result(cols, rows)


def ask(question):
    """门面：单一入口。路由器分流 vector/sql，SQL 失败自动降级向量链。"""
    route = router_chain.invoke({"question": question}).strip()
    print(f"  [路由器] 问题类型：{route}")
    if route == "sql":
        ans = sql_answer(question)
        if ans is not None:
            return ans
        print("  [路由器] SQL 链无结果，降级向量检索")
    return chain.invoke({"question": question}).content


# ========== 7. 运行 ==========
if __name__ == "__main__":
    question = input("请输入你的问题（关于知识库内容）：")
    response = ask(question)
    print(f"\n回答：{response}")
