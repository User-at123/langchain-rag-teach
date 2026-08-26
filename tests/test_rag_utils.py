"""rag.py 纯函数测试：文件指纹 / 索引状态 / OCR 缓存 / 格式化 / 各格式加载器。

不调用 init()，不碰 LLM / 向量库 / 嵌入模型——import rag 后只测确定性逻辑。
这些用例同时是规划 A（加载器自研化）的"行为锁定"：替换实现后必须全部仍通过。
"""

import rag
from langchain_core.documents import Document


# ===== 文件指纹（V8.1 增量索引的判定依据） =====

def test_file_fingerprint_stable_and_sensitive(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("hello", encoding="utf-8")
    fp1 = rag.file_fingerprint(str(p))
    fp2 = rag.file_fingerprint(str(p))
    assert fp1 == fp2  # 同一文件两次调用指纹一致
    p.write_text("hello world", encoding="utf-8")
    assert rag.file_fingerprint(str(p)) != fp1  # 内容变大 → 指纹变化


# ===== 增量索引状态读写（.index_state.json） =====

def test_index_state_roundtrip(tmp_path, monkeypatch):
    state_file = str(tmp_path / "state.json")
    monkeypatch.setattr(rag, "INDEX_STATE_FILE", state_file)
    assert rag.load_index_state() == {}  # 文件不存在 → 空
    rag.save_index_state({"a.txt": "1-10", "b.txt": "2-20"})
    assert rag.load_index_state() == {"a.txt": "1-10", "b.txt": "2-20"}
    with open(state_file, "w", encoding="utf-8") as f:
        f.write("{broken json")  # 解析失败 → 按无状态处理
    assert rag.load_index_state() == {}


# ===== OCR 缓存读写（V8.0 扫描件支持） =====

def test_ocr_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "OCR_CACHE_DIR", str(tmp_path / "cache"))
    cache_path = rag._ocr_cache_path(str(tmp_path / "扫描件.pdf"))
    page_texts = {0: "第一页正文", 2: "第二页含\n换行和特殊符号 $#%"}
    rag._write_ocr_cache(cache_path, page_texts)
    assert rag._read_ocr_cache(cache_path) == page_texts


# ===== 格式化（进提示词前的文本整理） =====

def test_format_docs_with_source_and_fallback():
    docs = [
        Document(page_content="内容A", metadata={"source": "a.txt"}),
        Document(page_content="内容B", metadata={}),
    ]
    out = rag.format_docs(docs)
    assert "[来源: a.txt]" in out and "内容A" in out
    assert "[来源: 未知来源]" in out and "内容B" in out


def test_format_sql_result_uses_dict_values():
    """V9.0 踩坑回归：rows 是 dict 列表，取值必须 r[c]（zip 取到的是键不是值）。"""
    cols = ["客户名称", "行业"]
    rows = [
        {"客户名称": "华信银行", "行业": "金融"},
        {"客户名称": "蓝天航空", "行业": "航空"},
    ]
    out = rag.format_sql_result(cols, rows)
    assert "共 2 条结果" in out
    assert "客户名称: 华信银行" in out and "行业: 金融" in out
    assert "行业: 航空" in out


# ===== 加载器（规划 A 替换对象，测试锁定现状行为） =====

def test_load_text(tmp_path):
    p = tmp_path / "note.txt"
    p.write_text("这是知识库正文", encoding="utf-8")
    docs = rag.load_text(str(p), encoding="utf-8")
    assert len(docs) == 1
    assert docs[0].page_content == "这是知识库正文"
    assert docs[0].metadata["source"] == str(p)


def test_load_csv_each_row_a_document(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("客户名称,行业\n华信银行,金融\n蓝天航空,航空\n", encoding="utf-8")
    docs = rag.load_csv(str(p), encoding="utf-8")
    assert len(docs) == 2  # 表头不计入，每行一个 Document
    assert "客户名称" in docs[0].page_content and "华信银行" in docs[0].page_content
    assert "source" in docs[0].metadata


def test_load_xlsx_pairs_field_names(tmp_path):
    from openpyxl import Workbook

    p = tmp_path / "客户.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.title = "customers"
    ws.append(["客户名称", "行业"])
    ws.append(["华信银行", "金融"])
    ws.append(["蓝天航空", "航空"])
    wb.save(p)

    docs = rag.load_xlsx(str(p))
    assert len(docs) == 2
    assert "客户名称: 华信银行" in docs[0].page_content
    assert "customers" in docs[0].metadata["source"]


# ===== 切分（不依赖重资源，验证 source 保留） =====

def test_split_documents_by_format_keeps_source():
    doc = Document(page_content="知识库内容" * 300, metadata={"source": "x.txt"})
    chunks = rag.split_documents_by_format([doc])
    assert len(chunks) > 1  # 900 字按 chunk_size=200 必然被切多块
    assert all("source" in c.metadata for c in chunks)
