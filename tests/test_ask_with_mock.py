"""ask() 链路逻辑测试：用 FakeLLM / FakeRetriever 注入，验证路由分流与降级逻辑。

不调真实 API（免费、毫秒级、结果确定）；测的是"链路逻辑"而不是模型能力。
FakeLLM 按提示词类型（路由器 / SQL 生成 / 问答）分桶匹配，模拟真实 LLM 各环节输出。
"""

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.retrievers import BaseRetriever

import rag
from sql_db import SQLiteDb


class FakeLLM:
    """LCEL 兼容假 LLM：按 system 提示词区分环节，按 human 问题匹配预设输出。

    router:  {问题关键词: "vector"/"sql"}
    sql:     {问题关键词: "SELECT ..."}（同时服务首次生成与失败重写）
    answer:  {问题关键词: "最终回答文本"}
    """

    def __init__(self, router=None, sql=None, answer=None):
        self.script_router = router or {}
        self.script_sql = sql or {}
        self.script_answer = answer or {}
        self.calls = []  # 记录每次收到的 human 文本，便于断言调用次数

    def __call__(self, input):
        msgs = input.messages if hasattr(input, "messages") else []
        system = next((m.content for m in msgs if getattr(m, "type", "") == "system"), "")
        human = next((m.content for m in msgs if getattr(m, "type", "") == "human"), "")
        self.calls.append(human)
        if "问题路由器" in system:
            for k, v in self.script_router.items():
                if k in human:
                    return AIMessage(content=v)
            return AIMessage(content="vector")
        if "SQL" in system:
            for k, v in self.script_sql.items():
                if k in human:
                    return AIMessage(content=v)
            return AIMessage(content="SELECT NULL")
        for k, v in self.script_answer.items():
            if k in human:
                return AIMessage(content=v)
        return AIMessage(content="（默认回答）")


class FakeRetriever(BaseRetriever):
    """固定返回预设文档列表的假检索器。

    继承 BaseRetriever：LCEL 管道 `lambda | retriever` 走 Runnable.__ror__，
    普通对象不支持 `|` 操作（与真实链路 MultiQueryRetriever 同协议）。
    """

    docs: list

    def _get_relevant_documents(self, query: str, *, run_manager=None):
        return self.docs


@pytest.fixture
def test_db(tmp_path):
    """临时 SQLiteDb（1 表 3 行），供 SQL 路径断言真实执行结果。"""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "客户案例"
    ws.append(["客户名称", "行业"])
    ws.append(["华信银行", "金融"])
    ws.append(["蓝天航空", "航空"])
    ws.append(["绿野农业", "农业"])
    wb.save(docs_dir / "客户案例.xlsx")

    inst = SQLiteDb(db_path=str(tmp_path / "test.db"), docs_dir=str(docs_dir))
    yield inst
    inst.conn.close()


def _fake_retriever():
    return FakeRetriever(docs=[Document(page_content="无关占位", metadata={"source": "fake.txt"})])


def test_ask_sql_path_runs_real_query(test_db):
    """路由器判 sql → SQL 生成 → 真实执行临时库 → 返回格式化结果。"""
    llm = FakeLLM(
        router={"客户总数": "sql"},
        sql={"客户总数": "SELECT COUNT(*) FROM 客户案例"},
    )
    out = rag.ask("客户总数是多少？", llm=llm, retriever=_fake_retriever(), db=test_db)
    assert "共 1 条结果" in out
    assert "3" in out  # 临时库 3 行数据
    assert len(llm.calls) == 2  # 路由器 1 次 + SQL 生成 1 次，无重写


def test_ask_vector_path_returns_answer():
    """路由器判 vector → 问答链 → 返回 mock 回答。"""
    llm = FakeLLM(
        router={"李云龙": "vector"},
        answer={"李云龙": "李云龙的两位妻子是秀芹和田雨"},
    )
    out = rag.ask("李云龙妻子有几个？", llm=llm, retriever=_fake_retriever())
    assert "秀芹" in out and "田雨" in out
    assert len(llm.calls) == 2  # 路由器 + 问答


def test_ask_sql_fail_retry_then_fallback(test_db):
    """SQL 生成坏语句 → 执行失败 → 重写仍失败 → 降级向量链（V9.0 降级链路回归）。"""
    llm = FakeLLM(
        router={"客户总数": "sql"},
        sql={"客户总数": "SELECT 不存在的列 FROM 客户案例"},  # 永远失败
        answer={"客户总数": "降级回答：向量检索结果"},
    )
    out = rag.ask("客户总数是多少？", llm=llm, retriever=_fake_retriever(), db=test_db)
    assert out == "降级回答：向量检索结果"
    assert len(llm.calls) == 4  # 路由器 + 生成 + 重写 + 降级问答


def test_ask_sql_empty_result_falls_back(test_db):
    """SQL 合法但查不到数据（空结果）→ 不重写，直接降级向量链。"""
    llm = FakeLLM(
        router={"某行业": "sql"},
        sql={"某行业": "SELECT 客户名称 FROM 客户案例 WHERE 行业 = '不存在'"},
        answer={"某行业": "降级回答"},
    )
    out = rag.ask("某行业有哪些客户？", llm=llm, retriever=_fake_retriever(), db=test_db)
    assert out == "降级回答"
    assert len(llm.calls) == 3  # 路由器 + 生成 + 降级问答（无重写）


def test_ask_injection_does_not_pollute_globals(test_db):
    """注入 mock 后全局 chain/retriever 不应被修改（保持 init 前原样）。"""
    before = (rag.chain, rag.retriever, rag.router_chain)
    llm = FakeLLM(router={"李云龙": "vector"}, answer={"李云龙": "回答"})
    rag.ask("李云龙是谁？", llm=llm, retriever=_fake_retriever(), db=test_db)
    assert (rag.chain, rag.retriever, rag.router_chain) == before
