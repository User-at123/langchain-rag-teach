"""多轮记忆 + 跨轮状态（第 9 步）链路测试：注入 FakeLLM 验证三路路由。

MemoryFakeLLM 在 FakeLLM 基础上新增"记忆补全器"分桶（system 含"对话记忆处理器"），
其余分桶规则与 test_ask_with_mock 完全一致。所有用例走注入模式（不调真实 API）。

覆盖点：
1. 追问指代：路由器判 memory → 补全器改写为独立问题 → 按 base 走 vector 链
2. 跨轮状态：SQL 结果暂存会话 → 引用"刚才结果"时 base=last_result 直接答（不重查库）
3. 补全器解析失败 → 降级走完整路由（不崩溃）
4. 无 session_id 时不写会话（保持 V10 无记忆行为）
"""

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.retrievers import BaseRetriever

import rag
from sql_db import SQLiteDb


class MemoryFakeLLM:
    """4 桶假 LLM：router / memory（补全器）/ sql / answer。

    router:   {问题关键词: "vector"/"sql"/"memory"}
    memory:   {问题关键词: '{"rewritten": "...", "base": "..."}'}
    sql:      {问题关键词: "SELECT ..."}
    answer:   {问题关键词: "最终回答文本"}（同时服务向量链与 last_result 直答）
    """

    def __init__(self, router=None, memory=None, sql=None, answer=None):
        self.script_router = router or {}
        self.script_memory = memory or {}
        self.script_sql = sql or {}
        self.script_answer = answer or {}
        self.calls = []

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
        if "对话记忆处理器" in system:
            for k, v in self.script_memory.items():
                if k in human:
                    return AIMessage(content=v)
            return AIMessage(content='{"rewritten": "%s", "base": "vector"}' % human)
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
    """固定返回预设文档列表的假检索器（与 test_ask_with_mock 同实现）。"""

    docs: list

    def _get_relevant_documents(self, query: str, *, run_manager=None):
        return self.docs


@pytest.fixture
def test_db(tmp_path):
    """临时 SQLiteDb（1 表 3 行），供跨轮状态（last_result）断言真实执行结果。"""
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


@pytest.fixture(autouse=True)
def clean_sessions():
    """每个用例前后清空会话存储，避免跨用例污染。"""
    rag.sessions.clear()
    yield
    rag.sessions.clear()


def _fake_retriever():
    return FakeRetriever(docs=[Document(page_content="无关占位", metadata={"source": "fake.txt"})])


def test_memory_rewrite_then_vector():
    """追问指代：路由器判 memory → 补全器改写为独立问题 → 走 vector 链回答。"""
    llm = MemoryFakeLLM(
        router={"李云龙": "vector", "她们": "memory"},
        memory={"她们": '{"rewritten": "秀芹和田雨的结局是什么", "base": "vector"}'},
        answer={"李云龙": "李云龙的两位妻子是秀芹和田雨",
                "秀芹和田雨的结局": "秀芹在战斗中牺牲，田雨陪伴李云龙到老"},
    )
    sid = "s-rewrite"
    # 第 1 轮：普通问题建立历史
    out1 = rag.ask("李云龙的妻子是谁？", session_id=sid, llm=llm,
                   retriever=_fake_retriever())
    assert "秀芹" in out1
    # 第 2 轮：追问指代 → 补全器改写后走向量链
    out2 = rag.ask("那她们结局如何？", session_id=sid, llm=llm,
                   retriever=_fake_retriever())
    assert "秀芹在战斗中牺牲" in out2
    # 调用序列：第 1 轮 路由器+问答；第 2 轮 路由器+补全器+问答
    assert len(llm.calls) == 5
    # 会话已记录两轮问答
    assert rag.sessions[sid]["history"][-1] == ("assistant", out2)


def test_memory_uses_last_sql_result(test_db):
    """跨轮状态：上轮 SQL 结果暂存 → 引用"刚才结果"直接基于 last_result 回答（不重查库）。"""
    llm = MemoryFakeLLM(
        router={"客户总数": "sql", "刚才": "memory"},
        memory={"刚才": '{"rewritten": "刚才的结果", "base": "last_result"}'},
        sql={"客户总数": "SELECT COUNT(*) FROM 客户案例"},
        answer={"刚才": "上轮结果里有 3 家客户"},
    )
    sid = "s-last-result"
    # 第 1 轮：SQL 路径真实执行，结果暂存为跨轮状态
    out1 = rag.ask("客户总数是多少？", session_id=sid, llm=llm,
                   retriever=_fake_retriever(), db=test_db)
    assert "3" in out1
    assert rag.sessions[sid]["last_result"].startswith("[数据库查询]")
    # 第 2 轮：引用上轮结果 → 路由器判 memory → base=last_result → 直接回答
    out2 = rag.ask("刚才结果里有多少家客户？", session_id=sid, llm=llm,
                   retriever=_fake_retriever(), db=test_db)
    assert "3" in out2
    # 调用序列：第 1 轮 路由器+SQL 生成；第 2 轮 路由器+补全器+直答（无 SQL 生成、无向量检索）
    assert len(llm.calls) == 5


def test_memory_plan_parse_fallback():
    """补全器输出无法解析 → 降级走完整路由（路由器再判一次 + 向量链），不崩溃。"""
    llm = MemoryFakeLLM(
        router={"李云龙": "vector", "她们": "memory"},  # 降级后 _run_route 再判仍 memory → 走向量链
        memory={"她们": "抱歉，我不知道"},  # 不是 JSON → 解析失败
        answer={"李云龙": "李云龙的两位妻子是秀芹和田雨",
                "她们": "兜底回答：向量链结果"},
    )
    sid = "s-fallback"
    rag.ask("李云龙的妻子是谁？", session_id=sid, llm=llm, retriever=_fake_retriever())
    out = rag.ask("那她们结局如何？", session_id=sid, llm=llm, retriever=_fake_retriever())
    assert out == "兜底回答：向量链结果"
    # 第 2 轮调用：路由器 + 补全器 + 降级路由器 + 向量问答
    assert len(llm.calls) == 6


def test_ask_without_session_stays_legacy():
    """不传 session_id → 不写会话存储，保持 V10 无记忆行为。"""
    llm = MemoryFakeLLM(router={"李云龙": "vector"}, answer={"李云龙": "回答"})
    rag.ask("李云龙是谁？", llm=llm, retriever=_fake_retriever())
    assert rag.sessions == {}
