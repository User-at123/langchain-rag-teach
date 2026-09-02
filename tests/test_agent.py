"""Agent 工具化（第 10 步）链路测试：注入 FakeLLM 验证多跳自主编排。

AgentFakeLLM 在 MemoryFakeLLM 基础上新增"自主 Agent"分桶（system 含"自主 Agent"），
并按调用顺序依次吐出脚本里的 Action / Final Answer，模拟 ReAct 循环。
关键差异：ReAct 循环直接把 messages 列表传给 llm.invoke()，所以 AgentFakeLLM
必须兼容 list 输入（普通 PromptValue 输入走 .messages 属性，两者都支持）。

覆盖点：
1. 多跳编排：路由器判 agent → Agent 先调 query_sql → 基于 Observation 输出 Final Answer
2. retrieve_vector 工具：Agent 调文档检索工具拿到回答
3. read_memory 工具：Agent 读会话历史/上轮结果后回答
4. 超过步数上限未收敛 → 回退快路径（_run_route），不崩溃
5. 输出无法解析（无 Action 无 Final Answer）→ 回退快路径，不崩溃
"""

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.retrievers import BaseRetriever

import rag


class AgentFakeLLM:
    """5 桶假 LLM：router / agent（按脚本顺序）/ memory / sql / answer。

    - agent: 按 self.script_agent 顺序返回（可模拟多步循环，第 N 步对应第 N 个元素）；
      脚本用尽后默认输出 Final Answer 兜底。
    - 兼容 list 输入：ReAct 循环直接传 messages 列表给 invoke，普通链传 PromptValue。
    """

    def __init__(self, router=None, agent=None, memory=None, sql=None, answer=None):
        self.script_router = router or {}
        self.script_agent = agent or []
        self.script_memory = memory or {}
        self.script_sql = sql or {}
        self.script_answer = answer or {}
        self.agent_idx = 0
        self.calls = []

    def __call__(self, input):
        # ReAct 循环传 list；其余传 PromptValue（带 .messages）
        if isinstance(input, list):
            msgs = input
        else:
            msgs = input.messages if hasattr(input, "messages") else []
        system = next((m.content for m in msgs if getattr(m, "type", "") == "system"), "")
        # agent 循环里取"最后一条 human"（最新的 Observation 或原问题），其余取第一条 human
        human_list = [m.content for m in msgs if getattr(m, "type", "") == "human"]
        human = human_list[-1] if human_list else ""
        self.calls.append(human)
        if "自主 Agent" in system:
            if self.agent_idx < len(self.script_agent):
                out = self.script_agent[self.agent_idx]
                self.agent_idx += 1
                return AIMessage(content=out)
            return AIMessage(content="Final Answer: （Agent 默认收敛）")
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
    """固定返回预设文档列表的假检索器（与 test_memory 同实现）。"""

    docs: list

    def _get_relevant_documents(self, query: str, *, run_manager=None):
        return self.docs


@pytest.fixture
def test_db(tmp_path):
    """临时 SQLiteDb（1 表 3 行），供 query_sql 工具真实执行结果断言。"""
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

    inst = rag.SQLiteDb(db_path=str(tmp_path / "test.db"), docs_dir=str(docs_dir))
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


def test_agent_multi_hop_query_sql_then_answer(test_db):
    """多跳编排：路由器判 agent → Agent 调 query_sql（真实执行）→ 基于 Observation 收尾。"""
    llm = AgentFakeLLM(
        router={"先统计": "agent"},
        agent=[
            "Action: query_sql\nAction Input: 各行业客户数",
            "Final Answer: 统计完成，金融、航空、农业各 1 家。",
        ],
        sql={"各行业客户数": "SELECT 行业, COUNT(*) FROM 客户案例 GROUP BY 行业"},
    )
    out = rag.ask("先统计各行业客户数，再总结结果", session_id="s-1", llm=llm,
                  retriever=_fake_retriever(), db=test_db)
    assert out == "统计完成，金融、航空、农业各 1 家。"
    # Agent 用了 2 步（Action → Final Answer）
    assert llm.agent_idx == 2
    # Agent 循环内 query_sql 结果同步进了跨轮状态，后续 read_memory 可读
    assert rag.sessions["s-1"]["last_result"].startswith("[数据库查询]")
    # 会话已记录本轮问答
    assert rag.sessions["s-1"]["history"][-1] == ("assistant", out)


def test_agent_tool_retrieve_vector():
    """Agent 调 retrieve_vector 工具：走文档检索链拿回答。"""
    llm = AgentFakeLLM(
        router={"介绍": "agent"},
        agent=[
            "Action: retrieve_vector\nAction Input: 零售行业客户特点",
            "Final Answer: 零售客户以连锁经营为主。",
        ],
        answer={"零售行业客户特点": "零售客户以连锁经营为主"},
    )
    out = rag.ask("介绍一下零售行业客户特点", session_id="s-2", llm=llm,
                  retriever=_fake_retriever())
    assert out == "零售客户以连锁经营为主。"
    assert llm.agent_idx == 2


def test_agent_tool_read_memory():
    """Agent 调 read_memory 工具：读到会话历史后结合上下文回答。"""
    llm = AgentFakeLLM(
        router={"李云龙": "vector", "结合": "agent"},
        agent=[
            "Action: read_memory\nAction Input: ",
            "Final Answer: 结合历史，李云龙的妻子是秀芹和田雨。",
        ],
        answer={"李云龙": "李云龙的两位妻子是秀芹和田雨"},
    )
    sid = "s-3"
    # 第 1 轮普通问答建立历史
    rag.ask("李云龙的妻子是谁？", session_id=sid, llm=llm, retriever=_fake_retriever())
    # 第 2 轮 agent 问题：先读历史再回答
    out = rag.ask("结合对话历史回答", session_id=sid, llm=llm, retriever=_fake_retriever())
    assert "秀芹和田雨" in out
    assert llm.agent_idx == 2


def test_agent_exceeds_steps_falls_back():
    """Agent 一直输出 Action 不收敛 → 超过步数上限 → 回退快路径，不崩溃。"""
    llm = AgentFakeLLM(
        router={"先统计": "agent"},  # 回退时 _run_route 再判仍 agent → 落到向量链兜底
        agent=["Action: query_sql\nAction Input: 各行业客户数"] * (rag.MAX_AGENT_STEPS + 2),
        answer={"先统计": "兜底向量回答"},
    )
    out = rag.ask("先统计各行业客户数", session_id="s-4", llm=llm,
                  retriever=_fake_retriever())
    assert out == "兜底向量回答"
    # Agent 只跑了 MAX_AGENT_STEPS 步就放弃（第 5 个脚本元素未消费）
    assert llm.agent_idx == rag.MAX_AGENT_STEPS


def test_agent_unparsable_falls_back():
    """Agent 输出既无 Action 也无 Final Answer → 立即回退快路径，不崩溃。"""
    llm = AgentFakeLLM(
        router={"先统计": "agent"},
        agent=["抱歉，我不知道怎么做"],
        answer={"先统计": "兜底向量回答"},
    )
    out = rag.ask("先统计各行业客户数", session_id="s-5", llm=llm,
                  retriever=_fake_retriever())
    assert out == "兜底向量回答"
    assert llm.agent_idx == 1  # 只消费了 1 个脚本元素


def test_agent_without_session_read_memory_returns_none_context():
    """无 session_id 时 agent 也能跑（read_memory 返回"无会话上下文"，不报错）。"""
    llm = AgentFakeLLM(
        router={"先统计": "agent"},
        agent=[
            "Action: read_memory\nAction Input: ",
            "Final Answer: 没有可用上下文，我直接回答。",
        ],
    )
    out = rag.ask("先统计各行业客户数", llm=llm, retriever=_fake_retriever())
    assert out == "没有可用上下文，我直接回答。"
    assert rag.sessions == {}  # 不写会话
