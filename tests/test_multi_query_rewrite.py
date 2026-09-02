"""多查询改写的回归测试（V11.1 召回修复固化）。

背景：V11.0 实测发现"田雨最后的结局"答不出——抽象问法（结局）vs 原文具体写法
（割腕自杀）存在语义鸿沟，子查询全用抽象词导致 BM25 零匹配、割腕 chunk 在
RRF 粗召回被截断（recall_top_n=15 时排 25）。V11.1 修复：
  1. 机制版改写 prompt：抽象问法必须具象化（联想具体词并分散到不同子查询）、
     至少一路保留原问题关键词、专有名词原样；
  2. recall_top_n 15→30、per_query_top_n 3→5（粗召回兜住罕见词 chunk）。

本文件不调真实 API，用假 LLM 断言 prompt 规则与默认参数，防止这些修复被
无意改回去（真实召回已用离线探针验证：recall_top_n=30 后「田雨 结局 自杀」
能召回"田雨在狱中割腕自杀"chunk）。
"""

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.retrievers import BaseRetriever

from rag import MultiQueryRetriever


class _EchoLLM:
    """记录收到的 prompt，返回预设响应（模拟 llm.invoke）。"""

    def __init__(self, response):
        self.response = response
        self.last_prompt = None

    def invoke(self, prompt):
        self.last_prompt = prompt
        return AIMessage(content=self.response)


class _FakeRetriever(BaseRetriever):
    """最小假检索器：只满足 BaseRetriever 协议，不实际检索。"""

    def _get_relevant_documents(self, query, *, run_manager=None):
        return [Document(page_content="占位", metadata={"source": "fake.txt"})]


def _make(recall=30, per_query=5, top=8):
    return MultiQueryRetriever(
        base_retriever=_FakeRetriever(),
        llm=_EchoLLM(response='["子查询1", "子查询2"]'),
        recall_top_n=recall,
        per_query_top_n=per_query,
        top_n=top,
    )


# ---------- 1. 机制版 prompt 规则 ----------

def test_prompt_has_abstraction_rules():
    """prompt 必须含「抽象问法具象化」机制（V11.1 核心修复，防改回补丁版）。"""
    llm = _EchoLLM(response='["a"]')
    r = MultiQueryRetriever(base_retriever=_FakeRetriever(), llm=llm)
    r._generate_sub_queries("田雨最后的结局")
    p = llm.last_prompt
    for kw in ["抽象问法必须具象化", "具体词", "分散到不同子查询"]:
        assert kw in p, f"prompt 缺少机制版规则: {kw}"


def test_prompt_death_direction_rule():
    """死亡类抽象词必须覆盖「正常死亡+非正常死亡」两个方向（V11.1.1 修复）。

    实测坑：只让 LLM 自由联想具体词，它会随机给出"去世/病故"而漏掉"自杀"——
    「田雨和秀芹的结局」那次 5 路子查询全没带"自杀"，割腕自杀 chunk 一路未召回，
    答不出田雨结局。方向规则把"自杀/牺牲/被杀"从碰运气变成必须覆盖。
    """
    llm = _EchoLLM(response='["a"]')
    r = MultiQueryRetriever(base_retriever=_FakeRetriever(), llm=llm)
    r._generate_sub_queries("田雨最后的结局")
    p = llm.last_prompt
    for kw in ["正常死亡", "非正常死亡", "自杀", "牺牲", "被杀"]:
        assert kw in p, f"prompt 缺少死亡方向规则: {kw}"


def test_prompt_general_direction_rule():
    """方向拆解必须对所有抽象词生效，不局限于死亡类（V11.1.2 修复）。

    用户质疑：死亡方向规则只约束「结局/命运」这类词，其他抽象词（婚姻/事业/战斗结果）
    遇到同类 LLM 随机性漏词怎么办？→ 第 4 条升级为通用「方向拆解」：任何抽象词
    都要先拆 2-3 个语义方向、各成一路；死亡类降级为其中的「重要案例」。
    断言 prompt 必须含：方向拆解机制 + 非死亡类方向示例（婚姻/事业/战斗结果）+ 防词表化。
    """
    llm = _EchoLLM(response='["a"]')
    r = MultiQueryRetriever(base_retriever=_FakeRetriever(), llm=llm)
    r._generate_sub_queries("李云龙和田雨的婚姻怎么样")
    p = llm.last_prompt
    for kw in ["方向拆解", "婚姻", "事业", "战斗结果"]:
        assert kw in p, f"prompt 缺少通用方向拆解规则: {kw}"
    # 方向示例必须成对出现（拆成不同方向），且声明"不局限于本示例"防词表化
    assert ("结婚" in p and "离婚" in p) or ("升迁" in p and "贬职" in p) or ("胜利" in p and "战败" in p), \
        "prompt 必须给出非死亡类抽象词的方向拆解示例（两两成对）"
    assert "不局限于本示例" in p, "prompt 必须声明方向自由拆解，防止方向表被写死成词表"


def test_prompt_keeps_original_keywords_rule():
    """必须保留「至少一路用原问题原始关键词」规则，防止改写丢失原文信息。"""
    llm = _EchoLLM(response='["a"]')
    r = MultiQueryRetriever(base_retriever=_FakeRetriever(), llm=llm)
    r._generate_sub_queries("田雨最后的结局")
    assert "原始关键词" in llm.last_prompt


def test_prompt_keeps_proper_noun_rule():
    """专有名词必须原样带上（人名/地名/机构名）。"""
    llm = _EchoLLM(response='["a"]')
    r = MultiQueryRetriever(base_retriever=_FakeRetriever(), llm=llm)
    r._generate_sub_queries("李云龙和楚云飞什么关系")
    assert "专有名词" in llm.last_prompt


# ---------- 2. 召回参数（V11.1：15→30 / 3→5） ----------

def test_default_recall_params_not_regressed():
    """默认召回参数必须兜住罕见词 chunk（实测「田雨 结局 自杀」RRF 排 25）。

    V11.1.3 补：top_n 6→8——「田雨死了吗」割腕 chunk 精排第 2 但只在一路出现，
    RRF 融合分 1/62 排第 7，top6 被"多路同现"常规 chunk 占满而截断。
    """
    r = _make()
    assert r.recall_top_n == 30, "recall_top_n 被改小会导致罕见词 chunk 在粗召回被截断"
    assert r.per_query_top_n == 5, "per_query_top_n 被改小会压缩每路精排保留"
    assert r.top_n == 8, "top_n 被改小会让'只在一路出现但高度相关'的 chunk 被融合截断"


# ---------- 3. 解析降级 ----------

def test_parse_failure_falls_back_to_original_query():
    """LLM 返回无法解析的文本时，降级为原问题本身（不能返回空）。"""
    for garbage in ["这不是JSON", "{}", "[]", ""]:
        llm = _EchoLLM(response=garbage)
        r = MultiQueryRetriever(base_retriever=_FakeRetriever(), llm=llm)
        assert r._generate_sub_queries("田雨最后的结局") == ["田雨最后的结局"]


def test_parse_extracts_json_from_noise():
    """LLM 在杂讯里混入 JSON 数组时，应能正则抓出。"""
    llm = _EchoLLM(response="好的，这是子查询：[\"田雨 自杀\", \"田雨 去世\"] 请查收")
    r = MultiQueryRetriever(base_retriever=_FakeRetriever(), llm=llm)
    assert r._generate_sub_queries("田雨最后的结局") == ["田雨 自杀", "田雨 去世"]


# ---------- 4. 唤醒式改造（方案二）：覆盖任意非枚举抽象类型 ----------

def test_prompt_uses_readers_imagination_for_any_abstract_word():
    """抽象问法的具象化必须走"唤醒式"而非"枚举式"（方案二，2026-09）。

    背景：V11.1 的"方向拆解"靠 prompt 里枚举抽象词类型（结局/婚姻/事业/战斗结果）去引导
    LLM，导致只对枚举过的类型可靠——"心情/爱情/去向"等未枚举的抽象说法，LLM 未必触发拆解。
    方案二改法：不再问"这算不算需要拆的抽象词"（分类，靠词表），而是让 LLM 像读者一样
    "预想原文会用什么具体词/情节来表达这个抽象概念"（生成/联想，模型天生擅长），从而覆盖
    任意抽象类型而不依赖穷举。断言：
    1. prompt 用任务式措辞（预想原文写法）唤醒联想，而不是给一份可穷举的死词表；
    2. 示例同时覆盖心情/爱情/去向等非枚举类别，但声明"不局限于本示例"防词表化。
    """
    llm = _EchoLLM(response='["a"]')
    r = MultiQueryRetriever(base_retriever=_FakeRetriever(), llm=llm)
    r._generate_sub_queries("李云龙当时的内心情绪怎么样")
    p = llm.last_prompt
    # 唤醒式措辞：预想原文写法（模型用世界知识联想，而非查词表）
    assert "预想原文" in p, "prompt 必须用'预想原文写法'唤醒模型联想，而非枚举分类"
    assert "读者" in p, "prompt 应引导模型像读者一样预想作品里抽象概念的具体写法"
    # 示例覆盖非枚举抽象类别（心情/爱情/去向），证明不只面向结局/死亡类
    for kw in ["心情", "爱情", "去向"]:
        assert kw in p, f"prompt 必须给出非枚举抽象类别的示例引导: {kw}"
    # 防词表化：示例只是示范，远非全部
    assert "不局限于本示例" in p, "prompt 必须声明预想随问题自由展开，防止示例被当成词表"


def test_prompt_death_class_remains_strong_anchor():
    """死亡/生命状态类仍是强锚定（方案二保留，防回归）。

    方案二的主体改为唤醒式后，仍保留死亡类的"正常+非正常死亡"双向强约束——这是实测
    最易漏的高价值案例（「田雨死了吗」漏割腕 chunk 的历史教训），不能因改唤醒式而放松。
    """
    llm = _EchoLLM(response='["a"]')
    r = MultiQueryRetriever(base_retriever=_FakeRetriever(), llm=llm)
    r._generate_sub_queries("田雨最后的结局")
    p = llm.last_prompt
    for kw in ["正常死亡", "非正常死亡", "自杀", "牺牲", "被杀"]:
        assert kw in p, f"死亡类强锚被放松: {kw}"
