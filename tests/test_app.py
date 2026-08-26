"""app.py 接口测试：/health、/ask 空问题 400、首页渲染。

TestClient 不进入 with 块时 lifespan 不执行 → 不触发 rag.init()，
这三个用例零 LLM、零索引构建，秒级完成。
"""

from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_ask_empty_question_returns_400():
    r = client.post("/ask", json={"question": "   "})
    assert r.status_code == 400


def test_index_page_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "<html" in r.text.lower()
