"""FastAPI Web 服务：把 rag.py 的问答封装成 HTTP 接口。

- GET  /        返回前端页面（templates/index.html）
- POST /ask     接收 {"question": "..."}，返回 {"answer": "..."}
- GET  /health  健康检查（局域网设备可先访问它确认服务在线）

启动（支持局域网访问）：
    python app.py                        # 直接启动，端口默认 8000
    python -m uvicorn app:app --host 0.0.0.0 --port 8000   # 等价

说明：
- --host 0.0.0.0 监听所有网卡：本机用 http://127.0.0.1:8000，
  同一局域网设备用 http://<本机局域网IP>:8000
- 启动时执行一次初始化（加载文档/建索引/建 SQLite），之后每个请求复用
"""

import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import uvicorn  # 供 python app.py 直接启动

# rag.py 顶层零副作用（init() 化后 import 不建索引）；启动时在 lifespan 中显式 init()
from rag import ask, init


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时一次性初始化（加载索引、建 SQLite 等）；uvicorn 启动时自动执行。
    测试用 TestClient 不进入 lifespan 时不会初始化，可测 /health 等无依赖接口。"""
    init()
    yield


app = FastAPI(title="RAG 知识库问答", version="1.0.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str
    session_id: str | None = None  # 可选：前端生成的会话 ID（多轮记忆用）


class AskResponse(BaseModel):
    answer: str
    session_id: str  # 回传会话 ID：首次未传时由后端生成，前端沿用实现多轮记忆


def _render_index() -> str:
    """读取前端页面（内嵌 CSS/JS，无需静态文件服务）。"""
    path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    with open(path, encoding="utf-8") as f:
        return f.read()


@app.get("/", response_class=HTMLResponse)
def index():
    """返回前端页面。"""
    return _render_index()


@app.get("/health")
def health():
    """健康检查：局域网设备访问 http://<IP>:8000/health 确认服务在线。"""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest):
    """核心接口：问题交给 ask() 门面，三路路由器自动分流 vector / sql / memory。"""
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")
    sid = req.session_id or uuid.uuid4().hex  # 首次提问由后端生成会话 ID
    try:
        answer = ask(question, session_id=sid)
    except Exception as e:
        # LLM 余额不足（402）、网络异常等：返回友好提示，而不是 500 堆栈
        raise HTTPException(status_code=500, detail=f"服务暂时不可用：{e}")
    return AskResponse(answer=answer, session_id=sid)


if __name__ == "__main__":
    # python app.py 直接启动；--host 0.0.0.0 支持同一局域网访问
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
