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

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import uvicorn  # 供 python app.py 直接启动

# 启动时一次性初始化（加载索引、建 SQLite 等）；
# rag.py 顶层无 input()/sys.exit 等交互副作用，可安全 import
from rag import ask

app = FastAPI(title="RAG 知识库问答", version="1.0.0")


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


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
    """核心接口：问题交给 ask() 门面，路由器自动分流 vector / sql。"""
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")
    try:
        answer = ask(question)
    except Exception as e:
        # LLM 余额不足（402）、网络异常等：返回友好提示，而不是 500 堆栈
        raise HTTPException(status_code=500, detail=f"服务暂时不可用：{e}")
    return AskResponse(answer=answer)


if __name__ == "__main__":
    # python app.py 直接启动；--host 0.0.0.0 支持同一局域网访问
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
