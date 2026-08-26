"""第 6 步（V10.0）：FastAPI 封装 + 极简深色前端。

把 rag.py 的 CLI 问答包装成 Web 服务：
- GET  /        返回前端页面（templates/index.html）
- POST /ask     接收 {"question": "..."}，返回 {"answer": "..."}
- GET  /health  健康检查（局域网设备可先访问它确认服务在线）

启动命令（支持局域网访问）：
    python app.py                      # 直接启动（等价于下面一行，端口可改）
    python -m uvicorn app:app --host 0.0.0.0 --port 8000

说明：
- --host 0.0.0.0 让同一局域网内的手机/其他主机也能访问；
  本机访问用 http://127.0.0.1:8000，其他设备用 http://<本机局域网IP>:8000
- from rag import ask 会在服务启动时触发一次初始化
  （加载文档/构建索引/导入 SQLite），之后每个 /ask 请求复用，
  这正是第 5 步 ask() 门面模式的用意
"""

import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# 支持 python app.py 直接启动（等价于 uvicorn app:app --host 0.0.0.0 --port 8000）
import uvicorn

# 启动时一次性初始化（加载索引、建 SQLite 等），import 安全：
# rag.py 顶层没有 input()/sys.exit 等交互副作用
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
