# LangChain 教学项目 · 学习思维导图（版本演进全记录）

> 本项目如何从一行 `prompt | llm` 逐步长成一个多格式 RAG 系统。
> 每个版本：改了什么 / 用到什么工具 / 踩过什么坑 / 加了什么依赖。
> 配合 `improvements.txt`（未来规划）一起看效果最好。

---

## 一、项目整体脉络（思维导图主干）

```
LangChain 教学项目
├── V1 雏形：最简问答（main.py）
│   ├── 目标：跑通「提示词 + 模型」的一问一答
│   ├── 核心概念：prompt | llm 链式调用
│   └── 依赖：langchain / langchain-openai / python-dotenv
│
├── V2 基础 RAG：检索增强生成（rag.py + knowledge_base.txt）
│   ├── 新增流程：加载 → 切分 → 嵌入 → 检索 → 生成
│   ├── 向量库：InMemoryVectorStore（内存版，重启即丢）
│   └── 依赖新增：langchain-text-splitters / langchain-huggingface / sentence-transformers
│
├── V3 打通 DeepSeek + 本地嵌入
│   ├── ChatOpenAI + base_url = https://api.deepseek.com/v1（OpenAI 兼容接口）
│   ├── 嵌入模型：BAAI/bge-small-zh-v1.5（DeepSeek 没有 Embedding API）
│   └── 坑：HuggingFace 下载报 SSL 错 → .env + HF_ENDPOINT 镜像 + USE_INSECURE_SSL
│
├── V4 修复 LCEL 并行链报错
│   ├── 坑：retriever 拿到整个 dict → 'dict' object has no attribute 'replace'
│   └── 修：用 (lambda x: x["question"]) | retriever 先取出字符串
│
├── V5 Chroma 持久化（git: 5d73e18）
│   ├── InMemoryVectorStore → Chroma
│   ├── 索引落盘 ./chroma_db，第二次运行秒加载
│   └── 注意：改知识库内容后必须删 chroma_db 重建索引
│
├── V6 多格式加载（git: 6dc301f）
│   ├── docs/ 目录 + DirectoryLoader + loader_map 按扩展名路由
│   ├── 支持 .txt / .md / .pdf / .docx / .csv
│   └── 依赖新增：pypdf / docx2txt / langchain-community
│
├── V7 兼容修复（当前工作区，未提交）
│   ├── 坑：langchain-community 0.4.2 移除了 loader_map 参数
│   ├── 修：改为 os.walk 手动遍历 + LOADER_MAP 路由表
│   └── 坑：Windows 默认 GBK 解码中文报错 → 显式 encoding="utf-8"
│
└── V8+ 未来规划（见 improvements.txt）
    ├── 2.5 切分差异化（按格式选切分器）/ 原生 .xlsx
    ├── 3   BM25 混合检索 + bge-reranker 重排序
    ├── 4   视频字幕 srt / 语音转写接入
    ├── 5   assets/logo 素材目录规范
    ├── 6   FastAPI 封装 + 前端页面
    └── 7   海报 / 文案生成（logo 程序化叠加）
```

---

## 二、各版本详解

### V1 雏形：最简问答（main.py）

**改动内容**
- 创建 `main.py`，用 LangChain 最核心的两步完成问答：定义提示词、接上模型
- 整个程序就一条链：`chain = prompt | llm`
- 配套 `requirements.txt`、`.env.example`、`README.md`（教学三件套）

**使用的工具**
- `ChatPromptTemplate`：提示词模板，`{question}` 占位符
- `ChatOpenAI`：语言模型对象，`temperature=0.7` 控制随机性
- `load_dotenv()`：从 `.env` 读配置，API Key 不写死在代码里
- 管道符 `|`：LangChain 的核心语法，把组件连成链

**注意事项**
- `.env.example` 只是模板，必须复制成 `.env` 并填真实 Key，否则模型调用报错
- DeepSeek 兼容 OpenAI 接口，所以用 `ChatOpenAI`，只需改 `base_url` 和模型名

---

### V2 基础 RAG：检索增强生成（rag.py + knowledge_base.txt）

**改动内容**
- 新增 `rag.py`：在"提问 → 回答"之间插入知识库检索环节
- 新增 `knowledge_base.txt`：单文件知识库（示例内容：星辰科技公司资料）
- 与 V1 的区别一句话：**V1 靠模型"背"知识，V2 让模型"查"知识**

**新增流程（RAG 五步）**
```
加载文档 → 切分 → 嵌入 → 检索 → 生成
  open()    splitter  embeddings  retriever  prompt | llm
```

**使用的工具**
- `RecursiveCharacterTextSplitter`：递归字符切分器，`chunk_size=200, chunk_overlap=50`
- `HuggingFaceEmbeddings`：本地开源嵌入模型，把文本转成向量
- `InMemoryVectorStore`：内存向量库（教学用，重启即消失）
- `ChatPromptTemplate`：system 里塞【参考资料】`{context}`，让模型先查后答

**新增依赖**
- `langchain-text-splitters`（切分）、`langchain-huggingface`（本地嵌入）、`sentence-transformers`（嵌入模型底层）

**注意事项**
- `split_documents` 保留来源 metadata；`format_docs` 标注来源
- DeepSeek **没有** Embedding API，所以嵌入用本地开源模型 `BAAI/bge-small-zh-v1.5`（首次运行自动下载约 100MB）

---

### V3 打通 DeepSeek + 本地嵌入

**改动内容**
- `ChatOpenAI` 配置 `base_url=https://api.deepseek.com/v1`、`model=deepseek-chat`
- 确认嵌入方案：本地 `BAAI/bge-small-zh-v1.5`（中文效果好、体积小、离线可用、不耗 API 额度）

**踩过的坑（重点）**
- **HuggingFace 下载 SSL 报错** `SSL: CERTIFICATE_VERIFY_FAILED`，两层原因：
  1. 根因：`.env` 文件不存在，HF 相关环境变量根本没读到 → 必须先 `load_dotenv()` 且 `.env` 要存在
  2. Windows/公司代理环境证书验证失败
- 双重解法（写在 `rag.py` 顶部，且**必须在 import huggingface 库之前**执行）：
  - `HF_ENDPOINT=https://hf-mirror.com`：走国内镜像加速/修复下载
  - `USE_INSECURE_SSL=1`：跳过证书验证（仅教学用）

**注意事项**
- 环境变量只在 import 时读取一次，所以 `.env` 加载代码必须放在 import huggingface 之前

---

### V4 修复 LCEL 并行链报错

**改动内容**
- 并行分支写法：`{"context": ..., "question": ...} | prompt | llm`
- 修正检索输入

**踩过的坑（重点）**
- 报错：`AttributeError: 'dict' object has no attribute 'replace'`
- 原因：retriever 只接受**字符串**，但并行链里它收到的是整个输入 dict
- 修复：`"context": (lambda x: x["question"]) | retriever | format_docs`
  - `lambda x: x["question"]` 先从 dict 里取出问题字符串
  - 再交给 retriever 检索
  - 最后 `format_docs` 拼成带【来源】标注的文本

---

### V5 Chroma 持久化（git: 5d73e18）

**改动内容**
- 向量库从 `InMemoryVectorStore`（内存）换成 `Chroma`（磁盘持久化）
- 解决"内存向量库重启就消失"的痛点

**核心逻辑**
```python
if os.path.exists(CHROMA_DIR):
    vector_store = Chroma(embedding_function=embeddings,
                          persist_directory=CHROMA_DIR)      # 已有索引，直接加载
else:
    vector_store = Chroma.from_documents(chunks, embedding=embeddings,
                                         persist_directory=CHROMA_DIR)  # 首次建库
```

**新增依赖**
- `langchain-chroma`（Chroma 的 LangChain 封装）

**注意事项（重点）**
- Chroma 索引**不会自动感知**知识库文件变化：改了 `knowledge_base.txt` 或 `docs/` 后必须删索引重建
  - cmd：`rmdir /s /q chroma_db`
  - PowerShell：`Remove-Item -Recurse -Force chroma_db`（`rmdir /s /q` 是 cmd 语法，PowerShell 不认！）

---

### V6 多格式加载（git: 6dc301f）

**改动内容**
- 新增 `docs/` 多格式知识库目录，`load_documents()` 按扩展名路由到对应 Loader
- 加载规则：`docs/` 有文件 → 加载 docs/ 全部；`docs/` 为空 → 回退 `knowledge_base.txt`
- 回答时通过 `metadata["source"]` 标注来源文件名

**使用的工具**
- `DirectoryLoader`：目录批量加载器，`glob="**/*"` 递归匹配
- 格式路由表：`.txt/.md → TextLoader`、`.pdf → PyPDFLoader`、`.docx → Docx2txtLoader`、`.csv → CSVLoader`

**新增依赖**
- `langchain-community`（文档加载器集合）、`pypdf`（PDF 解析）、`docx2txt`（Word 解析）

**已知边界**
- `.xlsx` 不支持，需另存为 CSV（原生支持留待 2.5 步）
- 切分策略所有格式统一，未按格式区分（留待 2.5 步）
- `docs/` 与 `knowledge_base.txt` 是"二选一回退"关系，不是合并：docs/ 有文件时 knowledge_base.txt 被忽略

---

### V7 兼容修复（当前工作区，未提交）

**改动内容**
- `load_documents()` 重写：`DirectoryLoader + loader_map` → `os.walk` 手动遍历 + `LOADER_MAP` 路由表

**踩过的坑（重点）**
- **坑 1：新版 langchain-community（0.4.2）移除了 `loader_map` 参数**
  - 报错：`TypeError: DirectoryLoader.__init__() got an unexpected keyword argument 'loader_map'`
  - 原因：新版本 DirectoryLoader 只支持单个默认 `loader_cls`，不能按扩展名路由
  - 修复：自己写循环 `for root, _, files in os.walk(DOCS_DIR)`，逐个文件取扩展名查路由表
  - 好处：不依赖框架特定版本的 API，代码更直白，更好教学
- **坑 2：Windows 默认 GBK 编码解码中文报错**
  - 报错：`UnicodeDecodeError: 'gbk' codec can't decode byte 0xac ...`
  - 原因：`sample.md` 是 UTF-8（含中文），Windows 上 Python 默认按 GBK 打开文本文件
  - 修复：`LOADER_MAP` 升级为 `(Loader 类, 参数)` 结构，文本类 Loader 显式传 `encoding="utf-8"`
  - PDF/Word 是二进制解析，无需编码参数

**当前 LOADER_MAP 结构**
```python
LOADER_MAP = {
    ".txt": (TextLoader, {"encoding": "utf-8"}),
    ".md":  (TextLoader, {"encoding": "utf-8"}),
    ".pdf": (PyPDFLoader, {}),
    ".docx": (Docx2txtLoader, {}),
    ".csv": (CSVLoader, {"encoding": "utf-8"}),
}
```

---

### V8+ 未来规划（详见 improvements.txt）

| 步骤 | 内容 | 关键工具 |
| --- | --- | --- |
| 2.5 | 按格式差异化切分 + 原生 .xlsx | MarkdownHeaderTextSplitter / pandas |
| 3 | BM25 混合检索 + 重排序 | langchain-retrievers / bge-reranker |
| 4 | 视频字幕 / 语音转写 | srt 解析 / faster-whisper |
| 5 | assets/logo 素材规范 | 本地目录 → 对象存储 |
| 6 | FastAPI 封装 + 前端 | FastAPI / 简单网页 |
| 7 | 海报 / 文案生成 | LLM 文案 + 程序化叠加 logo |

---

## 三、依赖包清单（当前 requirements.txt）

| 包 | 作用 | 对应功能 | 备注 |
| --- | --- | --- | --- |
| `langchain` | 核心框架 | 链式组合 `prompt \| llm` | 实际安装 1.3.16 |
| `langchain-openai` | OpenAI 兼容接口 | `ChatOpenAI` 调 DeepSeek | 只需改 base_url |
| `langchain-text-splitters` | 文本切分 | `RecursiveCharacterTextSplitter` | |
| `langchain-huggingface` | 本地嵌入 | `HuggingFaceEmbeddings` | DeepSeek 无 Embedding API |
| `langchain-chroma` | 向量库 | `Chroma` 持久化 | 索引存 ./chroma_db |
| `langchain-community` | 文档加载器 | Text/PyPDF/Docx2txt/CSVLoader | 0.4.2 起移除 loader_map！ |
| `pypdf` | PDF 解析 | `PyPDFLoader` | V6 新增 |
| `docx2txt` | Word 解析 | `Docx2txtLoader` | V6 新增 |
| `sentence-transformers` | 嵌入模型底层 | bge-small-zh 运行环境 | |
| `python-dotenv` | 环境变量 | `load_dotenv()` 读 .env | |

---

## 四、常用命令速查

```powershell
# 运行普通问答
python main.py

# 运行 RAG 问答（首次自动下载嵌入模型 + 建索引）
python rag.py

# 修改知识库后重建索引（PowerShell 语法！）
Remove-Item -Recurse -Force chroma_db

# 修改知识库后重建索引（cmd 语法）
rmdir /s /q chroma_db

# 安装依赖
pip install -r requirements.txt
```

---

## 五、踩坑记录汇总（按主题）

| 主题 | 现象 | 原因 | 解决 |
| --- | --- | --- | --- |
| HuggingFace 下载 | SSL 证书验证失败 | .env 缺失 + Windows 证书问题 | 建 .env；HF_ENDPOINT 镜像；USE_INSECURE_SSL=1 |
| LCEL 链 | 'dict' object has no attribute 'replace' | retriever 收到整个 dict | `(lambda x: x["question"]) \| retriever` |
| 版本兼容 | DirectoryLoader 不认 loader_map | langchain-community 0.4.2 移除该参数 | os.walk 手动遍历 + LOADER_MAP |
| 文件编码 | 'gbk' codec can't decode byte ... | Windows 默认 GBK 读 UTF-8 中文文件 | TextLoader/CSVLoader 显式 encoding="utf-8" |
| 索引陈旧 | 改了知识库回答还是旧的 | Chroma 持久化不感知源文件变化 | 删 chroma_db 重建 |
| PowerShell | rmdir /s /q 报错找不到参数 | /s /q 是 cmd 语法 | 用 Remove-Item -Recurse -Force |
| git push | SSL peer certificate 错误 | Windows 证书链问题 | `git config http.sslVerify false`（项目级） |

---

## 六、版本与 git 提交对照

> 教学启发：开发迭代和 git 提交不必一一对应——V1~V4 都是第一次提交前完成的，
> 第一次提交（0c8a6da）把雏形 + RAG + DeepSeek 适配 + 踩坑修复一起打包了。

| git 提交 | 说明 | 对应版本 |
| --- | --- | --- |
| `0c8a6da` | LangChain 入门教学项目：普通问答 + RAG | V1 ~ V4（合并提交） |
| `9f072ce` | 增加 improvements.txt 改进方向文档 | 文档 |
| `06e7b31` | 补充 logo 素材存储规范 | 文档 |
| `92caefa` | 数据存储的改进方向（切分指南等） | 文档 |
| `5d73e18` | 更换数据库为 chroma | V5 |
| `6dc301f` | 多格式加载 .txt/.md/.pdf/.docx/.csv | V6 |
| （未提交） | os.walk 手动遍历 + UTF-8 编码修复 + docs/ 示例 | V7 + docs/ |
