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
├── V7 兼容修复（已提交）
│   ├── 坑：langchain-community 0.4.2 移除了 loader_map 参数
│   ├── 修：改为 os.walk 手动遍历 + LOADER_MAP 路由表
│   └── 坑：Windows 默认 GBK 解码中文报错 → 显式 encoding="utf-8"
│
├── V7.5 切分差异化 + 原生 xlsx（已提交）
│   ├── 目标：不同格式的"语义单元"不同，切分也该按格式路由
│   ├── 实现：.md 按标题切（MarkdownHeaderTextSplitter）、.csv/.xlsx 按行切
│   ├── 原生 .xlsx：openpyxl 自写 load_xlsx（不用重量级 unstructured）
│   └── 新增依赖：openpyxl
│
├── V7.6 xlsx 表头拼接修复（已提交）
│   ├── 坑：表格按行切后表头行与数据行分离，问"客户行业"命中表头拿不到值
│   └── 修：load_xlsx 把第 1 行表头拼进每个数据行 → "客户行业: 金融 | ..."
│
├── V7.7 检索参数调优（已提交）
│   ├── 坑：k=2 太小，问"所有客户行业"只召回 1 条，答不全
│   ├── 试错：MMR 检索反而把结构相似的客户案例行当"重复"丢弃
│   └── 定稿：普通相似度检索 + k=6，3 条案例行全进候选，模型能汇总出完整列表
│
├── V7.8 认知边界记录（纯文档，无代码改动）
│   ├── k=6 只是当前 10 个片段知识库的调优值，不是通用解
│   ├── "列举/聚合"类问题 ≠ Top-k 检索问题（见核心概念 4）
│   ├── 三档正解：整表引用 → 分批聚合 → Text-to-SQL（生产级）
│   └── 补充：引用"上一轮导出结果"的推理 ≠ 检索（走 memory，见核心概念 5 / 第 9 步）
│
├── V7.9 混合检索 + 重排序（第三阶段，待提交）
│   ├── 单一向量检索 → BM25 + 向量 + RRF 融合 + bge-reranker 精排
│   ├── k 的职责：从"回答视野"（V7.7）降级为"粗筛漏斗"，精排后定视野
│   ├── 坑：langchain-retrievers 包装不上 → 手写 RRFRetriever / RerankerRetriever
│   └── 坑：BM25 中文必须接 jieba 分词（preprocess_func 参数）
│
└── V8+ 未来规划（见 improvements.txt）
    ├── 3   已完成（V7.9）：BM25 混合检索 + bge-reranker 重排序
    ├── 4   视频字幕 srt / 语音转写接入
    ├── 5   Text-to-SQL + 路由器（问题分流：向量 vs SQL；SQLite 起步 → MySQL）
    ├── 6   FastAPI 封装 + 前端页面（封装完整能力，含 SQL 路由）
    ├── 7   assets/logo 素材目录规范
    ├── 8   海报 / 文案生成（logo 程序化叠加）
    ├── 9   多轮记忆 + 跨轮状态（路由三路：vector / sql / memory）
    └── 10  Agent 工具化（query_sql / retrieve_vector / read_memory 自主编排）
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

### V7 兼容修复（已提交）

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

### V7.5 切分差异化 + 原生 xlsx（第二阶段，已提交）

**改动内容**
- 新增 `split_documents_by_format()`：切分阶段也按格式路由（V7 只是加载按格式路由，切分仍一刀切）
- 新增 `load_xlsx()`：原生 `.xlsx` 支持（此前需手动另存为 CSV）
- 文件头升级为"第二阶段：多格式加载 + 按格式差异化切分"

**切分路由规则**
```
.md        → MarkdownHeaderTextSplitter：按 #/##/### 标题层级切，章节不拆散
.csv/.xlsx → 表格语义在"行"：加载时已按行拆好，切分阶段直接透传
.txt/.pdf/.docx → RecursiveCharacterTextSplitter（chunk_size=200, overlap=50，保留）
```

**关键实现细节（注意）**
- `MarkdownHeaderTextSplitter` 的输入是**字符串而非 Document**，切出的块没有 source，需手动补回 `metadata["source"]`
- `.xlsx` 不用 LangChain 的 `UnstructuredExcelLoader`（要装重量级 unstructured 包），改用 `openpyxl` 自写：每行一个 Document、单元格 `|` 连接、跳过全空行、source 标注「工作表名 + 行号」

**新增依赖**
- `openpyxl`（原生 Excel 解析）

**注意事项（验证方法）**
- 验证时避免用 `HuggingFaceEmbeddings(...)` 初始化——它联网访问 HF，网络不通会卡死报
  "连接方在一段时间后没有正确答复"。纯本地验证用 `chromadb.PersistentClient` 直接读索引内容即可

---

### V7.6 xlsx 表头拼接修复（第二阶段，已提交）

**改动内容**
- 修复 bug：问"客户行业是什么"只返回表头字段名、拿不到具体值
- `load_xlsx()` 升级：第 1 行视为表头（字段名），数据行拼接为 `字段名: 值 | 字段名: 值`，
  让每一行数据自带字段名；表头行不再单独入库
- 跳过全空行、跳过与表头重复的行；无表头的表（表头全空）退化为原 `|` 连接行为

**改造前后对比（以 sample.xlsx 客户案例表为例）**
```
改造前：表头行入库存为 "客户行业 | 客户名称 | 使用产品 | 合作年份"（没有值，检索命中它回答不了）
改造后：数据行入库为 "客户行业: 金融 | 客户名称: 华信银行 | 使用产品: 星辰问答企业版 | 合作年份: 2022"
```

**验证结果**
- 加载 7 个文档（md 1 + xlsx 6，表头不再占位置）、切分 10 个片段（md 4 + xlsx 6）
- 检索"客户行业是什么"命中的数据行自带字段名，回答正确："华信银行，所属行业：金融"

**注意事项**
- 加载/切分逻辑变化后必须删 `chroma_db` 重建索引才生效（本次已重建）

---

### V7.7 检索参数调优（第二阶段，已提交）

**改动内容**
- 修复 bug：问"所有客户的行业有哪些"，模型只答出 1 个客户（华信银行-金融）
- 根因：检索器 `k=2` 只返回最相关的 2 个片段，3 条客户案例行只有 1 条进候选，
  k 就是"回答的视野"
- 定稿：`vector_store.as_retriever(search_kwargs={"k": 6})`（普通相似度检索 + 更大的 k）

**试错记录（重要）**
- 先试了 MMR 检索（`search_type="mmr", k=3, fetch_k=20, lambda_mult=0.5`）：
  它在"相关 + 多样"之间平衡，结果反而把结构高度相似的 3 条客户案例行当成"重复"
  只保留 1 条，多出的名额给了 sample.md 里不相关的块 → 还是答不全
- 对比实测 k=6：普通检索按相似度排序，3 条案例行相关性都高，全部进候选 ✅
- 结论：MMR 的"多样性去重"适合**长文档中内容雷同**的场景；对**表格多行同类记录**
  是副作用（同类记录恰恰都要保留）。小知识库用普通相似度检索更直观

**验证结果**
- 问"所有客户的行业有哪些？" → 回答完整列出：华信银行（金融）、优品超市（零售）、
  明德学院（教育）
- 检索参数变化不需要重建 chroma_db（只影响查询，不影响索引内容）

---

### V7.8 认知边界记录（纯文档，无代码改动，已提交）

**记录内容（来自用户提问"一万个客户时 k=6 能保证吗？"）**
- 明确承认：k=6 只是当前 10 个片段知识库的调优值，**不是通用解**
- 一万客户时两种崩法：k 太小漏召回、k 调到 1 万则提示词 token 超限 + 噪声爆炸
- 本质认知："列举/聚合/统计"类问题需要**全量访问**，不属于 Top-k 检索的舒适区
- 三档正解：整表引用（几百行）→ 分批聚合（几千~几万行）→ Text-to-SQL（生产级 1 万+）
- 当前选择：暂不改代码，保留 k=6 为"事实查询"教学配置；数据量到级别后再升级

**为什么不做整表引用演示**：整表引用对几百行有效，但对一万行同样塞不进提示词，
治标不治本；真正支撑一万客户的是数据入库 + 问题转 SQL（结构化 RAG），
属于架构级升级，等数据量真实增长时再做（见核心概念 4）

**补充：引用前面结果的推理 ≠ 检索（多轮对话场景）**
- 场景：第 1 轮问"所有客户的信息"，走 SQL 全量导出；
  第 2 轮说"基于刚才导出的第 3 条分析一下"
- 关键认知：第 2 轮**不该走向量检索**——知识库里没有"上一轮的导出结果"，
  检索永远命中不了；这是对上一轮结构化结果的推理，应走 **memory / 跨轮状态**
- 三项技术：①会话记忆（session_id + ChatMessageHistory）②跨轮状态管理
  （SQL 结果暂存会话，路由优先查状态）③Agent 工具化
  （query_sql / retrieve_vector / read_memory 自主编排）
- 归入规划：第 9 步（多轮记忆 + 跨轮状态，CLI 先验证再接 FastAPI）、
  第 10 步（Agent 工具化）；路由器升级为三路：vector / sql / memory

---

### V7.9 混合检索 + 重排序（第三阶段，待提交）

**改动内容**
- 单一向量检索（`retriever = vector_store.as_retriever(k=6)`）→ 三环节流水线：
  **BM25 关键词检索 → 向量检索 → RRF 融合 → bge-reranker 精排 Top 6**
- k 的职责变化（承接 V7.7）：k 从"回答视野"降级为"粗筛漏斗"——
  两路都召回 50，精排后只留 6 条进提示词，最终视野仍是 6

**为什么手写 RRF 和 Reranker（关键教学决策）**
- 官方 `EnsembleRetriever` / `CrossEncoderReranker` 位于独立包 langchain-retrievers，
  但该包国内镜像（清华/阿里）未同步、PyPI 也无法安装 → 装不上
- 改为手写两个类（各约 20 行）：
  * `RRFRetriever`：互惠排名融合，score(d) = Σ 1/(k + rank_i(d))，只比排名不比分数
  * `RerankerRetriever`：sentence-transformers CrossEncoder（BAAI/bge-reranker-base）
    懒加载，候选逐条与问题算相关度，重排取 top_n
- 教学收益：把官方"黑盒"讲透，且不依赖装不上的包（类似 V7.5 自写 load_xlsx 的思路）

**使用工具**
- `BM25Retriever`（langchain_community.retrievers）+ rank-bm25（算法）+ jieba（分词）
- `RRFRetriever` / `RerankerRetriever`（手写 BaseRetriever 子类）
- 新增依赖：rank-bm25、jieba

**踩坑记录（重要）**
- 坑 1：BM25 默认按空格分词，中文整句会被当成一个 token，检索直接失效
  → 必须传 jieba 分词器：`preprocess_func=lambda t: list(jieba.cut(t))`
- 坑 2：本项目 langchain-community 版 BM25Retriever 的参数名是 `preprocess_func`，
  新版独立包才叫 `tokenizer`——照抄新版文档会报 TypeError
- 坑 3：langchain-retrievers 独立包在国内镜像/PyPI 均无法安装 → 手写方案
- 坑 4：reranker 模型首次运行才下载（懒加载设计），import 阶段不会联网卡住

**验证结果**
- BM25 问"华信银行"精准命中金融案例行；向量问"给零售客户推荐什么"语义命中优品超市
- RRF 融合后 3 条客户案例行全部进 Top 候选
- bge-reranker 精排后 3 条案例行稳居前 3，噪声（主营/团队介绍）被压后
- 回归：问"所有客户的行业有哪些？" → 完整回答零售、金融、教育 ✅

**注意事项**
- 检索参数变化**不需要**重建 chroma_db（只影响查询，不影响索引内容）
- 与第 5 步的关系：本步只让"事实查询"更准，**替代不了 SQL**（见核心概念 6）
- 与 V7.8 的关系：验证了"混合检索 + 精排"让 Top-k 检索在更大数据量下依然可用，
  但仍受"原文进提示词"的物理限制

---

### V8+ 未来规划（详见 improvements.txt）

| 步骤 | 内容 | 关键工具 |
| --- | --- | --- |
| 3 | BM25 混合检索 + 重排序（✅ 已完成 V7.9） | 手写 RRFRetriever / RerankerRetriever（bge-reranker） |
| 4 | 视频字幕 / 语音转写 | srt 解析 / faster-whisper |
| 5 | Text-to-SQL + 路由器（问题分流） | SQLite 起步 → MySQL / RunnableBranch |
| 6 | FastAPI 封装 + 前端（封装完整能力） | FastAPI / 简单网页 |
| 7 | assets/logo 素材规范 | 本地目录 → 对象存储 |
| 8 | 海报 / 文案生成 | LLM 文案 + 程序化叠加 logo |
| 9 | 多轮记忆 + 跨轮状态（引用上轮结果） | ChatMessageHistory / 会话状态 |
| 10 | Agent 工具化（多步任务自主编排） | query_sql / retrieve_vector / read_memory |

---

## 三、核心概念：Document（文档）≠ 文件

> 为什么 `docs/` 只有 2 个文件，`rag.py` 却打印"加载了 7 个文档"？

**Document 是数据单元，不是磁盘文件**：一个 `Document` = 一段文本（`page_content`）+ 元数据（`metadata`）。
一个**文件**可以产生**多个 Document**，取决于用哪个加载器（Loader）读它：

| 格式 | 加载器 | 一个文件 → 几个 Document |
| --- | --- | --- |
| `.md` / `.txt` | TextLoader | 1 个（整个文件一块） |
| `.pdf` | PyPDFLoader | 每页 1 个 |
| `.csv` | CSVLoader | 每行 1 个（表头行除外） |
| `.xlsx` | load_xlsx（自写） | 表头拼进数据行，每行数据 1 个 |

**本项目的 7 = 1 + 6**：
- `sample.md` → 1 个 Document（整篇）
- `sample.xlsx` → 6 个 Document（产品价格表 3 行 + 客户案例 3 行，表头不再单独占块）

之后"切分成 10 个片段"：md 的 1 个按标题切成 4 块，xlsx 的 6 行保持不动 → 4 + 6 = 10。

**教学含义**：知识的最小粒度可以是"文件 / 页 / 行"，加载器和切分器决定这个粒度——
这正是 2.5 步"按格式差异化加载 + 切分"的意义。

---

### 核心概念 2：表格按行切分 → 表头与数据分离（字段名"有头无身"）

> 为什么 V7.5 刚实现按行切表时，问"客户行业是什么"会回答"是表格中的一个字段，但具体内容未显示"？

**表格按行切成 Document 的副作用**：表头行（字段名）和数据行（值）被拆成了**不同的块**。
检索"客户行业"时，向量匹配命中"客户行业"字样最多的表头行——但那行只有字段名没有值，所以答不上来。

**解决思路（V7.6）**：让数据行**自包含**字段名——把表头拼进每一行，
一行变成 `客户行业: 金融 | 客户名称: 华信银行 | ...`。这样：
1. 检索"客户行业"会命中**数据行**（因为数据行里也有"客户行业"字样）；
2. 命中的块本身就带着值，模型直接能回答，不用跨块"猜"。

**本项目实战验证**：
- 改造前：检索"客户行业是什么"命中表头行 `客户行业 | 客户名称 | 使用产品 | 合作年份`，回答"字段未显示"；
- 改造后：命中数据行 `客户行业: 金融 | 客户名称: 华信银行 | 使用产品: 星辰问答企业版 | 合作年份: 2022`，回答"华信银行，所属行业：金融"。

**教学含义**：**检索命中的最小单元必须是"自带完整语义"的**。切分粒度不能只看
"切得碎不碎"，还要看"切出来的块离开上下文后是否仍然自洽"——表头分离就是典型的
"切碎了但不自洽"案例。

---

### 核心概念 3：k 是"回答的视野"——检索返回条数决定模型能看到多少

> 为什么 V7.6 修好表头后，问"所有客户的行业"还是只答 1 个客户？

**k 的含义**：`as_retriever(search_kwargs={"k": N})` 的 k = 每次检索返回给模型的
**片段数量上限**。模型只能看到这 k 个片段，回答完全基于它们——**k 就是模型回答的视野**。

**本项目实战**（同一批数据，不同 k 的差异）：
- `k=2`：问"所有客户的行业"→ 3 条案例行只有 1 条进候选 → 答"仅有一条客户案例：华信银行（金融）"
- `k=6`：3 条案例行相关性都高，全部进候选 → 答"华信银行（金融）、优品超市（零售）、明德学院（教育）"

**k 太小 vs 太大的权衡**：
- k 太小：漏召回，**"列举所有"类问题必然答不全**
- k 太大：噪声多，不相干的块也进提示词，模型可能被误导（价格表行也会混进来）
- 经验：k 至少 ≥ 同一类记录的数量级；本项目 6 个数据行，取 k=6 正好全覆盖

**进阶（V8+ 第 3 步）**：真实项目数据量大时不能只靠调大 k——
用 BM25 混合检索先广召回、再用 reranker 精排，才能"又全又准"。

---

### 核心概念 4：列举/聚合类问题 ≠ Top-k 检索问题（RAG 的边界）

> 为什么 k=6 也只是权宜之计？以后有一万个客户，问"所有客户的行业"怎么办？

**先认清问题类型**：RAG 的 Top-k 检索只解决"找最相关的少数事实"；
"列举所有 / 汇总 / 统计"类问题需要**全量访问**，天生不是 Top-k 的舒适区。

| 问题类型 | 例子 | 正确路径 |
| --- | --- | --- |
| 事实查询 | 华信银行的行业是什么？ | Top-k 检索（命中 1~3 条就够）✅ |
| 列举/聚合/统计 | 所有客户的行业有哪些？ | 全量访问，Top-k 范式天然不适合 ❌ |

**一万个客户时 k=6 为什么崩（本项目实战推演）**：
- `k=6` 只能看到 6 条 → 和现在一样答不全；
- 把 k 调到 10000 → 提示词 token 超限（每行约 50 字 × 1 万 = 50 万 token），
  且检索噪声爆炸，无关行干扰回答。
- 结论：**k 只是"事实查询"的旋钮，拿它解决"列举"问题治标不治本**。

**数据量增长的三档正解**：
| 数据量 | 方案 | 原理 | 局限 |
| --- | --- | --- | --- |
| 几百行 | 整表引用 | 列举类问题直接把整个源文件当上下文，不走 Top-k | 提示词塞不下就崩 |
| 几千~几万行 | 分批聚合 | 分页检索全部相关块，汇总后再生成 | 多次调 LLM，慢且贵 |
| 生产级（1 万+） | 结构化 RAG / Text-to-SQL | 客户数据进数据库，"列举所有行业"翻译成 `SELECT DISTINCT 行业 FROM 客户`，返回聚合结果 | 需要建库 + SQL 能力，改动大 |

**当前项目的选择（V7.8）**：先记录边界、暂不改代码——
`k=6` 保留为"事实查询"的教学配置；等数据量真到那个级别，
走 Text-to-SQL（数据入库 + 问题转 SQL）才是生产级正解。

---

### 核心概念 5：引用"上一轮结果"的推理 ≠ 检索（多轮对话的边界）

> 为什么第 2 轮问"基于刚才导出的第 3 条分析一下"不能走向量检索？

**先认清信息来源**：RAG 检索只能从**知识库**（已入库的片段）里找答案；
而"上一轮的导出结果"是一个**运行时中间产物**，只存在于会话里，不在知识库里。
所以这类问题检索必然命中不了——它不是检索问题，是 **memory / 状态** 问题。

**本项目实战推演**（延续第 5 步 Text-to-SQL 场景）：
- 第 1 轮：问"所有客户的信息" → 路由器判断是"列举"类，走 SQL 链，全量导出
  3 条客户记录（华信银行、优品超市、明德学院）
- 第 2 轮：问"基于刚才导出的第 3 条分析一下" → 如果走**向量检索**，
  Chroma 里只有原始知识库片段，没有"第 3 条导出记录"这个对象，
  检索结果必然无关；正确做法是从**会话状态**里取出第 3 条再交给模型推理

**三条技术路线（第 9、10 步）**：
| 技术 | 作用 | 对应步骤 |
| --- | --- | --- |
| 会话记忆（session_id + ChatMessageHistory） | 记住"上轮问了什么、答了什么" | 第 9 步 |
| 跨轮状态管理（SQL 结果暂存会话，路由优先查状态） | 用户引用上轮结果时直接从状态取 | 第 9 步 |
| Agent 工具化（query_sql / retrieve_vector / read_memory） | 模型自主决定查库还是读状态 | 第 10 步 |

**教学含义**：RAG 有两条边界——①**数据量边界**（列举类问题需要全量访问，
见核心概念 4）；②**信息来源边界**（只能答知识库里有的，答不了"会话里才有的
中间结果"）。后者靠记忆/状态补，路由器升级为三路：vector / sql / memory。

---

### 核心概念 6：换 BM25/reranker 也替代不了 SQL（检索 vs 查询）

> 为什么第 3 步的 BM25 + bge-reranker 在 1 万个客户时，依然替代不了第 5 步的 SQL？
> （接核心概念 4：那里说"列举类 ≠ Top-k"，这里进一步问"把 Top-k 换成更好的检索器行不行"）

**先认清"检索"和"查询"是两种能力**：
- **检索**（向量 / BM25 / reranker）＝按"相关度"排序，返回前 N 条**原文片段**——
  本质是"挑最像的少数几条"，无论排序算法多好，返回的还是原文
- **查询**（SQL）＝按条件**过滤、聚合、统计**，返回**压缩后的结果**——
  本质是"集合运算"，结果体积远小于原始数据

**为什么换检索器也不行（本项目 1 万客户实战推演，问"所有客户的行业有哪些？"）**：
| 方案 | 结果 | 为什么不行 |
| --- | --- | --- |
| 向量 k=10000 | ❌ | 1 万行原文塞提示词 → token 超限 + 噪声爆炸 |
| BM25 k=10000 | ❌ | token 一样超限；且"行业"关键词每行都有（V7.6 表头拼接），BM25 打分全相同，排序失效，退化成顺序截断 |
| reranker 精排 1 万条 | ❌ | 交叉编码器逐条算分极慢；且它只做"排序取前 N"，**不做去重/聚合** |
| SQL：`SELECT DISTINCT 行业 FROM 客户` | ✅ | 返回"金融、零售、教育"3 行，结果被压缩 |

**核心差异一句话**：检索返回"原文"，1 万条原文塞不进提示词——这是**物理限制**；
SQL 返回"计算结果"，体积被压缩——这才是万级数据可行的**本质**。

**两者的正确分工（第 3 步 vs 第 5 步，不是平替是互补）**：
- 第 3 步 BM25 + reranker：让**事实查询**（"华信银行是什么行业"）在万级库里依然又全又准
  （广召回 Top 50 → 精排 Top 6），不因数据量变大而漏掉关键块；
- 第 5 步 Text-to-SQL：让**列举/统计/聚合**（"所有客户的行业有哪些"）在万级数据下可行；
- 路由器按问题类型分流：事实查询 → 检索链；列举/统计 → SQL 链。

**教学含义**：数据量变大后，优化检索器（第 3 步）是"把事实查询做得更准"，
但解决不了"全量访问"类问题——那是架构升级（数据入库 + SQL）的事。
类比：检索是**图书管理员按相关度挑 10 本书**，SQL 是**会计对 1 万张发票做统计**；
你要"全量字段"是会计的活，换哪个管理员（BM25 还是向量）都变不成会计。

---

## 四、依赖包清单（当前 requirements.txt）

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
| `openpyxl` | Excel 解析 | `load_xlsx`（自写函数） | V7.5 新增 |
| `rank-bm25` | BM25 关键词检索算法 | `BM25Retriever` 底层 | V7.9 新增 |
| `jieba` | 中文分词 | BM25 中文预处理（preprocess_func） | V7.9 新增 |
| `sentence-transformers` | 嵌入模型底层 | bge-small-zh 运行环境 + bge-reranker 交叉编码 | |
| `python-dotenv` | 环境变量 | `load_dotenv()` 读 .env | |

---

## 五、常用命令速查

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

## 六、踩坑记录汇总（按主题）

| 主题 | 现象 | 原因 | 解决 |
| --- | --- | --- | --- |
| HuggingFace 下载 | SSL 证书验证失败 | .env 缺失 + Windows 证书问题 | 建 .env；HF_ENDPOINT 镜像；USE_INSECURE_SSL=1 |
| LCEL 链 | 'dict' object has no attribute 'replace' | retriever 收到整个 dict | `(lambda x: x["question"]) \| retriever` |
| 版本兼容 | DirectoryLoader 不认 loader_map | langchain-community 0.4.2 移除该参数 | os.walk 手动遍历 + LOADER_MAP |
| 文件编码 | 'gbk' codec can't decode byte ... | Windows 默认 GBK 读 UTF-8 中文文件 | TextLoader/CSVLoader 显式 encoding="utf-8" |
| 索引陈旧 | 改了知识库回答还是旧的 | Chroma 持久化不感知源文件变化 | 删 chroma_db 重建 |
| 表头分离 | 问"客户行业"答"字段未显示" | 表格按行切，表头行与数据行分离 | 表头拼进数据行（字段名: 值） |
| 列举答不全 | 问"所有客户行业"只答 1 个 | k=2 太小，同类记录没进候选 | k 调大（本项目 k=6） |
| MMR 误伤 | 用 MMR 后同类行仍只回 1 条 | MMR 把结构相似的表格行当"重复"去重 | 表格场景用普通相似度检索 |
| BM25 中文失效 | BM25 检索中文整句当一个词，命中全乱 | rank_bm25 默认按空格分词 | 传 jieba 分词器（preprocess_func） |
| 参数名不兼容 | BM25Retriever 报 TypeError | langchain-community 版参数是 preprocess_func，新版才叫 tokenizer | 以本项目已装版本为准 |
| 包装不上 | langchain-retrievers 无法安装 | 国内镜像未同步 / PyPI 无此包 | 手写 RRFRetriever / RerankerRetriever |
| 验证脚本卡死 | 运行 _verify.py 报"连接方...没有正确答复" | HuggingFaceEmbeddings 初始化联网访问 HF 超时 | 验证索引用 chromadb 直接读，不加载嵌入模型 |
| PowerShell | rmdir /s /q 报错找不到参数 | /s /q 是 cmd 语法 | 用 Remove-Item -Recurse -Force |
| git push | SSL peer certificate 错误 | Windows 证书链问题 | `git config http.sslVerify false`（项目级） |

---

## 七、版本与 git 提交对照

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
| （已提交） | os.walk 手动遍历 + UTF-8 编码修复 + docs/ 示例 | V7 + docs/ |
| （已提交） | 切分差异化 + 原生 .xlsx + openpyxl 依赖 | V7.5 |
| （已提交） | xlsx 表头拼接修复（字段名拼进数据行） | V7.6 |
| （已提交） | 检索参数调优：k=2 → k=6（修复列举类问题答不全） | V7.7 |
| （已提交） | 认知边界记录：列举类问题 ≠ Top-k 检索（纯文档） | V7.8 |
| （待提交） | BM25 混合检索 + RRF 融合 + bge-reranker 精排（第三阶段） | V7.9 |
