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
├── V7.8 认知边界记录（纯文档，已提交）
│   ├── k=6 只是当前 10 个片段知识库的调优值，不是通用解
│   ├── "列举/聚合"类问题 ≠ Top-k 检索问题（见核心概念 4）
│   ├── 三档正解：整表引用 → 分批聚合 → Text-to-SQL（生产级）
│   └── 补充：引用"上一轮导出结果"的推理 ≠ 检索（走 memory，见核心概念 5 / 第 9 步）
│
├── V7.9 混合检索 + 重排序（第三阶段，已提交）
│   ├── 单一向量检索 → BM25 + 向量 + RRF 融合 + bge-reranker 精排
│   ├── k 的职责：从"回答视野"（V7.7）降级为"粗筛漏斗"，精排后定视野
│   ├── 坑：langchain-retrievers 包装不上 → 手写 RRFRetriever / RerankerRetriever
│   └── 坑：BM25 中文必须接 jieba 分词（preprocess_func 参数）
│
├── V8.0 OCR 扫描件加载（已提交）
│   ├── 场景：docs/ 放入扫描版 PDF（每页是图片、无文字层），检索结果为空
│   ├── load_pdf 升级：逐页判定——文字层够的页直接用；扫描页 PyMuPDF 渲染成位图 + RapidOCR 离线识别
│   ├── OCR 模型懒加载：只影响扫描件，普通文字版 PDF 启动不拖慢
│   ├── OCR 进度可视化（tqdm 进度条）+ OCR 结果缓存（ocr_cache/，二次构建不重跑）
│   └── 新增依赖：PyMuPDF / rapidocr-onnxruntime / Pillow / tqdm
│
├── V8.1 增量索引（已提交）
│   ├── 场景：加新文件必须删库重建、扫描 PDF 每次全量重 OCR（最贵一环反复支付）
│   ├── 实现：.index_state.json 记录文件指纹（mtime+size）→ 启动三向比对
│   │        （新增/变更/删除/跳过），只处理有变化的文件，其余零成本跳过
│   ├── 关键联动：BM25 改为从 Chroma 全量取回文本重建——内存不再有全量 chunks
│   │        （教学点：向量索引持久化 / BM25 无状态，见核心概念 11）
│   └── 实测：加文件只处理该文件、改文件只重建该文件、问答质量不变
│
├── V8.2 MultiQuery 多路检索（已提交）
│   ├── 场景：《亮剑》"李云龙妻子有几个"只答田雨——漏了秀芹（书里有两位）
│   ├── 根因：一个问法只匹配一种"说法"——"妻子"命中田雨，秀芹是"婆娘/娶媳妇"
│   ├── 实现：LLM 拆子查询（集合类逐成员拆）→ 每路独立召回精排 → rank 融合合并
│   └── 纯手写：复用 CrossEncoder/ChatOpenAI，零新增依赖（详见核心概念 9/10）
│
├── V8.3 MultiQuery 性能优化：跨路去重（已提交，规划 C 第 1 项）
│   ├── 场景：5 路子查询是同一问题的不同问法，命中的语料高度重叠——重复率实测 47%
│   ├── 实现：收集候选时按 page_content 去重（约 5 行），同一片段只打一次分
│   └── 实测：100 → 53 对，invoke 6.5s，回答质量不变
│
└── V8+ 未来规划（见 improvements.txt）
    ├── 3   已提交（V7.9）：BM25 混合检索 + bge-reranker 重排序
    ├── 3.5 已提交（V8.0）：扫描件 PDF 自动 OCR（PyMuPDF + RapidOCR）
    ├── 4   视频字幕 srt / 语音转写接入
    ├── 5   Text-to-SQL + 路由器（问题分流：向量 vs SQL；SQLite 起步 → MySQL）
    ├── 6   FastAPI 封装 + 前端页面（封装完整能力，含 SQL 路由）
    ├── 7   assets/logo 素材目录规范
    ├── 8   海报 / 文案生成（logo 程序化叠加）
    ├── 9   多轮记忆 + 跨轮状态（路由三路：vector / sql / memory）
    ├── 10  Agent 工具化（query_sql / retrieve_vector / read_memory 自主编排）
    ├── 11  依赖解耦重构（方案 A，待实施）：去掉 langchain-community，手写薄封装
    └── 12  已提交（V8.1）：增量索引——.index_state.json + 文件级差异，免全量重建
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

### V7.9 混合检索 + 重排序（第三阶段，已提交）

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

### V8.0 OCR 扫描件加载（已提交）

**改动内容**
- 背景：用户向 `docs/` 放入扫描版《亮剑》PDF（446 页），检索结果为空——
  "加载了 453 个文档"（446 空页 + md + xlsx 行）但"切分 10 个片段"（空页切后仍为空）
- `load_pdf` 升级为"双层判定"：
  1. 先用 PyPDFLoader 按页提取文字层
  2. 逐页检查：页面文字 < `OCR_MIN_TEXT_LEN`（20 字符）→ 判定为"扫描页"
  3. 文字层够的页直接用（保持原行为）；扫描页走 OCR：PyMuPDF 渲染成位图
     （dpi=200）→ RapidOCR（离线中文识别）识别 → 按行从上到下排序拼回 Document
  4. 打印统计：X 页文字层 / Y 页 OCR；OCR 页 metadata 标记 `"ocr": True` 便于溯源
- OCR 模型**懒加载**：只有遇到扫描页才 import/初始化 RapidOCR，普通文字版 PDF 完全不受影响

**为什么"逐页判定"而不是"整本判定"**
- 整本判定：只要 PDF 有文字层就全走文字，遇到"前 10 页文字、后面扫描"的混合型会丢内容
- 逐页判定：每页独立看文字量，混合型 PDF 也正确——是更通用的加载策略

**使用的工具**
- `PyMuPDF`（`import pymupdf as fitz`）：PDF 页渲染成位图（dpi=200，越高越清晰也越慢）
- `rapidocr-onnxruntime`：离线中文 OCR，模型内置于包内，不联网、免费
- `Pillow`：位图转 numpy 数组喂给 OCR
- `tqdm`：OCR 循环进度条——当前页/总页数、每页耗时、预计剩余时间一目了然

**OCR 进度可视化 + 结果缓存（V8.0 增强）**
- 背景：446 页全扫描 OCR 要跑几十分钟，且原实现跑完不落盘——删库重建又全量重跑，
  "最贵的一环反复支付"
- 进度可视化：OCR 循环用 tqdm 进度条，实时显示 当前页/总页数、每页耗时（s/页）、
  预计剩余时间，能立刻判断"在正常跑"还是"卡在某页"
- 结果缓存：识别文本落盘 `ocr_cache/<PDF 文件名>.txt`（目录必须在 docs/ 之外！），
  下次构建索引命中缓存（缓存 mtime ≥ PDF mtime）直接读文本，OCR 一次不跑；
  缓存格式 `<<<PAGE:页索引>>>` 分隔行 + 正文（正文可含换行）
- 实测：单页 OCR 约 1.3~2.2 秒（CPU），446 页约 10~16 分钟

**踩过的坑（重点）**
- 坑 1：扫描版 PDF 没有文字层——`PyPDFLoader` 提取每页为空字符串（实测 446 页
  平均 0 字符），不是加载失败也不是编码问题，是 PDF 本身的内容形态
- 坑 2：新版 PyMuPDF 弃用 `fitz` 模块名 → 警告 DeprecationWarning；
  应 `import pymupdf as fitz`（pymupdf 是新包名，fitz 兼容旧名）
- 坑 3：命令行传中文文件名 + 打印中文在 Windows 控制台（GBK）会编码报错，
  干扰验证；验证脚本应把结果写 UTF-8 文件再读
- 坑 4：OCR 缓存文件不能放 docs/ 目录——os.walk 会把它当普通文档重复加载，
  内容翻倍；缓存目录必须在知识库目录之外（本项目 ./ocr_cache）
- 坑 5：Windows 下 open("w") 默认把 \n 写成 \r\n，分隔行格式解析缓存会错位
  → 缓存读写都要显式指定 newline="\n"

**新增依赖**
- `PyMuPDF`、`rapidocr-onnxruntime`、`Pillow`

**验证结果**
- 前 3 页 OCR：第 1 页封面 0 行（纯图片，正常）；第 2 页"地势坤，君子以厚德载物。"
  第 3 页完整识别出正文文字——中文识别质量良好

**注意事项**
- OCR 有固有成本：每页约 1~3 秒（CPU），446 页全扫描预计几分钟~十几分钟，一次性
- OCR 可能有少量错字（生僻字/书名号），但检索是"片段匹配"，影响不大
- 加载逻辑变化后必须删 `chroma_db` 重建索引才生效

---

### V8.2 MultiQuery 多路检索（已提交）

**改动内容**
- 背景：问《亮剑》"李云龙妻子一共有几个"只答出田雨——书里明确有两位妻子（秀芹 + 田雨）
- 诊断（三层实验定位，见核心概念 9）：
  1. 召回层：单路 RRF top50 含"秀芹"片段 **0 条**——书里秀芹从不用"妻子"称呼
     （婆娘/娶媳妇/新婚妻子），问题里恰好带"妻子"→ BM25/向量全偏田雨
  2. 精排层：MultiQuery 召回 16 条秀芹后，用【原问题】统一精排，秀芹最高 #15
     （score 0.764 vs 田雨 0.959）——弱相关但关键的片段照样被压掉
  3. 正解：每路子查询用【自己的子查询】打分，各自取 top 再合并——秀芹 3 条进 top6
- rag.py 新增手写 `MultiQueryRetriever`（约 90 行，零新增依赖）：
  1. LLM **一次调用**把问题拆成 5 路子查询（要求 JSON 数组，三级容错解析：
     json.loads → 正则抓 [...] → 退化原问题）
  2. 每路子查询走 RRF 粗召回 top20（recall_top_n，性能优化：50 条/路纯属浪费，
     精排每路只取前 4 条）→ 全部候选**合并一个 batch** 用各自子查询精排
     （批量并行，见核心概念 10）
  3. 每路取前 per_query_top_n(4) → **rank 融合分**排序（不同路的分数不可比，只比排名）
     → 去重取 top6
- 管道一行不变：`| retriever |`（换实现不动骨架——核心概念 8 的又一次实例）

**踩过的坑（重点）**
- 坑 1（最关键）：LLM 拆的子查询不"纯"——"李云龙 感情线 妻子 田雨 秀芹 杨秀芹"
  一个子查询混 4 个名字，reranker 打分时秀芹被同路的田雨压出前 3 被截掉
  → 修 prompt：集合类问题**为每个成员单独生成子查询**、每路只聚焦一个实体、
  直接带人名（"李云龙第一任妻子 杨秀芹"）；效果立竿见影
- 坑 2：per_query_top_n=3 太小，弱相关但关键的信息容易被截 → 3→4

**验证结果（真实链路：LLM 拆 + 检索 + 问答）**
- LLM 拆出 5 路：第一任妻子 杨秀芹 / 第二任妻子 田雨 / 杨秀芹关系 / 田雨关系 / 妻子人物关系
- 最终视野 top6 含秀芹 3 条（结婚场景 / 人物介绍 / 被俘段落）
- 完整问答："李云龙一共有两位妻子：1. 秀芹（第一任，被俘牺牲）2. 田雨（第二任）" ✅

**性能优化（recall_top_n=20，V8.2 实测数据）**
- 背景：V8.2 上线后问答明显变慢。耗时分解实测：拆查询 1.1s / 召回 0.1s /
  **reranker 250 对 24.6s** → 真正的大头不是"多了一次 LLM 调用"（仅 1.1s），
  而是 5 路子查询 × 每路召回 50 条 = 250 对候选在本地 CPU 上逐批推理
- 优化：每路粗召回 50 → 20（每路精排只取前 4 条进最终视野，前 20 条里足够）
- 实测：250 → 100 对，reranker 24.6s → 10.2s；相对单路总增量 ~20s → ~6s
- 质量验证（完整问答）：仍正确答出"两位妻子：秀芹（第一任）+ 田雨（第二任）" ✅
- 教学点（见核心概念 10）：batch 并行解决了"多路多次调用"的问题，但**喂给模型
  的样本总量大了照样慢**——先控制输入规模，再谈并行

**注意事项**
- 代价：每次问答多一次 LLM 调用（拆分子查询，约 1~2 秒）——用"多花一次小调用"
  换"枚举类问题不漏全集"
- 与核心概念 4 的关系：MultiQuery 让 Top-k 检索更"全"，但**仍受"原文进提示词"
  的物理限制**——万级"列举/统计"问题照样该走 SQL（数据入库 + 查询），不冲突

---

### V8.3 MultiQuery 性能优化：跨路去重（已提交，规划 C 第 1 项）

**改动内容**
- 背景：V8.2 的 recall_top_n=20 已把候选砍到 100 对（250→100，24.6s→10.2s），
  但"样本总量"仍有虚胖空间——5 路子查询是同一问题的不同问法，命中的语料高度重叠
- 实测数据：5 路 × 20 条 = 100 对候选，按 page_content 去重后只剩 53 个唯一片段
  （重复率 47%）——同一段文本被多路子查询重复召回，reranker 对相同内容算了不止一遍
- 实现（约 5 行）：收集候选时维护 seen 集合，按 page_content 跨路去重，
  同一片段只保留第一次出现，打分对数 100 → 53（下游分组/融合代码一行不用改）
- 实测：warm-up 后 invoke（拆路+召回+去重+打分+融合）6.5s；完整问答仍答出
  "两位妻子：田雨 + 秀芹" ✅
- 教学点：多路检索的"路"是问法维度，不是内容维度——不同问法命中同一段语料是常态；
  去重要在"打分前"做（最贵的环节），而不是"融合时"才去重（那是结果去重，浪费已发生）

**注意事项**
- 融合分微降：被多路命中的片段只保留第一路记录，rank 融合少一次排名贡献。
  实测排序基本不受影响——跨路重复的片段通常各路排名都不错，且腾出的名额
  让更多唯一内容进候选
- 与规划 C 的关系：本项是性价比最高、零新增依赖的第一刀（第 2 项动态拆路 /
  第 3 项量化 / 第 4 项两级过滤见 improvements 规划 C）

---

### V8+ 未来规划（详见 improvements.txt）

| 步骤 | 内容 | 关键工具 |
| --- | --- | --- |
| 3 | BM25 混合检索 + 重排序（✅ 已提交 V7.9） | 手写 RRFRetriever / RerankerRetriever（bge-reranker） |
| 3.5 | 扫描件 PDF 自动 OCR（✅ 已提交 V8.0） | PyMuPDF 渲染 + RapidOCR 离线识别 |
| 4 | 视频字幕 / 语音转写 | srt 解析 / faster-whisper |
| 5 | Text-to-SQL + 路由器（问题分流） | SQLite 起步 → MySQL / RunnableBranch |
| 6 | FastAPI 封装 + 前端（封装完整能力） | FastAPI / 简单网页 |
| 7 | assets/logo 素材规范 | 本地目录 → 对象存储 |
| 8 | 海报 / 文案生成 | LLM 文案 + 程序化叠加 logo |
| 9 | 多轮记忆 + 跨轮状态（引用上轮结果） | ChatMessageHistory / 会话状态 |
| 10 | Agent 工具化（多步任务自主编排） | query_sql / retrieve_vector / read_memory |
| 11 | 依赖解耦重构（方案 A，已记录待实施） | 手写 5 个加载器 + BM25Retriever，去掉 langchain-community |
| 12 | 增量索引（✅ 已提交 V8.1） | .index_state.json + Chroma 按 source 增删，免全量重建 |

**规划 A：依赖解耦重构（已记录，待实施；2026-08 记录，今日不改代码）**
- 动机：`langchain-community` 已官宣 sunset（rag.py 每次运行都有 DeprecationWarning）；
  `BM25Retriever` 参数名已随版本变过（preprocess_func vs tokenizer）——薄封装最易踩版本坑
- 决策：**LCEL 管道不重写**（保留 `|` 声明式写法，见核心概念 8），只解耦"依赖不稳定的具体实现"
- 改动范围（约 60~80 行，只动加载器 + BM25）：
  1. 手写 TextLoader（open + encoding 读，~5 行）
  2. 手写 PyPDFLoader（pymupdf 逐页 page.get_text()，~10 行，顺带删 pypdf 依赖）
  3. 手写 CSVLoader（标准库 csv 逐行，~10 行）
  4. 手写 Docx2txtLoader（zipfile 解 document.xml 取文本，~15 行）
  5. 手写 BM25Retriever（rank_bm25.BM25Okapi + jieba 分词，~20 行）
- 效果：`langchain-community` 依赖完全消失，最大版本风险源清零；pypdf / docx2txt 也可删
- 判断标准（教学点）：①看封装在哪个包——community（sunset）必换、core（基石）可留；
  ②看封装有多薄——"读文件→包对象"薄如纸手写最划算、"持久化+索引"厚封装别重复造轮子

**规划 B：增量索引（✅ 已实施 V8.1，2026-08-25；对应 improvements 第一部分第 5 条）**
- 痛点（为什么做）：V5 起的设计缺陷——`chroma_db` 存在直接加载、不存在全量重建，
  加新文件必须删库重建，扫描 PDF 每次全量重 OCR（最贵一环反复支付）
- 实现（V8.1）：
  1. **状态文件** `.index_state.json`：`{文件路径: 指纹}`，指纹 = mtime+size
     （快，教学够用；注释说明生产可换 md5）
  2. **启动三向分支**：①无 chroma_db → 全量构建 + 记全量指纹；
     ②有索引无状态文件（旧索引）→ 全量校准（只记指纹、不重嵌，教学假设索引与磁盘一致）；
     ③有状态 → 增量比对
  3. **增量比对**：新增 → `add_documents`；变更 → `delete(where={"source": 路径})` + 重建；
     删除 → 同上 delete；未变 → 跳过。启动打印 `增量比对：新增 n 变更 n 删除 n 跳过 n`
  4. **BM25 数据源改造（关键联动）**：从内存 chunks 改为从 Chroma 全量取回文本重建
     （增量后内存不再有全量 chunks）——教学点：向量索引持久化 / BM25 无状态（核心概念 11）
  5. **配套**：`load_documents` 拆出 `load_single_file`（增量的最小处理单元）；
     `.gitignore` 加 `.index_state.json`（可重建产物不入库）
- 验证（临时目录隔离 6 场景 + 真实环境）：
  - 首次全量 / 无变化全部跳过 / 新增只处理新增 / 变更只重建变更 / 删除索引同步删 /
    回归问答答出"两位妻子：秀芹 + 田雨" ✅
  - 真实旧索引自动走"全量校准"（不重嵌），第二次启动走增量（跳过 3 文件）

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

### 核心概念 7：扫描版 PDF ≠ 检索失败——"无文字层"要先转文本（OCR）

> 为什么《亮剑》PDF 放进 docs/ 后，程序"加载了 453 个文档"却检索不出任何内容？

**先认清 PDF 的两种形态**：PDF 里的文字可能以两种方式存在——
- **文字层**（文字版 PDF）：文字是"可复制"的字符，`PyPDFLoader` 直接能提取；
- **图像层**（扫描版 PDF）：每页是一张图片，文字只是画面，没有字符——
  提取出来的 `page_content` 是**空字符串**（实测 446 页平均 0 字符）。

**"加载成功" ≠ "有内容"**：`PyPDFLoader` 不会报错，它会老老实实返回 446 个
空文本的 Document——加载环节正常，切分环节切出的也是空块，嵌入后检索自然
什么都命中不了。**这是最容易误判的一类故障**：程序不报错，但结果为空。

**本项目实战验证**：
- 现象："加载了 453 个文档"（446 个空 PDF 页 + md + xlsx 行）、
  "切分成 10 个片段"（空页切后仍为空，只剩 md/xlsx 的片段）→ 检索 PDF 内容必为空；
- 修法（V8.0）：`load_pdf` 逐页判定，文字量 < 20 字符的页判定为扫描页，
  用 PyMuPDF 渲染成位图 + RapidOCR（离线中文识别）转成文字，再走原管道；
- 效果：第 3 页正文完整识别，中文识别质量良好。

**教学含义**：多模态内容的通用思路是"**先转文本，再进 RAG**"（对应
improvements.txt 第二部分）。PDF 扫描件 → OCR 转文本、视频 → 语音转写、
图片 → 视觉模型描述，都是同一条路。判定"是否需要转"的方法是看加载器
提取出的内容是否为**空/过短**，而不是看程序报不报错。

---

### 核心概念 8：管道 `|`（Pipeline）——声明式数据流，不是"黑魔法"

> 为什么 `chain = {"context": ...} | prompt | llm` 一行能串起检索 + 提示 + 生成？
> （2026-08 决策：管道保留不重写，为什么？）

**管道是什么**：`|`（pipe）把"组件"串成"数据流"——左边组件的输出自动成为
右边组件的输入。`rag.py` 里就是一条从"问题"流向"答案"的流水线：

```
{"question": "..."}
   ↓
{"context": (lambda x: x["question"]) | retriever | format_docs,  ← 分支1：检索→拼文本
 "question": lambda x: x["question"]}                             ← 分支2：问题透传
   ↓
prompt（ChatPromptTemplate：把 {context} 和 {question} 填进模板）
   ↓
llm（ChatOpenAI）
   ↓
{"content": "..."}
```

**三个核心特性（本项目都有实例）**：
1. **组件可串可并**：`retriever | format_docs` 串接（Document 列表 → 文本）；
   `{"context": ..., "question": ...}` 是**并行分支**——两路独立计算，最后合并进 prompt
2. **换实现不动骨架**：V7.9 把单一 `retriever` 换成 `RRFRetriever → RerankerRetriever`
   组合，管道代码 `| retriever |` 一行没变——声明式的价值：**组合方式与实现解耦**
3. **输入输出是"鸭子类型"**：节点只需关心"上家给了什么、下家要什么"。副作用是 V4 的坑：
   retriever 只接受字符串，并行分支里它却收到整个 dict → 必须 `(lambda x: x["question"])`
   先取出字符串

**为什么本项目"不重写"管道（2026-08 决策）**：
- 手写函数链（`context = format_docs(retriever.invoke(q)); resp = llm.invoke(...)`）
  逻辑等价，但损失"换节点不动骨架"的灵活性
- LCEL 属于 langchain-**core**（生态基石，版本稳定），不是已 sunset 的 community；
  "薄封装"（加载器/检索器）才是版本坑高发区，管道不是
- 判断标准：**管道本身就是"组合抽象"，重写它等于再造轮子**；要解耦的是
  "依赖不稳定的具体实现"（方案 A 的 community 包），不是"组合方式"

**教学含义**：学 RAG 不只学"加载→切分→嵌入→检索→生成"五个零件，更要理解
**零件怎么插**。管道 = 组件间的"插座协议"，它让"换检索器、加过滤器、插记忆"
都是局部改动。V4 的坑正好说明：声明式越优雅，越要清楚每个节点接什么类型。

---

### 核心概念 9：一个问法只能匹配一种"说法"——Multi-Query 多路子查询

> 为什么问"李云龙妻子有几个"只答出田雨？书里明明有两位妻子（秀芹 + 田雨）？

**问题的本质：检索是"匹配"，不是"推理"**。同一事实在原文里可以有多种说法：
- 田雨 = **"妻子"**（书里明确写"你的妻子田雨"）
- 秀芹 = **"婆娘 / 娶媳妇 / 新婚妻子 / 嫂子"**（通篇几乎没有"妻子"字样）

你的问题只带"妻子"这一个说法 → BM25 按词命中全偏田雨；向量语义同样偏田雨。
**一个问法只能匹配一种说法，这就是单路检索的天花板。**

**枚举/集合类问题（"有几个 / 都有谁"）为什么最容易踩**：正确答案分散在多个片段
（秀芹的结婚段落、被俘段落，田雨的结婚段落……），没有任何一段同时说"两任妻子"，
而检索器只找"相似度最高的少数几条"——天然漏"全集"。

**Multi-Query 的解法（三步）**：
1. **拆**：LLM 把问题拆成多个子查询，覆盖不同说法/人名/角度——"妻子"漏了秀芹，
   但"李云龙第一任妻子 杨秀芹"、"杨秀芹和李云龙是什么关系"能把她捞回来
2. **各自查**：每路子查询独立召回 + 精排——关键：每路用【自己的子查询】打分，
   秀芹在"杨秀芹"这路的精排 top 里必有她（不会被田雨压掉）
3. **合**：合并去重进最终视野（不同路的分数不可比，用 rank 融合分排序）

**本项目实战验证（三层实验，同样的问题）**：
| 方案 | 秀芹的最终结局 |
| --- | --- |
| 单路 RRF + reranker（V8.1 原实现） | 召回层就 0 条，根本进不了精排 |
| 多路召回 + 【原问题】统一精排 | 秀芹最高 #15（0.764 vs 田雨 0.959）——还是漏 |
| 多路召回 + 每路用【自己的子查询】精排 | 秀芹 3 条进 top6，问答答出"两位妻子" ✅ |

**教学含义**：RAG 的"检索质量"不只看排序算法（BM25 / 向量 / reranker），
还看"**问法覆盖**"——同义改写、枚举成员、关系式问法，都是把"一种问法"扩展成
"多种问法"的手段。官方 `MultiQueryRetriever` 就是干这个的；本项目手写约 90 行
（复用已有 CrossEncoder / ChatOpenAI，零新增依赖），与 RRF/Reranker 同为
"官方有、手写能讲透"的组件。

---

### 核心概念 10：并行提速的正确姿势——batch 批量并行 > 多线程

> MultiQuery 拆了 5 路子查询，每路都要召回 + 精排，能不能用多线程加速？

**先分清"什么能并行、什么不能"**：
- **LLM 拆分子查询：没有并行空间**——一次调用就生成全部子查询；
  多线程拆 5 次 = 5 次 API 调用 = 5 倍成本 + 5 倍延迟
- **每路的召回（BM25/向量）：毫秒级**，多线程收益甚微，还吃 GIL 调度开销
- **reranker 精排（交叉编码器）：真正的耗时大头，且天然可以批量** ↓

**正解：把全部候选对合并成一个 batch，一次 predict**：
- CrossEncoder.predict 支持批量：5 路 × 50 候选 = 250 对，**一次推理**完成
- 模型只加载一次、单次前向传播吃满 CPU/GPU——比"每路单独 predict"
  （模型加载多次 + 多次小推理）快数倍
- 这就是**批量并行（batch）**：把多条样本合成一个张量一起算，
  是单机推理的正确并行姿势

**为什么不用多线程（Python 特有）**：
- CPU 密集的模型推理受 **GIL（全局解释器锁）** 限制——多线程同一时刻只能有一个
  执行 Python 字节码；推理的 native 代码还要反复抢 GIL，多线程**反而更慢**
- 多进程能绕开 GIL，但每个进程要**重新加载一次模型**（几百 MB），内存爆炸
- 结论：**单机推理提速 = 增大 batch，不是开线程**；多线程适合 IO 密集
  （等网络/磁盘/API），不适合 CPU 密集的模型推理

**本项目实例**：`MultiQueryRetriever` 里
`self._get_cross_encoder().predict(pairs, batch_size=64)`——250 对一次算完，
比逐路 predict 快数倍；而拆子查询老老实实一次 LLM 调用，不"并行"。

**但 batch 并行也要控制总量（V8.2 实测数据，重要补充）**：
- 250 对合并一个 batch 也要 **24.6s**（本地 CPU）——并行姿势对了，
  样本总量 5 倍照样慢 5 倍
- 追问：这 250 对真的都需要精排吗？每路只取精排前 4 条进最终视野，
  召回 50 条/路纯属浪费 → 砍到 20 条/路（recall_top_n=20）：100 对 → **10.2s**，
  相对单路总增量 ~20s → ~6s，问答质量不变（仍答出两位妻子）
- 结论：**先控制输入规模，再谈并行**——batch 解决"怎么算得快"，
  召回量决定"算多少"，后者省得更多、也更简单（改一个参数而已）

**教学含义**："并行"不是万能加速器，要分**场景**：IO 密集 → 多线程/异步；
CPU 密集的模型推理 → batch（合并样本一次算）；远程 API → 减少调用次数
（一次多给几个问题），而不是并发轰炸。

---

### 核心概念 11：向量索引持久化 / BM25 无状态——两种"存储"的哲学

> 为什么 V8.1 增量索引后，BM25 要从 Chroma 里取回文本重建？

**先看现象（本项目 V8.1 改动的直接后果）**：增量索引后，`rag.py` 内存里不再有
"全量 chunks"——只有变化文件的局部片段。但 BM25 检索器必须拿到**全部**文本来建
词频索引，否则检索会漏掉没变过的文件。

**两种"存储"的本质区别**：
- **向量索引（Chroma）：持久化、有状态**——嵌入向量 + 原文落盘到 `chroma_db/`，
  重启直接加载；可以**增量维护**（新增 `add_documents`、删除按 `source` 过滤）。
  因为"内容"在磁盘上有唯一权威副本，索引只是它的投影。
- **BM25：内存态、无状态**——词频统计在内存里现场构建，不落盘；每次启动
  从权威副本（向量库存的原文）取回重建。它的"状态"就是内存，进程结束即消失。

**教学含义**：判断一个索引能否增量，看它的"状态存在哪"——落盘的可以增量维护，
纯内存的每次现造。本项目 BM25 每次从 Chroma 取回 2541 个片段重建（实测 1 秒级），
代价可接受，换来的是"只有一个权威数据源（向量库），不维护两份状态"。

**本项目实例**：`_all_docs = vector_store.get(include=["documents", "metadatas"])`
→ `BM25Retriever.from_texts(...)` 每次启动现场重建（无状态）；而 Chroma 的
`add_documents` / `delete(where={"source": 路径})` 做增量维护（有状态）。

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
| `PyMuPDF` | PDF 页渲染成位图 | `load_pdf` OCR 分支（扫描件） | V8.0 新增 |
| `rapidocr-onnxruntime` | 离线中文 OCR（包内置模型） | 扫描页文字识别 | V8.0 新增 |
| `Pillow` | 图像处理（位图 → numpy 数组） | OCR 前处理 | V8.0 新增 |
| `tqdm` | 进度条 | OCR 循环进度可视化（页数/速度/ETA） | V8.0 新增 |
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
| 扫描版 PDF 检索为空 | 加载了文档但检索 PDF 内容为空 | PDF 每页是图片、无文字层，PyPDFLoader 提取为空 | 逐页判定 + OCR（PyMuPDF 渲染 + RapidOCR） |
| fitz 弃用警告 | import fitz 报 DeprecationWarning | 新版 PyMuPDF 推荐 pymupdf 模块名 | `import pymupdf as fitz` |
| 控制台中文乱码 | 命令行打印中文报 UnicodeEncodeError | Windows 控制台默认 GBK 编码 | 验证脚本结果写 UTF-8 文件再读 |
| OCR 缓存重复 | OCR 缓存放 docs/ 被 os.walk 当文档重复加载 | 缓存目录在知识库目录内 | 缓存放 ./ocr_cache（docs/ 之外） |
| 缓存解析错位 | 读缓存内容错位/乱序 | Windows 写文件 \n 变 \r\n，分隔行匹配不上 | 缓存读写都指定 newline="\n" |
| MultiQuery 子查询不纯 | "妻子有几个"只答田雨，秀芹全漏 | LLM 拆子查询混多个名字，弱相关关键信息被同路强相关压出前 N | prompt 强制"集合类逐成员拆、每路只聚焦一个实体、直接带人名" |
| PowerShell | rmdir /s /q 报错找不到参数 | /s /q 是 cmd 语法 | 用 Remove-Item -Recurse -Force |
| git push | SSL peer certificate 错误 | Windows 证书链问题 | `git config http.sslVerify false`（项目级） |
| git push | GH001 大文件被拒（786MB PDF 超 100MB） | 大文件误入 commit 历史，pre-receive hook 拒收 | `git reset --soft HEAD~1` 撤回 → `git rm --cached` 移出跟踪（本地保留）→ .gitignore 加 `docs/*.pdf` → 重新提交推送；可重建产物（ocr_cache/、chroma_db/）同理不入库 |
| PowerShell | commit message 中文乱码 | PS 向 native 进程传中文参数发生 UTF-8↔GBK 双重转换 | message 写入 UTF-8 文件，`git commit -F 文件` 读取（绕过传参编码） |

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
| （已提交） | BM25 混合检索 + RRF 融合 + bge-reranker 精排（第三阶段） | V7.9 |
| （已提交） | 扫描件 PDF 自动 OCR + 进度条 + 结果缓存（PyMuPDF + RapidOCR + tqdm） | V8.0 |
| （已提交） | MultiQuery 多路检索：LLM 拆子查询 + 每路独立精排 + rank 融合合并（手写，零新依赖） | V8.2 |
| （已提交） | MultiQuery 跨路去重：收集候选按 page_content 去重，打分对数 100→53（规划 C 第 1 项） | V8.3 |
