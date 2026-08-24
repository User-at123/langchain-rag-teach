# LangChain 入门教学项目

一个最简单的 LangChain 示例：把「提示词模板 + 模型」组合成一条链，实现一问一答。
已适配 **DeepSeek API**（OpenAI 兼容接口）。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key
cp .env.example .env    # 然后编辑 .env，填入你的 DeepSeek API Key

# 3. 运行
python main.py    # 普通问答
python rag.py     # RAG 问答（首次运行会下载嵌入模型并建索引，之后运行秒加载）
```

## 配置说明（.env）

| 变量 | 说明 |
| --- | --- |
| `OPENAI_API_KEY` | 你的 DeepSeek API Key |
| `OPENAI_BASE_URL` | DeepSeek 接口地址，默认已填好 |
| `LLM_MODEL` | 对话模型，默认 `deepseek-chat`（也可填 `deepseek-reasoner`） |
| `EMBEDDING_MODEL` | 嵌入模型，默认 `BAAI/bge-small-zh-v1.5`（本地开源模型） |
| `RERANKER_MODEL` | 可选，重排序模型，默认 `BAAI/bge-reranker-base`（首次运行自动下载） |
| `CHROMA_DIR` | 向量索引存储目录，默认 `./chroma_db`（首次运行创建，之后直接加载） |
| `OCR_CACHE_DIR` | OCR 结果缓存目录，默认 `./ocr_cache`（扫描 PDF 识别文本落盘，避免重复 OCR） |
| `HF_ENDPOINT` | 可选，HuggingFace 国内镜像（`https://hf-mirror.com`），加速/修复模型下载 |
| `USE_INSECURE_SSL` | 可选，设为 `1` 可跳过 SSL 证书验证（仅教学用，修复证书报错） |

> **为什么嵌入不用 DeepSeek？** DeepSeek 目前不提供 Embedding API，所以 `rag.py`
> 使用本地开源模型 `BAAI/bge-small-zh-v1.5`（中文效果好、体积小），首次运行自动下载，
> 之后离线可用，也不消耗 API 额度。

## 常见问题

**1. 下载嵌入模型报 `SSL: CERTIFICATE_VERIFY_FAILED`**

Windows/公司代理环境的常见问题，二选一解决：

- 在 `.env` 中取消注释 `USE_INSECURE_SSL=1`（最快，仅教学用）
- 或在命令行先执行 `set HF_ENDPOINT=https://hf-mirror.com` 再用国内镜像下载

**2. 模型下载很慢或一直失败**

在 `.env` 中取消注释 `HF_ENDPOINT=https://hf-mirror.com`，走国内镜像。

**3. 修改了 docs/ 里的知识库文件，但 rag.py 回答还是旧内容**

Chroma 索引是持久化的，不会自动感知源文件变化。更新知识库后，
删掉 `chroma_db` 目录再运行即可重建索引（当前为全量重建模式；
"增量更新"已规划为方案 B——记录文件指纹，只处理新增/变更的文件）：
```bash
rmdir /s /q chroma_db    # Windows（PowerShell 用 Remove-Item -Recurse -Force chroma_db）
# 或 rm -rf chroma_db    # Mac/Linux
python rag.py
```

**4. 放了 PDF 进去，但检索 PDF 内容为空**

大概率是**扫描版 PDF**（每页是图片、没有文字层），`PyPDFLoader` 提取出来是空文本，
程序不报错但检索结果为空。V8.0 起 `rag.py` 会自动检测并走 OCR（离线中文识别）：
- 文字层够的页直接用；扫描页自动 OCR（首次碰到会加载 OCR 模型，稍慢）
- OCR 全程有进度条（tqdm）：实时显示当前页/总页数、每页耗时、预计剩余时间，
  等多久都看得见进度，不用猜"是不是卡住了"
- 识别结果会缓存到 `ocr_cache/`（自动创建，无需手动管），下次构建索引直接读缓存，
  不再重跑 OCR
- 446 页全扫描预计 10~16 分钟（约 1.5~2 秒/页，CPU），属正常（OCR 固有成本，一次性）
- 改完记得删 `chroma_db` 重建索引

## 知识库支持格式（docs/ 目录）

| 格式 | 说明 |
| --- | --- |
| `.txt` | 直接放入即可 |
| `.md` | 直接放入即可；按标题层级切分（章节不拆散） |
| `.pdf` | 直接放入即可。文字版直接提取；扫描版（每页是图片）自动走 OCR 识别（V8.0：PyMuPDF 渲染 + RapidOCR 离线识别） |
| `.docx` | Word 文档，直接放入即可 |
| `.csv` | 直接放入即可；按行切分，一行一条记录 |
| `.xlsx` | Excel，直接放入即可；按行切分，表头字段名会拼进每一行数据（2.6 步起） |

> `docs/` 目录有文件时优先加载 docs/；为空或不存在时回退加载 `knowledge_base.txt`。
> 每个知识库文件的来源文件名会作为元数据保存，回答时会标注「来源」。

> **切分策略（2.5 步）**：不同格式语义单元不同，切分也按格式路由——
> `.md` 按标题层级切（MarkdownHeaderTextSplitter），`.csv`/`.xlsx` 按行切（一行一条记录），
> 其余格式（txt/pdf/docx）用通用切分器（RecursiveCharacterTextSplitter）。
>
> **表格处理（2.6 步）**：`.xlsx` 的第 1 行视为表头，数据行入库为 `字段名: 值` 形式
> （如 `客户行业: 金融 | 客户名称: 华信银行`），避免表头行与数据行分离导致"问字段名拿不到值"。
>
> **混合检索 + 重排序（第 3 步）**：检索从"单一向量 Top-k"升级为三环节流水线——
> ① BM25 关键词检索（jieba 分词，命中专有名词/精确词）
> ② 向量语义检索（命中意思相近的表述）
> ③ RRF 互惠排名融合两路结果（只比排名不比分数），再经 bge-reranker 精排，
> 只留 Top 6 进提示词。k 从"回答视野"（V7.7 的 k=6）变为"粗筛漏斗"。
> 手写实现 RRFRetriever / RerankerRetriever（官方 EnsembleRetriever 所在的
> langchain-retrievers 包装不上，见 mindmap 踩坑表）。
>
> **MultiQuery 多路检索（V8.2）**：问"李云龙妻子有几个"单路检索漏掉第一任妻子秀芹——
> 书里从不用"妻子"称呼她（婆娘/娶媳妇/新婚妻子），一个问法只匹配一种"说法"。
> 解法：LLM 把问题拆成多个子查询（"李云龙第一任妻子 杨秀芹"等），每路子查询独立
> 召回 + 精排（每路用【自己的子查询】打分），rank 融合合并去重。手写 MultiQueryRetriever
>（复用已有 CrossEncoder/ChatOpenAI，零新增依赖）。代价：每次问答多一次 LLM 调用
>（拆分子查询，约 1~2 秒）——换"枚举/集合类问题不漏全集"。

## 代码结构

| 文件 | 作用 |
| --- | --- |
| `main.py` | 普通问答：加载配置 → 创建模型 → 定义提示词 → 组合成链 → 运行 |
| `rag.py` | RAG 问答：多格式知识库 → 切分 → 嵌入 → Chroma 检索 → 生成 |
| `knowledge_base.txt` | 单文件知识库（docs/ 为空时的回退来源） |
| `docs/` | 多格式知识库目录（txt/md/pdf/docx/csv/xlsx） |
| `requirements.txt` | 依赖清单 |
| `.env.example` | 环境变量模板（复制为 `.env` 后填写） |

## 学习要点（按 main.py 中的顺序）

1. **`load_dotenv()`**：从 `.env` 读取配置，密钥不要写死在代码里。
2. **`ChatOpenAI(...)`**：创建一个模型对象，`temperature` 控制回答的随机性。
3. **`ChatPromptTemplate`**：提示词模板，用 `{变量}` 占位，运行时填充。
4. **`prompt | llm`**：LangChain 的核心语法——用管道符把组件连成一条链。
5. **`chain.invoke({...})`**：传入参数并执行整条链，得到回答。

## RAG 学习要点（rag.py）

RAG = 检索（Retrieval）+ 增强（Augmented）+ 生成（Generation），
即"先查资料，再回答"。与 main.py 相比多出以下环节：

| 环节 | 对应代码 | 作用 |
| --- | --- | --- |
| 加载文档 | `load_documents()`：`os.walk` 遍历 + `LOADER_MAP` 路由 | 扫描 docs/，按扩展名路由到对应加载函数（txt/md/pdf/docx/csv/xlsx）；PDF 逐页判定：扫描页自动 OCR（V8.0，带进度条 + 结果缓存到 ocr_cache/） |
| 切分 | `split_documents_by_format()` 按格式路由 | 长文本切小块；.md 按标题切、.csv/.xlsx 按行切、其余通用切分 |
| 嵌入 | `HuggingFaceEmbeddings` | 用本地开源模型把文本转成向量 |
| 存储 | `Chroma` | 向量库，索引落盘到 `./chroma_db`，重启后直接加载无需重建 |
| 检索 | `retriever`（V8.2 MultiQuery 多路检索：LLM 拆子查询 → 每路 BM25+向量 RRF 召回 → bge-reranker 精排 → rank 融合合并） | 解决"枚举/集合类问题"（有几个/都有谁）单路检索漏全集：一个问法只匹配一种"说法"，多路拆解覆盖不同说法后合并（实例：李云龙两任妻子 秀芹+田雨 都能召回）。每路内部仍是 BM25 关键词 + 向量语义 + RRF 融合 + reranker 精排。局限：万级全量列举/统计类问题仍靠 SQL（见 mindmap 核心概念 6） |
| 增强 | 把 `{context}` 塞进提示词 | 让模型"带着资料"作答，并标注来源 |

**建议体验**：先用 `main.py` 问"星辰科技是哪年成立的"（模型不知道，会编造），
再用 `rag.py` 问同样的问题，可以看到它基于知识库给出正确回答。
