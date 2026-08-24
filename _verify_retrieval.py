# 临时验证脚本：第三阶段混合检索 + 重排序（验证后删除）
import rag

print("=== 1. BM25 关键词检索（问：华信银行） ===")
for d in rag.bm25_retriever.invoke("华信银行")[:5]:
    print("-", d.page_content[:60])

print("\n=== 2. 向量语义检索（问：给零售客户推荐什么产品） ===")
for d in rag.vector_retriever.invoke("给零售客户推荐什么产品")[:5]:
    print("-", d.page_content[:60])

print("\n=== 3. RRF 融合 Top 10（问：所有客户的行业有哪些） ===")
for d in rag.ensemble_retriever.invoke("所有客户的行业有哪些")[:10]:
    print("-", d.page_content[:60])

print("\n=== 4. bge-reranker 精排 Top 6（问：所有客户的行业有哪些） ===")
for d in rag.retriever.invoke("所有客户的行业有哪些"):
    print("-", d.page_content[:60])

print("\n=== 5. 端到端问答 ===")
for q in ["所有客户的行业有哪些？", "华信银行是哪一年开始合作的？"]:
    r = rag.chain.invoke({"question": q})
    print(f"\nQ: {q}\nA: {r.content}")
