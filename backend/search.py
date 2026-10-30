"""两种检索模式与加权 RRF 融合排序。

融合方案：Weighted Reciprocal Rank Fusion（加权 RRF）
--------------------------------------------------
1. 两路检索的分数尺度完全不同：
   - MySQL 的 MATCH..AGAINST 相关度是"无上界的正数"，取决于文档长度与词频；
   - Qdrant 返回 cosine 相似度（-1~1），整体集中在 0.4~0.9 的窄区间。
   直接相加会让 MySQL 的分数完全主导（数值大 1~2 个数量级），所以不能相加。
2. RRF 只用"排名"不用"原始分值"：score(d) = Σ_i w_i / (k + rank_i(d))
   - 与尺度无关，天然可融合异构检索；
   - k=60（原论文经验值）弱化头部排名的绝对优势，让两路的 Top10 都能贡献。
3. 权重 w_i 由查询意图动态决定（加分项）：
   - 查询里含"精确信号"（课程代号、年份、英文缩写、引号短语、专有名词长度短的实体词）
     时提高关键词权重，因为精确 token 命中是 MySQL 的强项；
   - 查询是长句/口语化/疑问句时提高语义权重，因为自然语言描述词面匹配不上。
   最终做 min-max 归一化到 [0,1] 方便解释，但不影响排序。
4. 资源级聚合：同一文件多个文本块，在该资源内部只取最好的一条 chunk 参与排名
   （chunk 级 RRF 分数取 max），资源之间再按 RRF 分数排序，避免同一文件重复出现；
   同时把其余命中块作为"相关片段"附在结果中。
"""
from __future__ import annotations

import re
import time
from typing import Any

import config
import db
from embedding import get_embedder
from vector_store import get_store

_EXACT_SIGNAL_RE = re.compile(r"[A-Za-z]{2,}\d*|\d{4}|\"[^\"]+\"|『[^』]+』|“([^”]+)”")
_NL_SIGNAL_RE = re.compile(r"[，。？！,.?!]|怎么|如何|为什么|哪些|哪种|介绍|讲讲|区别|原理|实现|步骤|流程|作用|概念")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

# ---------------- 热词缓存（选做加分项） ----------------
_HOT_CACHE: dict[str, Any] = {"data": [], "expire_at": 0.0}


def hot_queries(limit: int = 10) -> list[dict]:
    """热词排行，带 TTL 缓存，避免每次搜索都扫 search_logs。"""
    now = time.time()
    if _HOT_CACHE["expire_at"] > now:
        return _HOT_CACHE["data"][:limit]
    with db.get_conn() as conn:
        rows = db.hot_queries(conn, limit=50)
    _HOT_CACHE["data"] = rows
    _HOT_CACHE["expire_at"] = now + config.HOT_CACHE_TTL
    return rows[:limit]


def cache_hot_queries() -> None:
    _HOT_CACHE["expire_at"] = 0.0


def analyze_query(query: str) -> dict[str, Any]:
    """判断查询偏向"精确关键词"还是"自然语言描述"，用于动态权重。"""
    exact_hits = _EXACT_SIGNAL_RE.findall(query)
    nl_hits = _NL_SIGNAL_RE.findall(query)
    cjk_len = len(_CJK_RE.findall(query))
    token_count = len(re.findall(r"[A-Za-z0-9]+", query))
    q_len = len(query.strip())

    exact_strength = len(exact_hits) * 1.0 + (0.6 if q_len <= 8 else 0.0) + token_count * 0.3
    nl_strength = len(nl_hits) * 1.0 + (1.0 if q_len > 16 else 0.0) + (0.5 if cjk_len > 12 else 0.0)
    delta = exact_strength - nl_strength

    return {
        "exact_hits": exact_hits,
        "nl_hits": nl_hits,
        "exact_strength": round(exact_strength, 2),
        "nl_strength": round(nl_strength, 2),
        "query_type": "keyword-leaning" if delta >= 0 else "semantic-leaning",
        "delta": round(delta, 2),
    }


def dynamic_weights(query: str) -> tuple[float, float, dict[str, Any]]:
    info = analyze_query(query)
    # sigmoid 把 delta 压到 (-1,1)，再映射到权重区间
    import math

    z = math.tanh(info["delta"] / max(config.QUERY_WEIGHT_SENSITIVITY, 1e-6))
    w_keyword = config.WEIGHT_KEYWORD * (1 + 0.5 * z)
    w_semantic = config.WEIGHT_SEMANTIC * (1 - 0.5 * z)
    total = w_keyword + w_semantic
    return w_keyword / total, w_semantic / total, info


# ---------------- 语义检索 ----------------
def semantic_search(query: str, limit: int) -> list[dict]:
    """Qdrant-only 检索：返回 chunk 级命中，含映射回 MySQL 的资源信息。"""
    embedder = get_embedder()
    store = get_store()
    vector = embedder.embed_one(query)
    hits = store.search(vector, limit)
    if not hits:
        return []

    resource_ids = sorted({int(h["payload"].get("resource_id", 0)) for h in hits} - {0})
    meta: dict[int, dict] = {}
    if resource_ids:
        with db.get_conn() as conn, conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(resource_ids))
            cur.execute(
                f"SELECT id, resource_uid, title, course_name, file_name, source_type, status "
                f"FROM resources WHERE id IN ({placeholders})",
                resource_ids,
            )
            meta = {r["id"]: r for r in cur.fetchall()}

    results = []
    for h in hits:
        payload = h["payload"]
        rid = int(payload.get("resource_id", 0))
        info = meta.get(rid)
        if not info or info["status"] == "error":
            continue
        results.append({
            "resource_id": rid,
            "resource_uid": info["resource_uid"],
            "title": info["title"],
            "course_name": info["course_name"],
            "file_name": info["file_name"],
            "source_type": info["source_type"],
            "status": info["status"],
            "snippet": (payload.get("text") or "")[:200],
            "chunk_index": payload.get("chunk_index"),
            "point_id": h["point_id"],
            "semantic_score": h["score"],     # cosine 相似度
        })
    return results


# ---------------- 资源级聚合 ----------------
def aggregate_to_resources(chunk_hits: list[dict]) -> list[dict]:
    """同一资源的多个块聚合为一条：rank 用该资源最佳块的名次，其余块作为相关片段。"""
    best: dict[int, dict] = {}
    for rank, hit in enumerate(chunk_hits, start=1):
        rid = hit["resource_id"]
        hit = dict(hit, chunk_rank=rank)
        if rid not in best:
            best[rid] = hit
            best[rid]["other_chunks"] = []
        else:
            best[rid]["other_chunks"].append(hit)
    return list(best.values())


# ---------------- 融合 ----------------
def fuse(keyword_chunks: list[dict], semantic_chunks: list[dict], query: str,
         explain: bool = True) -> tuple[list[dict], dict[str, Any]]:
    w_kw, w_sem, info = dynamic_weights(query)
    k = config.RRF_K

    kw_res = aggregate_to_resources(keyword_chunks)
    sem_res = aggregate_to_resources(semantic_chunks)

    scores: dict[int, dict] = {}

    def _entry(rid: int, base: dict) -> dict:
        return scores.setdefault(rid, {
            "resource_id": rid,
            "resource_uid": base["resource_uid"],
            "title": base["title"],
            "course_name": base["course_name"],
            "file_name": base["file_name"],
            "source_type": base["source_type"],
            "status": base["status"],
            "rrf_keyword": 0.0,
            "rrf_semantic": 0.0,
            "keyword_rank": None,
            "semantic_rank": None,
            "keyword_score_raw": base.get("keyword_score_raw"),
            "ft_score": base.get("ft_score"),
            "exact_hit": base.get("exact_hit", 0.0),
            "semantic_score": None,
            "snippet": base.get("snippet", ""),
            "best_chunk_index": base.get("chunk_index"),
            "matched_chunks": 0,
        })

    # 资源级排名参与 RRF（每个资源只用最佳块的名次，避免多块刷分）
    for rank, item in enumerate(kw_res, start=1):
        e = _entry(item["resource_id"], item)
        e["rrf_keyword"] = w_kw / (k + rank)
        e["keyword_rank"] = rank
        e["matched_chunks"] += 1 + len(item.get("other_chunks", []))
        if not e["snippet"]:
            e["snippet"] = item.get("snippet", "")

    for rank, item in enumerate(sem_res, start=1):
        e = _entry(item["resource_id"], item)
        e["rrf_semantic"] = w_sem / (k + rank)
        e["semantic_rank"] = rank
        e["semantic_score"] = item["semantic_score"]
        e["matched_chunks"] += 1 + len(item.get("other_chunks", []))
        if not e["snippet"]:
            e["snippet"] = item.get("snippet", "")

    results = list(scores.values())
    for r in results:
        r["rrf_score"] = r["rrf_keyword"] + r["rrf_semantic"]

    ordered = sorted(results, key=lambda x: x["rrf_score"], reverse=True)
    max_score = ordered[0]["rrf_score"] if ordered else 0.0
    for r in ordered:
        r["score"] = round(r["rrf_score"] / max_score, 4) if max_score else 0.0
        r["rrf_keyword"] = round(r["rrf_keyword"], 6)
        r["rrf_semantic"] = round(r["rrf_semantic"], 6)
        r["rrf_score"] = round(r["rrf_score"], 6)
        r["match_type"] = (
            "both" if r["keyword_rank"] and r["semantic_rank"]
            else "keyword-only" if r["keyword_rank"] else "semantic-only"
        )

    meta = {
        "method": "weighted_rrf",
        "rrf_k": k,
        "weights": {"keyword": round(w_kw, 4), "semantic": round(w_sem, 4)},
        "query_analysis": info if explain else None,
        "candidates": {"keyword": len(kw_res), "semantic": len(sem_res), "fused": len(ordered)},
    }
    return ordered, meta


# ---------------- 对外入口 ----------------
def keyword_mode(query: str, limit: int, course: str | None = None) -> dict[str, Any]:
    start = time.perf_counter()
    with db.get_conn() as conn:
        hits = db.keyword_search(conn, query, limit, course)
        cost = int((time.perf_counter() - start) * 1000)
        try:
            db.log_search(conn, query, "keyword", len(hits), cost)
        except Exception:  # noqa: BLE001 - 日志失败不影响检索
            pass

    results = []
    max_raw = max((h["keyword_score_raw"] for h in hits), default=0.0)
    for rank, h in enumerate(hits, start=1):
        h = dict(h)
        h["rank"] = rank
        h["score"] = round(h["keyword_score_raw"] / max_raw, 4) if max_raw else 0.0
        h["keyword_rank"] = rank
        h["match_type"] = "keyword"
        results.append(h)
    return {"mode": "keyword", "query": query, "total": len(results), "cost_ms": cost, "results": results}


def smart_mode(query: str, limit: int, course: str | None = None) -> dict[str, Any]:
    start = time.perf_counter()
    with db.get_conn() as conn:
        kw_hits = db.keyword_search(conn, query, config.RECALL_LIMIT, course)
    sem_hits = semantic_search(query, config.RECALL_LIMIT)

    if course and sem_hits:
        sem_hits = [h for h in sem_hits if h["course_name"] == course]

    fused, meta = fuse(kw_hits, sem_hits, query)
    cost = int((time.perf_counter() - start) * 1000)
    results = fused[:limit]
    with db.get_conn() as conn:
        try:
            db.log_search(conn, query, "smart", len(results), cost)
        except Exception:  # noqa: BLE001
            pass

    return {
        "mode": "smart",
        "query": query,
        "total": len(results),
        "cost_ms": cost,
        "fusion": meta,
        "results": results,
    }
