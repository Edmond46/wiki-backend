"""端到端测试脚本：生成示例资料 -> 入库 -> 两种模式检索对比 -> OCR 验证。

使用方式（需先启动/配置 MySQL）：
    python tests/e2e_test.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import config  # noqa: E402
import db  # noqa: E402
import search  # noqa: E402
from demo_data import build_samples  # noqa: E402
from pipeline import ingest_file, remove_resource  # noqa: E402
from vector_store import get_store  # noqa: E402

QUERIES = [
    ("精确课程/年份关键词", "2021 线性回归"),
    ("精确课程关键词", "事务 隔离级别"),
    ("自然语言描述型", "为什么生产环境要尽量避免长时间运行的大事务"),
    ("自然语言描述型", "怎样让模型训练收敛更快又不容易震荡"),
]
TOP_N = 5


def parse_check(samples: list[dict]) -> list[dict]:
    """对生成的文件做一次独立解析校验，确认文本型 PDF 与扫描件都能拿到文本。"""
    from extractor import parse_file

    rows = []
    for item in samples:
        result = parse_file(Path(item["path"]))
        rows.append({
            "file": Path(item["path"]).name,
            "status": result.status,
            "source_type": result.source_type,
            "pages": result.page_count,
            "ocr_pages": result.ocr_pages,
            "text_length": len(result.text),
            "message": result.message,
            "preview": result.text[:80].replace("\n", " "),
        })
    return rows


def rank_block(results: list[dict]) -> list[dict]:
    return [
        {
            "rank": i,
            "title": r["title"],
            "course": r["course_name"],
            "score": r.get("score"),
            "match_type": r.get("match_type"),
            "snippet": (r.get("snippet") or "")[:70].replace("\n", " "),
        }
        for i, r in enumerate(results[:TOP_N], start=1)
    ]


def main() -> None:
    db.init_db()
    samples = build_samples(config.BASE_DIR / "data" / "samples")

    print("=" * 80)
    print("0. 文件解析 / OCR 验证")
    print("=" * 80)
    for row in parse_check(samples):
        print(json.dumps(row, ensure_ascii=False))

    print("\n" + "=" * 80)
    print("1. 入库（文本型 PDF / 扫描 PDF / 图片）")
    print("=" * 80)
    resource_ids = []
    for item in samples:
        res = ingest_file(Path(item["path"]), course_name=item["course"], title=item["title"])
        resource_ids.append(res["resource_id"])
        print(json.dumps({k: res[k] for k in
                          ("resource_id", "title", "status", "source_type", "page_count",
                           "ocr_pages", "chunk_count", "content_length", "cost_ms", "message")},
                         ensure_ascii=False))

    with db.get_conn() as conn:
        print("MySQL 统计:", db.metrics(conn))
    print("Qdrant 向量数:", get_store().count())

    print("\n" + "=" * 80)
    print("2. 两种模式检索对比")
    print("=" * 80)
    report = []
    for label, query in QUERIES:
        kw = search.keyword_mode(query, TOP_N)
        smart = search.smart_mode(query, TOP_N)
        entry = {
            "label": label,
            "query": query,
            "keyword_cost_ms": kw["cost_ms"],
            "smart_cost_ms": smart["cost_ms"],
            "fusion": smart["fusion"],
            "keyword": rank_block(kw["results"]),
            "smart": rank_block(smart["results"]),
        }
        report.append(entry)
        print(f"\n### [{label}] 查询: {query}")
        print(f"  动态权重: 关键词 {smart['fusion']['weights']['keyword']} / "
              f"语义 {smart['fusion']['weights']['semantic']}   查询倾向: "
              f"{smart['fusion']['query_analysis']['query_type']}")
        print("  ① 关键词检索（仅 MySQL）:")
        for r in entry["keyword"] or [{"rank": "-", "title": "无结果", "snippet": ""}]:
            print(f"     {r['rank']}. {r['title']}  {r['snippet']}")
        print("  ② 智能检索（融合）:")
        for r in entry["smart"] or [{"rank": "-", "match_type": "-", "title": "无结果", "snippet": ""}]:
            print(f"     {r['rank']}. [{r['match_type']}] {r['title']}  {r['snippet']}")

    print("\n" + "=" * 80)
    print("3. 索引删除同步验证")
    print("=" * 80)
    target = resource_ids[0]
    before = 0
    with db.get_conn() as conn:
        before = db.metrics(conn)["chunks"]
    remove_resource(target)
    with db.get_conn() as conn:
        after = db.metrics(conn)
    print(f"删除资源 {target} 前 chunk 数 {before} -> 后 {after['chunks']}，"
          f"Qdrant 向量数 {get_store().count()}")

    # 重新入库，保持数据完整，便于演示
    for item in samples:
        ingest_file(Path(item["path"]), course_name=item["course"], title=item["title"])

    out = config.BASE_DIR / "tests" / "report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n测试报告已写入 {out}")


if __name__ == "__main__":
    if not os.environ.get("MYSQL_PASSWORD") and not (config.BASE_DIR / ".env").exists():
        print("提示：未检测到 .env，将使用默认 MySQL 配置（root/root@127.0.0.1:3306）")
    main()
