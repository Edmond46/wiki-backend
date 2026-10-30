"""离线自检脚本：不需要 MySQL / Qdrant 服务，验证解析、OCR 与融合排序逻辑。

运行：python tests/offline_check.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import config  # noqa: E402
from demo_data import build_samples  # noqa: E402
from extractor import parse_file, split_chunks  # noqa: E402
from search import fuse  # noqa: E402


def main() -> None:
    print("=" * 70)
    print("1) 演示文件生成与解析（含 OCR）")
    print("=" * 70)
    samples = build_samples(config.BASE_DIR / "data" / "samples")
    for item in samples:
        path = Path(item["path"])
        res = parse_file(path)
        chunks = split_chunks(res.text)
        print(json.dumps({
            "file": path.name,
            "status": res.status,
            "source_type": res.source_type,
            "pages": res.page_count,
            "ocr_pages": res.ocr_pages,
            "chars": len(res.text),
            "chunks": len(chunks),
            "message": res.message,
            "preview": res.text[:60].replace("\n", " "),
        }, ensure_ascii=False))

    print("\n" + "=" * 70)
    print("2) 融合排序（加权 RRF）逻辑验证 —— 使用模拟的两路召回")
    print("=" * 70)
    # 模拟：A 只在关键词召回第 1，C 只在语义召回第 1，B 两路都有
    keyword = [
        {"resource_id": 1, "resource_uid": "a", "title": "A 精准关键词命中", "course_name": "数据库",
         "file_name": "a.pdf", "source_type": "pdf-text", "status": "ok", "snippet": "事务隔离",
         "keyword_score_raw": 120.0, "ft_score": 90.0, "exact_hit": 1.0},
        {"resource_id": 2, "resource_uid": "b", "title": "B 两路都命中", "course_name": "数据库",
         "file_name": "b.pdf", "source_type": "pdf-text", "status": "ok", "snippet": "并发控制",
         "keyword_score_raw": 80.0, "ft_score": 50.0, "exact_hit": 0.6},
        {"resource_id": 2, "resource_uid": "b", "title": "B 两路都命中", "course_name": "数据库",
         "file_name": "b.pdf", "source_type": "pdf-text", "status": "ok", "snippet": "B 的第二个块",
         "keyword_score_raw": 10.0, "ft_score": 5.0, "exact_hit": 0.0},
    ]
    semantic = [
        {"resource_id": 2, "resource_uid": "b", "title": "B 两路都命中", "course_name": "数据库",
         "file_name": "b.pdf", "source_type": "pdf-text", "status": "ok", "snippet": "长事务导致日志膨胀",
         "chunk_index": 0, "semantic_score": 0.83},
        {"resource_id": 3, "resource_uid": "c", "title": "C 仅语义命中", "course_name": "数据库",
         "file_name": "c.pdf", "source_type": "pdf-text", "status": "ok", "snippet": "避免大事务",
         "chunk_index": 4, "semantic_score": 0.79},
    ]
    for query in ["2021 线性回归", "为什么生产环境要避免长时间运行的大事务"]:
        fused, meta = fuse(keyword, semantic, query)
        print(f"\n查询: {query}")
        print(f"  动态权重: {meta['weights']}  查询倾向: {meta['query_analysis']['query_type']}")
        for i, r in enumerate(fused, start=1):
            print(f"  {i}. {r['title']}  score={r['score']}  kw_rank={r['keyword_rank']} "
                  f"sem_rank={r['semantic_rank']} matched_chunks={r['matched_chunks']} type={r['match_type']}")
        ids = [r["resource_id"] for r in fused]
        assert len(ids) == len(set(ids)), "资源级聚合失败：出现重复资源"
    print("\nOK：同一资源的多个块被聚合为一条，未出现重复。")


if __name__ == "__main__":
    main()
