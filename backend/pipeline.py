"""文档入库流水线：解析 -> 分块 -> 写 MySQL -> 写 Qdrant。"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import config
import db
from embedding import get_embedder
from extractor import parse_file, split_chunks
from vector_store import get_store, point_id_for


def _resource_uid(path: Path, course_name: str, title: str) -> str:
    """稳定的资源标识：文件名 + 课程名 + 标题。

    刻意不使用文件内容或大小做 uid：同一份资料被重新导出/重新上传时字节必然变化，
    但业务上是"同一个资源"，用内容做 uid 会让每次都新增一条记录并留下 Qdrant 孤儿向量。
    文件名 + 课程 + 标题既保证重复上传幂等覆盖，也能区分同名但课程/标题不同的文件。
    """
    key = f"{path.name}:{course_name}:{title}"
    return hashlib.md5(key.encode()).hexdigest()


def _safe_title(path: Path, override: str | None) -> str:
    return (override or path.stem)[:255]


def _cleanup_resource_indexes(conn, resource_id: int, resource_uid: str) -> None:
    """删除旧索引（MySQL chunk + Qdrant points），保证更新后两端同步。"""
    db.delete_chunks(conn, resource_id)
    try:
        get_store().delete_by_resource(resource_id)
    except Exception:  # noqa: BLE001 - 集合可能尚不存在
        pass


def ingest_file(file_path: Path, course_name: str = "", title: str | None = None) -> dict[str, Any]:
    file_path = Path(file_path)
    started = time.perf_counter()
    parse = parse_file(file_path)

    course_name = (course_name or "").strip()[:160]
    resource_title = _safe_title(file_path, title)
    uid = _resource_uid(file_path, course_name, resource_title)

    chunks = split_chunks(parse.text) if parse.status == "ok" else []
    if parse.status == "ok" and not chunks:
        parse.status = "empty"
        parse.message = "正文切分后为空"

    search_text = db.clean_search_text(resource_title, course_name, parse.text)

    record = {
        "resource_uid": uid,
        "course_id": None,
        "course_name": course_name,
        "title": resource_title,
        "file_name": file_path.name[:255],
        "file_type": file_path.suffix.lower().lstrip(".")[:16],
        "file_size": file_path.stat().st_size if file_path.exists() else 0,
        "source_type": parse.source_type or "unknown",
        "page_count": parse.page_count,
        "ocr_pages": parse.ocr_pages,
        "status": parse.status,
        "message": (parse.message or "; ".join(parse.warnings))[:255],
        "content": parse.text,
        "search_text": search_text,
        "chunk_count": len(chunks),
        "content_length": len(parse.text),
    }

    embedder = get_embedder()
    store = get_store()
    store.ensure_collection(embedder.dim)

    vectors: list[list[float]] = []
    payloads: list[dict] = []
    if chunks:
        vectors = embedder.embed(chunks)

    with db.get_conn() as conn:
        record["course_id"] = db.get_or_create_course(conn, course_name)
        existing = db.get_resource_by_uid(conn, uid)
        if existing:
            resource_id = existing["id"]
            db.update_resource(conn, resource_id, record)
            _cleanup_resource_indexes(conn, resource_id, uid)
        else:
            resource_id = db.insert_resource(conn, record)
            # uid 变更（同名文件换了课程/标题）时清掉旧记录及其向量，避免留下孤儿索引
            stale_ids = db.delete_stale_by_file(conn, file_path.name, resource_id)
            for stale_id in stale_ids:
                try:
                    store.delete_by_resource(stale_id)
                except Exception:  # noqa: BLE001
                    pass

        chunk_rows = []
        for idx, (text, vector) in enumerate(zip(chunks, vectors)):
            point_id = point_id_for(uid, idx)
            chunk_rows.append({
                "resource_id": resource_id,
                "chunk_index": idx,
                "content": text,
                "point_id": point_id,
            })
            payloads.append({
                "point_id": point_id,
                "resource_id": resource_id,
                "resource_uid": uid,
                "chunk_index": idx,
                "title": resource_title,
                "course_name": course_name,
                "text": text,
            })
        db.insert_chunks(conn, chunk_rows)

    if payloads:
        store.upsert_chunks(vectors, payloads)

    from search import cache_hot_queries

    cache_hot_queries()

    return {
        "resource_id": resource_id,
        "resource_uid": uid,
        "title": resource_title,
        "course_name": course_name,
        "status": parse.status,
        "message": parse.message,
        "source_type": parse.source_type,
        "page_count": parse.page_count,
        "ocr_pages": parse.ocr_pages,
        "chunk_count": len(chunks),
        "content_length": len(parse.text),
        "preview": parse.text[:300],
        "updated": bool(existing),
        "cost_ms": int((time.perf_counter() - started) * 1000),
    }


def remove_resource(resource_id: int) -> bool:
    """删除资源时同步删除 MySQL chunk 与 Qdrant 向量。"""
    with db.get_conn() as conn:
        row = db.delete_resource(conn, resource_id)
    if not row:
        return False
    try:
        get_store().delete_points([point_id_for(row["resource_uid"], i) for i in range(row["chunk_count"] + 1)])
        get_store().delete_by_resource(resource_id)
    except Exception:  # noqa: BLE001
        pass
    from search import cache_hot_queries

    cache_hot_queries()
    return True
