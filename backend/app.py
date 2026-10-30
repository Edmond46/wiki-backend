import json
import shutil
import time
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

import config
import db
import search
from pipeline import ingest_file, remove_resource
from vector_store import get_store

app = Flask(__name__, static_folder=str(config.BASE_DIR / "web"), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_MB * 1024 * 1024


def ok(data, **extra):
    return jsonify({"ok": True, "data": data, **extra})


def fail(message: str, code: int = 400, **extra):
    return jsonify({"ok": False, "error": message, **extra}), code


@app.errorhandler(413)
def too_large(_):
    return fail(f"文件超过 {config.MAX_CONTENT_MB}MB 限制", 413)


# ---------------- 页面 ----------------
@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.get("/api/health")
def health():
    info = {
        "mysql": False,
        "qdrant": False,
        "qdrant_points": 0,
        "embedding": config.EMBEDDING_PROVIDER,
        "qdrant_mode": config.QDRANT_MODE,
    }
    try:
        with db.get_conn() as conn:
            info["mysql"] = True
            info.update(db.metrics(conn))
    except Exception as exc:  # noqa: BLE001
        info["mysql_error"] = str(exc)
    try:
        info["qdrant_points"] = get_store().count()
        info["qdrant"] = True
    except Exception as exc:  # noqa: BLE001
        info["qdrant_error"] = str(exc)
    return ok(info)


# ---------------- 文件入库 ----------------
@app.post("/api/resources/upload")
def upload():
    files = request.files.getlist("files") or ([request.files["file"]] if "file" in request.files else [])
    if not files:
        return fail("未收到文件，请使用 form-data 字段 files")
    course = request.form.get("course", "")
    tmp_dir = config.UPLOAD_DIR / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for fs in files:
        if not fs.filename:
            continue
        suffix = Path(fs.filename).suffix.lower()
        tmp_path = tmp_dir / f"{uuid.uuid4().hex}{suffix}"
        fs.save(tmp_path)
        try:
            result = ingest_file(tmp_path, course_name=course, title=Path(fs.filename).stem)
            saved = config.UPLOAD_DIR / f"{result['resource_uid']}{suffix}"
            shutil.move(str(tmp_path), saved)
            results.append(result)
        except Exception as exc:  # noqa: BLE001
            results.append({"file_name": fs.filename, "status": "error", "message": str(exc)})
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
    return ok(results)


@app.get("/api/resources")
def list_resources():
    limit = max(1, min(int(request.args.get("limit", 100)), 500))
    with db.get_conn() as conn:
        rows = db.list_resources(conn, limit)
    for r in rows:
        if r.get("created_at"):
            r["created_at"] = r["created_at"].isoformat(sep=" ", timespec="seconds")
    return ok(rows)


@app.get("/api/resources/<int:resource_id>")
def resource_detail(resource_id: int):
    with db.get_conn() as conn:
        row = db.get_resource(conn, resource_id)
        if not row:
            return fail("资源不存在", 404)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, chunk_index, content, point_id FROM resource_chunks "
                "WHERE resource_id=%s ORDER BY chunk_index",
                (resource_id,),
            )
            chunks = cur.fetchall()
    row.pop("search_text", None)
    for key in ("created_at", "updated_at"):
        if row.get(key):
            row[key] = row[key].isoformat(sep=" ", timespec="seconds")
    row["content"] = (row.get("content") or "")[:3000]
    return ok({"resource": row, "chunks": chunks})


@app.delete("/api/resources/<int:resource_id>")
def delete_resource(resource_id: int):
    if not remove_resource(resource_id):
        return fail("资源不存在", 404)
    return ok({"deleted": resource_id})


# ---------------- 检索 ----------------
@app.get("/api/search")
def api_search():
    query = (request.args.get("q") or "").strip()
    mode = (request.args.get("mode") or "smart").lower()
    course = (request.args.get("course") or "").strip() or None
    limit = max(1, min(int(request.args.get("limit", 10)), 50))
    if not query:
        return fail("缺少查询参数 q")

    if mode == "keyword":
        data = search.keyword_mode(query, limit, course)
    elif mode == "smart":
        data = search.smart_mode(query, limit, course)
    else:
        return fail("mode 只能是 keyword 或 smart")
    return ok(data)


@app.get("/api/search/compare")
def api_compare():
    """同一条查询同时跑两种模式，直接对比（题目 B6 要求）。"""
    query = (request.args.get("q") or "").strip()
    limit = max(1, min(int(request.args.get("limit", 10)), 50))
    if not query:
        return fail("缺少查询参数 q")
    kw = search.keyword_mode(query, limit)
    smart = search.smart_mode(query, limit)

    kw_rank = {r["resource_id"]: i for i, r in enumerate(kw["results"], start=1)}
    sm_rank = {r["resource_id"]: i for i, r in enumerate(smart["results"], start=1)}
    all_ids = list(dict.fromkeys(list(kw_rank) + list(sm_rank)))
    titles = {r["resource_id"]: r["title"] for r in kw["results"] + smart["results"]}
    rows = [
        {
            "resource_id": rid,
            "title": titles.get(rid, ""),
            "keyword_rank": kw_rank.get(rid),
            "smart_rank": sm_rank.get(rid),
            "rank_delta": (kw_rank.get(rid, 99) - sm_rank.get(rid, 99)),
        }
        for rid in all_ids
    ]
    return ok({"query": query, "keyword": kw, "smart": smart, "diff": rows})


@app.get("/api/hot")
def api_hot():
    limit = max(1, min(int(request.args.get("limit", 10)), 50))
    return ok(search.hot_queries(limit))


# ---------------- 演示数据 ----------------
@app.post("/api/demo/sample")
def load_sample():
    """生成演示用资料（文本型 PDF / 扫描图片型 PDF / 图片），便于一键验证全流程。"""
    from demo_data import build_samples

    samples = build_samples(config.BASE_DIR / "data" / "samples")
    results = []
    for item in samples:
        results.append(ingest_file(Path(item["path"]), course_name=item["course"], title=item["title"]))
    return ok(results)


@app.post("/api/admin/reindex")
def reindex():
    """把 MySQL 中已有资源重新向量化写入 Qdrant（Qdrant 数据丢失/换 Embedding 时使用）。"""
    from embedding import get_embedder
    from vector_store import point_id_for

    embedder = get_embedder()
    store = get_store()
    store.ensure_collection(embedder.dim, recreate=True)
    count = 0
    with db.get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT id, resource_uid, title, course_name FROM resources")
        resources = cur.fetchall()
        for res in resources:
            cur.execute(
                "SELECT chunk_index, content FROM resource_chunks WHERE resource_id=%s ORDER BY chunk_index",
                (res["id"],),
            )
            rows = cur.fetchall()
            if not rows:
                continue
            vectors = embedder.embed([r["content"] for r in rows])
            payloads = [{
                "point_id": point_id_for(res["resource_uid"], r["chunk_index"]),
                "resource_id": res["id"],
                "resource_uid": res["resource_uid"],
                "chunk_index": r["chunk_index"],
                "title": res["title"],
                "course_name": res["course_name"],
                "text": r["content"],
            } for r in rows]
            store.upsert_chunks(vectors, payloads)
            count += len(rows)
    return ok({"reindexed_points": count})


if __name__ == "__main__":
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    try:
        db.init_db()
        print(f"[ok] MySQL 初始化完成: {config.MYSQL_HOST}:{config.MYSQL_PORT}/{config.MYSQL_DATABASE}")
    except Exception as exc:  # noqa: BLE001 - 数据库未就绪时仍启动，便于查看 /api/health 排查
        print(f"[warn] MySQL 不可用，检索接口会返回错误，请检查配置: {exc}")
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)
