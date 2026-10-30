"""MySQL 存储层：资源、文本块、检索日志。

中文检索方案说明
----------------
1. MySQL 默认 8.0 InnoDB FULLTEXT 不支持中文分词（只按空格/标点切词），
   直接建索引会让整段中文被当成一个 token，召回几乎为 0。
2. 这里采用 MySQL 官方推荐做法：为 FULLTEXT 索引指定 `WITH PARSER ngram`，
   ngram_token_size 默认为 2（binlog），把中文切成二元组（如"数据库" -> "数据""据库"），
   使用 `MATCH ... AGAINST (... IN BOOLEAN MODE)` 做短语查询。
   二元组比单词粒度更粗，所以额外乘上"精确命中加成"与 `IN NATURAL LANGUAGE MODE` 分数。
3. 同时保留 LIKE 兜底（短查询 <2 字时 ngram 无效，例如"图"），
   并在写入时把 title/course/text 合并进 `search_text` 冗余列，减少 join 成本。
"""
from __future__ import annotations

import re
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterable

import pymysql
from pymysql.cursors import DictCursor

import config

_LOCAL = threading.local()

STOPWORDS = {
    "的", "了", "和", "与", "及", "或", "在", "是", "有", "对", "中", "上", "下", "请问",
    "怎么", "如何", "什么", "一下", "相关", "资料", "内容", "文档", "课程", "一些", "哪些",
    "我想要", "帮我", "找", "查", "关于", "介绍", "讲讲", "说说", "有没有", "可以", "the", "a",
}
# 中文停用字（单字，仅用于构造 MATCH 查询短语时剔除高频虚词）
_STOP_CHARS = set("的地得了着和与及或在是有个这那我你他她它其吗呢吧啊")

_NGRAM_TOKEN_RE = re.compile(r"[a-zA-Z]+|\d{1,4}|[\u4e00-\u9fff]+")


def connect(with_db: bool = True) -> pymysql.connections.Connection:
    return pymysql.connect(
        host=config.MYSQL_HOST,
        port=config.MYSQL_PORT,
        user=config.MYSQL_USER,
        password=config.MYSQL_PASSWORD,
        database=config.MYSQL_DATABASE if with_db else None,
        charset="utf8mb4",
        cursorclass=DictCursor,
        autocommit=False,
    )


@contextmanager
def get_conn():
    conn = connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------- 初始化 ----------------
SCHEMA_STATEMENTS = [
    """CREATE TABLE IF NOT EXISTS courses (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(160) NOT NULL,
        code VARCHAR(64) DEFAULT NULL,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY uk_course_name (name)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE IF NOT EXISTS resources (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        resource_uid CHAR(32) NOT NULL,
        course_id INT DEFAULT NULL,
        course_name VARCHAR(160) NOT NULL DEFAULT '',
        title VARCHAR(255) NOT NULL,
        file_name VARCHAR(255) NOT NULL,
        file_type VARCHAR(16) NOT NULL,
        file_size INT NOT NULL DEFAULT 0,
        source_type VARCHAR(16) NOT NULL DEFAULT '',
        page_count INT NOT NULL DEFAULT 0,
        ocr_pages INT NOT NULL DEFAULT 0,
        status VARCHAR(16) NOT NULL DEFAULT 'ok',
        message VARCHAR(255) NOT NULL DEFAULT '',
        content MEDIUMTEXT,
        search_text MEDIUMTEXT,
        chunk_count INT NOT NULL DEFAULT 0,
        content_length INT NOT NULL DEFAULT 0,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        UNIQUE KEY uk_resource_uid (resource_uid),
        KEY idx_course (course_id),
        KEY idx_title (title),
        FULLTEXT KEY ft_search_text (search_text) WITH PARSER ngram
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE IF NOT EXISTS resource_chunks (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        resource_id BIGINT NOT NULL,
        chunk_index INT NOT NULL,
        content TEXT NOT NULL,
        point_id CHAR(36) NOT NULL,
        KEY idx_resource (resource_id),
        UNIQUE KEY uk_point (point_id),
        CONSTRAINT fk_chunk_resource FOREIGN KEY (resource_id) REFERENCES resources(id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE IF NOT EXISTS search_logs (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        query VARCHAR(255) NOT NULL,
        mode VARCHAR(16) NOT NULL,
        result_count INT NOT NULL DEFAULT 0,
        cost_ms INT NOT NULL DEFAULT 0,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        KEY idx_created (created_at),
        KEY idx_query (query)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
]


def init_db() -> None:
    """创建库与表；若库已存在但缺少 ngram 索引则自动补建。"""
    conn = connect(with_db=False)
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"CREATE DATABASE IF NOT EXISTS `{config.MYSQL_DATABASE}` "
                "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci"
            )
        conn.commit()
    finally:
        conn.close()

    with get_conn() as conn, conn.cursor() as cur:
        for stmt in SCHEMA_STATEMENTS:
            cur.execute(stmt)


def get_ngram_token_size(conn) -> int:
    with conn.cursor() as cur:
        try:
            cur.execute("SHOW VARIABLES LIKE 'ngram_token_size'")
            row = cur.fetchone()
            return int(row["Value"]) if row else 2
        except pymysql.MySQLError:
            return 2


# ---------------- 写入 ----------------
def get_or_create_course(conn, name: str) -> int | None:
    name = (name or "").strip()
    if not name:
        return None
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM courses WHERE name=%s", (name,))
        row = cur.fetchone()
        if row:
            return row["id"]
        cur.execute("INSERT INTO courses(name) VALUES(%s)", (name,))
        return cur.lastrowid


def insert_resource(conn, data: dict[str, Any]) -> int:
    sql = """INSERT INTO resources
        (resource_uid, course_id, course_name, title, file_name, file_type, file_size,
         source_type, page_count, ocr_pages, status, message, content, search_text,
         chunk_count, content_length)
        VALUES (%(resource_uid)s, %(course_id)s, %(course_name)s, %(title)s, %(file_name)s,
         %(file_type)s, %(file_size)s, %(source_type)s, %(page_count)s, %(ocr_pages)s,
         %(status)s, %(message)s, %(content)s, %(search_text)s, %(chunk_count)s, %(content_length)s)"""
    with conn.cursor() as cur:
        cur.execute(sql, data)
        return cur.lastrowid


def insert_chunks(conn, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO resource_chunks(resource_id, chunk_index, content, point_id) "
            "VALUES (%(resource_id)s, %(chunk_index)s, %(content)s, %(point_id)s)",
            rows,
        )


def update_resource(conn, resource_id: int, data: dict[str, Any]) -> None:
    """覆盖式更新（文件被重新导入时）。search_text 同步刷新，FULLTEXT 索引随之更新。"""
    sql = """UPDATE resources SET
        course_id=%(course_id)s, course_name=%(course_name)s, title=%(title)s,
        file_name=%(file_name)s, file_type=%(file_type)s, file_size=%(file_size)s,
        source_type=%(source_type)s, page_count=%(page_count)s, ocr_pages=%(ocr_pages)s,
        status=%(status)s, message=%(message)s, content=%(content)s, search_text=%(search_text)s,
        chunk_count=%(chunk_count)s, content_length=%(content_length)s
        WHERE resource_uid=%(resource_uid)s"""
    with conn.cursor() as cur:
        cur.execute(sql, {**data, "resource_uid": data["resource_uid"]})


def clean_search_text(title: str, course_name: str, content: str) -> str:
    """把标题、课程名加权重复写入 search_text：FULLTEXT 中词频越高相关度越高，
    这里让"标题/课程命中"天然获得更高 TF，从而在排序中被优先考虑。"""
    head = " ".join([
        (title or "") + "。",
        (title or "") + "。",
        (title or "") + "。",
        (course_name or "") + "。",
        (course_name or "") + "。",
    ])
    return head + (content or "")


def delete_chunks(conn, resource_id: int) -> list[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT point_id FROM resource_chunks WHERE resource_id=%s", (resource_id,))
        point_ids = [r["point_id"] for r in cur.fetchall()]
        cur.execute("DELETE FROM resource_chunks WHERE resource_id=%s", (resource_id,))
    return point_ids


# ---------------- 查询 ----------------
def get_resource_by_uid(conn, uid: str) -> dict | None:
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM resources WHERE resource_uid=%s", (uid,))
        return cur.fetchone()


def get_resource(conn, resource_id: int) -> dict | None:
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM resources WHERE id=%s", (resource_id,))
        return cur.fetchone()


def delete_stale_by_file(conn, file_name: str, keep_id: int) -> list[int]:
    """删除同名文件的历史记录（保留本次入库的 keep_id），返回被删除的资源 id。

    用于 uid 规则变更或同一文件换了课程/标题重新入库的场景，防止旧资源与其
    Qdrant 向量残留在库里影响检索结果。
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM resources WHERE file_name=%s AND id<>%s",
            (file_name, keep_id),
        )
        ids = [r["id"] for r in cur.fetchall()]
        if not ids:
            return []
        placeholders = ",".join(["%s"] * len(ids))
        cur.execute(f"DELETE FROM resource_chunks WHERE resource_id IN ({placeholders})", ids)
        cur.execute(f"DELETE FROM resources WHERE id IN ({placeholders})", ids)
    return ids


def delete_resource(conn, resource_id: int) -> dict | None:
    row = get_resource(conn, resource_id)
    if not row:
        return None
    delete_chunks(conn, resource_id)
    with conn.cursor() as cur:
        cur.execute("DELETE FROM resources WHERE id=%s", (resource_id,))
    return row


def list_resources(conn, limit: int = 100) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, resource_uid, course_name, title, file_name, file_type, source_type, "
            "page_count, ocr_pages, status, message, chunk_count, content_length, created_at "
            "FROM resources ORDER BY id DESC LIMIT %s",
            (limit,),
        )
        return cur.fetchall()


def metrics(conn) -> dict:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS c FROM resources")
        resources = cur.fetchone()["c"]
        cur.execute("SELECT COUNT(*) AS c FROM resource_chunks")
        chunks = cur.fetchone()["c"]
        cur.execute("SELECT status, COUNT(*) AS c FROM resources GROUP BY status")
        by_status = {r["status"]: r["c"] for r in cur.fetchall()}
    return {"resources": resources, "chunks": chunks, "by_status": by_status}


def log_search(conn, query: str, mode: str, count: int, cost_ms: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO search_logs(query, mode, result_count, cost_ms) VALUES(%s,%s,%s,%s)",
            (query[:255], mode, count, cost_ms),
        )


def hot_queries(conn, limit: int = 10, days: int = 30) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """SELECT query, COUNT(*) AS hits, MAX(created_at) AS last_at
               FROM search_logs
               WHERE created_at >= NOW() - INTERVAL %s DAY AND CHAR_LENGTH(query) > 1
               GROUP BY query ORDER BY hits DESC, last_at DESC LIMIT %s""",
            (days, limit),
        )
        return cur.fetchall()


# ---------------- 关键词检索 ----------------
def build_boolean_query(query: str, ngram_size: int = 2) -> tuple[str, list[str]]:
    """把查询转换为 BOOLEAN MODE 字符串。

    - 英文/数字：保留整词，加前缀通配 `word*`
    - 中文：按 ngram_token_size 切连续片段为短语（加引号保证顺序与邻近）
    - 过滤停用词，避免噪声词把排序带偏
    返回 (match_query, 高亮关键词列表)
    """
    match_parts: list[str] = []
    keywords: list[str] = []
    for token in _NGRAM_TOKEN_RE.findall(query):
        if token.lower() in STOPWORDS:
            continue
        if token.isascii():
            if len(token) < 2:
                continue
            match_parts.append(f"{token}*")
            keywords.append(token)
            continue
        cleaned = "".join(ch for ch in token if ch not in _STOP_CHARS)
        if len(cleaned) < ngram_size:
            if len(token) >= ngram_size:
                match_parts.append(f'"{token}"')
                keywords.append(token)
            continue
        # 长中文片段切成不超过 8 字的短语，兼顾 ngram 上限与召回
        for i in range(0, len(cleaned), 8):
            part = cleaned[i:i + 8]
            if len(part) >= ngram_size:
                match_parts.append(f'"{part}"')
        keywords.append(cleaned)
    return " ".join(match_parts), keywords


def keyword_search(conn, query: str, limit: int, course: str | None = None) -> list[dict]:
    """MySQL-only 检索：FULLTEXT(ngram) 排序 + 精确命中加成，返回资源级候选。

    评分构成（全部在 SQL / Python 内完成，不依赖其他服务）：
      ft_score      : MATCH ... AGAINST 相关度（ngram 相关度，已按文档长度归一化）
      exact_bonus   : 标题/课程名/正文中精确包含查询短语的加成（1.0 / 0.6 / 0.3）
      title_score   : 标题命中的额外加权
      final = ft_score * 1.0 + exact_bonus 权重化后再归一化
    """
    query = query.strip()
    if not query:
        return []

    ngram_size = get_ngram_token_size(conn)
    match_query, keywords = build_boolean_query(query, ngram_size)
    exact_phrase = "".join(keywords)

    params: list[Any] = []
    course_filter = ""
    if course:
        course_filter = " AND r.course_name = %s"
        params.append(course)

    rows: list[dict] = []
    with conn.cursor() as cur:
        if match_query:
            sql = f"""
                SELECT r.id, r.resource_uid, r.title, r.course_name, r.file_name, r.source_type,
                       r.status, r.content, r.chunk_count,
                       MATCH(r.search_text) AGAINST(%s IN NATURAL LANGUAGE MODE) AS ft_score
                FROM resources r
                WHERE MATCH(r.search_text) AGAINST(%s IN BOOLEAN MODE)
                      AND r.status <> 'error'{course_filter}
                ORDER BY ft_score DESC
                LIMIT %s
            """
            cur.execute(sql, [match_query, match_query, *params, limit * 3])
            rows = list(cur.fetchall())

        # LIKE 兜底：短查询 / ngram 未命中时保证"能查到"
        if len(rows) < limit:
            seen = {r["id"] for r in rows}
            like = f"%{query}%"
            sql = f"""
                SELECT r.id, r.resource_uid, r.title, r.course_name, r.file_name, r.source_type,
                       r.status, r.content, r.chunk_count, 0 AS ft_score
                FROM resources r
                WHERE r.status <> 'error' AND r.search_text LIKE %s{course_filter}
                LIMIT %s
            """
            cur.execute(sql, [like, *params, limit * 3])
            rows.extend([r for r in cur.fetchall() if r["id"] not in seen])

    results = []
    for row in rows:
        content = row.pop("content") or ""
        title = row["title"] or ""
        course_name = row["course_name"] or ""
        exact = 0.0
        if exact_phrase:
            if exact_phrase in title:
                exact += 1.0
            if exact_phrase in course_name:
                exact += 0.6
            if exact_phrase in content:
                exact += 0.3
        title_hit = 1.0 if exact_phrase and exact_phrase in title else 0.0
        raw = float(row["ft_score"] or 0.0) + exact * 10.0 + title_hit * 5.0

        snippet = make_snippet(content, keywords)
        chunk_score = row["ft_score"]

        results.append({
            "resource_id": row["id"],
            "resource_uid": row["resource_uid"],
            "title": title,
            "course_name": course_name,
            "file_name": row["file_name"],
            "source_type": row["source_type"],
            "status": row["status"],
            "snippet": snippet,
            "chunk_index": None,
            "keyword_score_raw": raw,
            "ft_score": float(row["ft_score"] or 0.0),
            "exact_hit": exact,
            "chunk_score": chunk_score,
            "aggregated_chunks": 1,
        })

    results.sort(key=lambda x: x["keyword_score_raw"], reverse=True)
    return results[:limit]


def make_snippet(content: str, keywords: list[str], width: int = 90) -> str:
    if not content:
        return ""
    content = content.replace("\n", " ")
    pos = -1
    for kw in keywords:
        if kw:
            pos = content.find(kw)
            if pos >= 0:
                break
    if pos < 0:
        return content[:width * 2] + ("…" if len(content) > width * 2 else "")
    start = max(0, pos - width)
    end = min(len(content), pos + width)
    return ("…" if start > 0 else "") + content[start:end] + ("…" if end < len(content) else "")
