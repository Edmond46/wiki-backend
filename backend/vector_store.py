"""Qdrant 向量库访问层。

- 每个文本块一个 point，point id 用 uuid5(namespace, resource_uid + chunk_index) 稳定生成，
  这样重复入库可幂等覆盖，且 point payload 中保存 resource_id / resource_uid，
  可以映射回 MySQL 记录（题目 B3 要求）。
- 支持 `server` 模式（docker 启动 Qdrant 服务）与 `local` 模式（本地文件，免服务端，便于演示）。
"""
from __future__ import annotations

import uuid
from functools import lru_cache

import config

POINT_NAMESPACE = uuid.UUID("6f1a3d0c-9a3b-4b8f-9a41-1b7c2d5e8f10")


def point_id_for(resource_uid: str, chunk_index: int) -> str:
    return str(uuid.uuid5(POINT_NAMESPACE, f"{resource_uid}:{chunk_index}"))


class VectorStore:
    def __init__(self) -> None:
        from qdrant_client import QdrantClient

        if config.QDRANT_MODE == "server":
            self.client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY or None)
        else:
            self.client = QdrantClient(path=config.QDRANT_PATH)
        self.collection = config.QDRANT_COLLECTION

    # ---------- 集合 ----------
    def ensure_collection(self, dim: int, recreate: bool = False) -> None:
        from qdrant_client import models

        exists = self.client.collection_exists(self.collection)
        if exists and recreate:
            self.client.delete_collection(self.collection)
            exists = False
        if not exists:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            )
            return
        info = self.client.get_collection(self.collection)
        current = info.config.params.vectors.size
        if current != dim:
            raise RuntimeError(
                f"Qdrant 集合 {self.collection} 维度为 {current}，与当前 Embedding 维度 {dim} 不一致；"
                "请更换 QDRANT_COLLECTION 名称或重建集合"
            )

    # ---------- 写入 ----------
    def upsert_chunks(self, vectors: list[list[float]], payloads: list[dict]) -> None:
        from qdrant_client import models

        points = [
            models.PointStruct(id=p["point_id"], vector=v, payload=p)
            for v, p in zip(vectors, payloads)
        ]
        self.client.upsert(collection_name=self.collection, points=points, wait=True)

    def delete_points(self, point_ids: list[str]) -> None:
        if not point_ids:
            return
        from qdrant_client import models

        self.client.delete(
            collection_name=self.collection,
            points_selector=models.PointIdsList(points=point_ids),
            wait=True,
        )

    def delete_by_resource(self, resource_id: int) -> None:
        from qdrant_client import models

        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="resource_id", match=models.MatchValue(value=resource_id))]
                )
            ),
            wait=True,
        )

    # ---------- 检索 ----------
    def search(self, vector: list[float], limit: int) -> list[dict]:
        """返回 [{point_id, score, payload}]，score 为 cosine 相似度（-1~1）。"""
        # qdrant-client >= 1.16 已移除 client.search，统一走 query_points
        if hasattr(self.client, "query_points"):
            resp = self.client.query_points(
                collection_name=self.collection,
                query=vector,
                limit=limit,
                with_payload=True,
            )
            hits = resp.points
        else:
            hits = self.client.search(
                collection_name=self.collection,
                query_vector=vector,
                limit=limit,
                with_payload=True,
            )
        return [
            {"point_id": str(h.id), "score": float(h.score), "payload": h.payload or {}}
            for h in hits
        ]

    def count(self) -> int:
        try:
            return self.client.count(self.collection, exact=True).count
        except Exception:  # noqa: BLE001 - 集合不存在等情况
            return 0


@lru_cache(maxsize=1)
def get_store() -> VectorStore:
    return VectorStore()
