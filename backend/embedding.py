"""Embedding 提供方：阿里云百炼（DashScope，OpenAI 兼容接口）/ 离线哈希向量。"""
from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache

import config

_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+")


def _tokens(text: str) -> list[str]:
    """中文按字 + 二元组，英文/数字按词，兼顾无分词器的场景。"""
    raw = _TOKEN_RE.findall(text.lower())
    out: list[str] = []
    prev_cjk = ""
    for tok in raw:
        out.append(tok)
        if len(tok) == 1 and "\u4e00" <= tok <= "\u9fff":
            if prev_cjk:
                out.append(prev_cjk + tok)
            prev_cjk = tok
        else:
            prev_cjk = ""
    return out


def _hash_vector(text: str, dim: int) -> list[float]:
    """Hashing Trick 生成定长向量：无需任何外部服务，用于离线演示与测试。"""
    vec = [0.0] * dim
    for tok in _tokens(text):
        digest = hashlib.md5(tok.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        weight = 1.0 + (len(tok) - 1) * 0.25          # 词组权重略高
        vec[idx] += sign * weight
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        vec[0] = 1.0
        return vec
    return [v / norm for v in vec]


class BaseEmbedder:
    name = "base"
    dim = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class HashEmbedder(BaseEmbedder):
    name = "hash"

    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim or config.EMBEDDING_DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [_hash_vector(t, self.dim) for t in texts]


class DashScopeEmbedder(BaseEmbedder):
    name = "dashscope"

    def __init__(self) -> None:
        from openai import OpenAI

        if not config.DASHSCOPE_API_KEY:
            raise RuntimeError("EMBEDDING_PROVIDER=dashscope 但未配置 DASHSCOPE_API_KEY")
        self.client = OpenAI(api_key=config.DASHSCOPE_API_KEY, base_url=config.DASHSCOPE_BASE_URL)
        self.model_name = config.DASHSCOPE_MODEL
        self.dim = config.EMBEDDING_DIM
        # 先用一条样例探测真实维度，避免与 Qdrant 集合维度不一致
        probe = self.client.embeddings.create(model=self.model_name, input=["维度探测"])
        self.dim = len(probe.data[0].embedding)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        batch = 10                                   # DashScope 单次最多 10 条
        for i in range(0, len(texts), batch):
            resp = self.client.embeddings.create(
                model=self.model_name,
                input=[t[:2000] for t in texts[i:i + batch]],
                dimensions=self.dim,
            )
            vectors.extend([d.embedding for d in resp.data])
        return vectors


@lru_cache(maxsize=1)
def get_embedder() -> BaseEmbedder:
    if config.EMBEDDING_PROVIDER.lower() == "dashscope":
        return DashScopeEmbedder()
    return HashEmbedder()
