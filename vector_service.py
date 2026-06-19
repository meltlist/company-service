"""向量检索服务：Embedding + Qdrant"""
import uuid
from typing import Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
from sentence_transformers import SentenceTransformer

from config import settings


class EmbeddingService:
    """Embedding 服务"""

    def __init__(self):
        self.model: Optional[SentenceTransformer] = None
        self.dimension: int = settings.EMBEDDING_DIM

    def load_model(self):
        """加载模型（延迟加载）"""
        if self.model is None:
            self.model = SentenceTransformer(
                settings.EMBEDDING_MODEL,
                device=settings.EMBEDDING_DEVICE,
            )
            self.dimension = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: str | list[str]) -> np.ndarray:
        """文本向量化"""
        self.load_model()
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings

    def get_dimension(self) -> int:
        """获取向量维度"""
        self.load_model()
        return self.dimension


class VectorStore:
    """Qdrant 向量存储"""

    def __init__(self):
        self.client: Optional[QdrantClient] = None
        self.collection_name: str = settings.QDRANT_COLLECTION

    def connect(self):
        """连接 Qdrant"""
        if self.client is None:
            if settings.QDRANT_IN_MEMORY:
                self.client = QdrantClient(":memory:")
            else:
                self.client = QdrantClient(
                    host=settings.QDRANT_HOST,
                    port=settings.QDRANT_PORT,
                )
        self._ensure_collection()

    def _ensure_collection(self):
        """确保 Collection 存在"""
        try:
            self.client.get_collection(self.collection_name)
        except (UnexpectedResponse, Exception):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=settings.EMBEDDING_DIM,
                    distance=models.Distance.COSINE,
                ),
            )

    def insert(
        self,
        doc_id: str,
        chunks: list[dict],
        embedding_service: EmbeddingService,
    ) -> list[str]:
        """插入文档块及其向量"""
        self.connect()

        texts = [chunk["content"] for chunk in chunks]
        vectors = embedding_service.encode(texts)

        vector_ids = []
        points = []

        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            vector_id = f"{doc_id}_{i}"
            vector_ids.append(vector_id)

            point = models.PointStruct(
                id=vector_id,
                vector=vector.tolist(),
                payload={
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "content": chunk["content"],
                    "chunk_type": chunk.get("chunk_type", "text"),
                    "metadata": chunk.get("metadata", {}),
                },
            )
            points.append(point)

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

        return vector_ids

    def search(
        self,
        query: str,
        embedding_service: EmbeddingService,
        doc_ids: list[str] = None,
        top_k: int = 5,
        score_threshold: float = 0.5,
    ) -> list[dict]:
        """向量检索"""
        self.connect()

        query_vector = embedding_service.encode(query)

        filter_conditions = None
        if doc_ids:
            filter_conditions = models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_id",
                        match=models.MatchAny(any=doc_ids),
                    )
                ]
            )

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            query_filter=filter_conditions,
            limit=top_k,
            score_threshold=score_threshold,
        )

        return [
            {
                "id": hit.id,
                "score": hit.score,
                "content": hit.payload["content"],
                "doc_id": hit.payload["doc_id"],
                "chunk_index": hit.payload.get("chunk_index", 0),
                "metadata": hit.payload.get("metadata", {}),
            }
            for hit in results
        ]

    def delete_by_doc_id(self, doc_id: str):
        """删除文档的所有向量"""
        self.connect()
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_id",
                            match=models.MatchValue(value=doc_id),
                        )
                    ]
                )
            ),
        )

    def delete_collection(self):
        """删除整个 Collection"""
        self.connect()
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass


class HybridSearch:
    """混合检索（向量 + 关键词）"""

    def __init__(self, vector_store: VectorStore, embedding_service: EmbeddingService):
        self.vector_store = vector_store
        self.embedding = embedding_service

    def search(
        self,
        query: str,
        doc_ids: list[str] = None,
        top_k: int = 5,
        alpha: float = 0.7,  # 向量权重
    ) -> list[dict]:
        """混合检索"""
        # 向量检索
        vector_results = self.vector_store.search(
            query=query,
            embedding_service=self.embedding,
            doc_ids=doc_ids,
            top_k=top_k * 2,  # 多取一些用于融合
            score_threshold=0.3,
        )

        # 关键词检索（简单 BM25）
        keyword_results = self._bm25_search(query, vector_results, top_k)

        # 结果融合
        fused = self._reciprocal_rank_fusion(vector_results, keyword_results, alpha, top_k)

        return fused[:top_k]

    def _bm25_search(
        self,
        query: str,
        vector_results: list[dict],
        top_k: int,
    ) -> list[dict]:
        """简化 BM25（基于已有向量结果重排序）"""
        # 对于简化实现，使用向量结果中的文本匹配度作为关键词分数
        query_terms = set(query.lower().split())
        results = []

        for item in vector_results:
            content_lower = item["content"].lower()
            matches = sum(1 for term in query_terms if term in content_lower)
            keyword_score = matches / max(len(query_terms), 1)
            if keyword_score > 0:
                results.append({
                    **item,
                    "keyword_score": keyword_score,
                })

        # 按关键词分数排序
        results.sort(key=lambda x: x.get("keyword_score", 0), reverse=True)
        return results[:top_k]

    def _reciprocal_rank_fusion(
        self,
        vector_results: list[dict],
        keyword_results: list[dict],
        alpha: float,
        top_k: int,
    ) -> list[dict]:
        """RRF 融合"""
        scores = {}

        for rank, item in enumerate(vector_results):
            key = item["id"]
            vector_score = item["score"]
            scores[key] = scores.get(key, 0) + alpha * vector_score * (1 / (rank + 60))

        for rank, item in enumerate(keyword_results):
            key = item["id"]
            keyword_score = item.get("keyword_score", 0)
            scores[key] = scores.get(key, 0) + (1 - alpha) * keyword_score * (1 / (rank + 60))

        # 合并结果
        all_results = {item["id"]: item for item in vector_results}
        for item in keyword_results:
            if item["id"] not in all_results:
                all_results[item["id"]] = item

        # 按融合分数排序
        ranked = sorted(
            [(k, scores[k]) for k in scores],
            key=lambda x: x[1],
            reverse=True,
        )

        return [all_results[k] for k, _ in ranked[:top_k]]


class QueryRewriter:
    """查询改写，提高召回率"""

    def rewrite(self, query: str) -> list[str]:
        """生成多个查询变体"""
        variations = [query]

        # 1. 提取关键实体
        # 简化实现：保留原查询 + 简短版本
        words = query.split()
        if len(words) > 5:
            variations.append(" ".join(words[:5]))
            variations.append(" ".join(words[-5:]))

        # 2. 去除停用词（简化）
        stopwords = {"的", "了", "是", "在", "和", "与", "及", "或", "等", "以及"}
        filtered = " ".join(w for w in words if w not in stopwords)
        if filtered != query:
            variations.append(filtered)

        return list(set(variations))
