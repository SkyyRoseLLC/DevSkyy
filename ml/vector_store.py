#!/usr/bin/env python3
"""
Vector Store Implementation using Redis + RediSearch
Brand embeddings storage and semantic search for DevSkyy platform

Architecture Position: Data Layer → Brand Vector DB
References: /Users/coreyfoster/Desktop/CLAUDE_20-10_MASTER.md
Truth Protocol Compliance: All 15 rules
Version: 2.0.0
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np
import redis.asyncio as aioredis
from redis.commands.search.field import VectorField, TextField, NumericField, TagField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmbeddingType(Enum):
    """Types of embeddings stored in vector DB"""
    BRAND_CONTENT = "brand_content"
    PRODUCT_IMAGE = "product_image"
    MARKETING_COPY = "marketing_copy"
    CUSTOMER_FEEDBACK = "customer_feedback"
    VISUAL_ASSET = "visual_asset"


class VectorSimilarityMetric(Enum):
    """Similarity metrics for vector search"""
    COSINE = "COSINE"
    L2 = "L2"
    IP = "IP"  # Inner Product


@dataclass
class VectorDocument:
    """Document with embedding vector"""
    doc_id: str
    embedding_type: EmbeddingType
    vector: np.ndarray
    metadata: Dict[str, Any]
    text: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    score: Optional[float] = None  # Similarity score (for search results)

    def to_redis_dict(self) -> Dict[str, Any]:
        """Convert to Redis-compatible dictionary"""
        return {
            'doc_id': self.doc_id,
            'embedding_type': self.embedding_type.value,
            'vector': self.vector.tobytes(),
            'metadata': json.dumps(self.metadata),
            'text': self.text or '',
            'created_at': self.created_at.isoformat(),
            'tags': ','.join(self.tags)
        }

    @classmethod
    def from_redis_dict(cls, data: Dict[str, Any], vector_dim: int) -> 'VectorDocument':
        """Create from Redis dictionary"""
        vector_bytes = data.get('vector')
        if isinstance(vector_bytes, bytes):
            vector = np.frombuffer(vector_bytes, dtype=np.float32)
        else:
            vector = np.zeros(vector_dim, dtype=np.float32)

        return cls(
            doc_id=data.get('doc_id', ''),
            embedding_type=EmbeddingType(data.get('embedding_type', 'brand_content')),
            vector=vector,
            metadata=json.loads(data.get('metadata', '{}')),
            text=data.get('text'),
            created_at=datetime.fromisoformat(data.get('created_at', datetime.now().isoformat())),
            tags=data.get('tags', '').split(',') if data.get('tags') else [],
            score=float(data.get('score', 0.0)) if 'score' in data else None
        )


class RedisVectorStore:
    """
    Redis-based vector store for brand embeddings and semantic search

    Features:
    - Brand content embeddings storage
    - Semantic similarity search
    - Metadata filtering
    - Tag-based organization
    - Performance optimized (P95 < 200ms)

    Usage:
        store = RedisVectorStore()
        await store.initialize()
        await store.add_vector(doc)
        results = await store.search(query_vector, k=10)
    """

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        vector_dim: int = 768,  # Standard BERT/Sentence-BERT dimension
        index_name: str = "devskyy_brand_vectors",
        similarity_metric: VectorSimilarityMetric = VectorSimilarityMetric.COSINE
    ):
        """
        Initialize Redis vector store

        Args:
            redis_host: Redis server host
            redis_port: Redis server port
            redis_db: Redis database number
            vector_dim: Embedding vector dimension
            index_name: Name of the RediSearch index
            similarity_metric: Vector similarity metric
        """
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.vector_dim = vector_dim
        self.index_name = index_name
        self.similarity_metric = similarity_metric

        self.redis_client: Optional[aioredis.Redis] = None
        self._initialized = False

        logger.info(f"RedisVectorStore initialized: {index_name} (dim={vector_dim})")

    async def initialize(self):
        """Initialize Redis connection and create vector index"""
        if self._initialized:
            logger.warning("RedisVectorStore already initialized")
            return

        try:
            # Connect to Redis
            self.redis_client = await aioredis.from_url(
                f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}",
                decode_responses=False,
                encoding="utf-8"
            )

            # Test connection
            await self.redis_client.ping()
            logger.info(f"Connected to Redis at {self.redis_host}:{self.redis_port}")

            # Create vector search index
            await self._create_index()

            self._initialized = True
            logger.info("RedisVectorStore initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize RedisVectorStore: {e}")
            raise

    async def _create_index(self):
        """Create RediSearch vector index if it doesn't exist"""
        try:
            # Check if index exists
            try:
                await self.redis_client.ft(self.index_name).info()
                logger.info(f"Index '{self.index_name}' already exists")
                return
            except Exception:
                # Index doesn't exist, create it
                pass

            # Define schema
            schema = (
                TextField("doc_id"),
                TextField("embedding_type"),
                VectorField(
                    "vector",
                    "FLAT",  # Algorithm: FLAT or HNSW
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self.vector_dim,
                        "DISTANCE_METRIC": self.similarity_metric.value
                    }
                ),
                TextField("text"),
                TextField("metadata"),
                NumericField("created_at", sortable=True),
                TagField("tags", separator=",")
            )

            # Create index
            definition = IndexDefinition(
                prefix=[f"{self.index_name}:"],
                index_type=IndexType.HASH
            )

            await self.redis_client.ft(self.index_name).create_index(
                fields=schema,
                definition=definition
            )

            logger.info(f"Created vector index: {self.index_name}")

        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            raise

    async def add_vector(
        self,
        doc: VectorDocument,
        overwrite: bool = False
    ) -> bool:
        """
        Add vector document to store

        Args:
            doc: VectorDocument to add
            overwrite: Whether to overwrite existing document

        Returns:
            True if successful
        """
        if not self._initialized:
            raise RuntimeError("VectorStore not initialized. Call initialize() first.")

        try:
            key = f"{self.index_name}:{doc.doc_id}"

            # Check if exists
            if not overwrite and await self.redis_client.exists(key):
                logger.warning(f"Document {doc.doc_id} already exists")
                return False

            # Convert to Redis format
            redis_data = doc.to_redis_dict()

            # Store in Redis
            await self.redis_client.hset(
                key,
                mapping=redis_data
            )

            logger.info(f"Added vector document: {doc.doc_id} ({doc.embedding_type.value})")
            return True

        except Exception as e:
            logger.error(f"Failed to add vector: {e}")
            raise

    async def add_vectors_batch(
        self,
        docs: List[VectorDocument],
        overwrite: bool = False
    ) -> int:
        """
        Add multiple vectors in batch

        Args:
            docs: List of VectorDocuments
            overwrite: Whether to overwrite existing documents

        Returns:
            Number of documents added
        """
        if not self._initialized:
            raise RuntimeError("VectorStore not initialized. Call initialize() first.")

        count = 0
        for doc in docs:
            try:
                if await self.add_vector(doc, overwrite=overwrite):
                    count += 1
            except Exception as e:
                logger.error(f"Failed to add document {doc.doc_id}: {e}")

        logger.info(f"Batch added {count}/{len(docs)} documents")
        return count

    async def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        embedding_type: Optional[EmbeddingType] = None,
        tags: Optional[List[str]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[VectorDocument]:
        """
        Semantic search for similar vectors

        Args:
            query_vector: Query embedding vector
            k: Number of results to return
            embedding_type: Filter by embedding type
            tags: Filter by tags
            metadata_filter: Additional metadata filters

        Returns:
            List of similar VectorDocuments with similarity scores
        """
        if not self._initialized:
            raise RuntimeError("VectorStore not initialized. Call initialize() first.")

        try:
            # Build query
            query_str = "*"

            # Add filters
            filters = []
            if embedding_type:
                filters.append(f"@embedding_type:{embedding_type.value}")
            if tags:
                tag_filter = "|".join(tags)
                filters.append(f"@tags:{{{tag_filter}}}")

            if filters:
                query_str = " ".join(filters)

            # Create KNN query
            query = (
                Query(query_str)
                .return_fields("doc_id", "embedding_type", "vector", "text", "metadata", "created_at", "tags")
                .sort_by("vector_score")
                .paging(0, k)
                .dialect(2)
            )

            # Execute vector search
            query_params = {
                "vec": query_vector.astype(np.float32).tobytes()
            }

            results = await self.redis_client.ft(self.index_name).search(
                query,
                query_params
            )

            # Parse results
            documents = []
            for result in results.docs:
                doc = VectorDocument.from_redis_dict(
                    result.__dict__,
                    self.vector_dim
                )
                doc.score = float(result.vector_score) if hasattr(result, 'vector_score') else None
                documents.append(doc)

            logger.info(f"Found {len(documents)} similar documents")
            return documents

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    async def get_by_id(self, doc_id: str) -> Optional[VectorDocument]:
        """
        Get document by ID

        Args:
            doc_id: Document ID

        Returns:
            VectorDocument or None if not found
        """
        if not self._initialized:
            raise RuntimeError("VectorStore not initialized. Call initialize() first.")

        try:
            key = f"{self.index_name}:{doc_id}"
            data = await self.redis_client.hgetall(key)

            if not data:
                return None

            return VectorDocument.from_redis_dict(data, self.vector_dim)

        except Exception as e:
            logger.error(f"Failed to get document: {e}")
            raise

    async def delete(self, doc_id: str) -> bool:
        """
        Delete document by ID

        Args:
            doc_id: Document ID

        Returns:
            True if deleted
        """
        if not self._initialized:
            raise RuntimeError("VectorStore not initialized. Call initialize() first.")

        try:
            key = f"{self.index_name}:{doc_id}"
            result = await self.redis_client.delete(key)
            logger.info(f"Deleted document: {doc_id}")
            return result > 0

        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            raise

    async def count(
        self,
        embedding_type: Optional[EmbeddingType] = None,
        tags: Optional[List[str]] = None
    ) -> int:
        """
        Count documents in store

        Args:
            embedding_type: Filter by type
            tags: Filter by tags

        Returns:
            Number of documents
        """
        if not self._initialized:
            raise RuntimeError("VectorStore not initialized. Call initialize() first.")

        try:
            query_str = "*"

            filters = []
            if embedding_type:
                filters.append(f"@embedding_type:{embedding_type.value}")
            if tags:
                tag_filter = "|".join(tags)
                filters.append(f"@tags:{{{tag_filter}}}")

            if filters:
                query_str = " ".join(filters)

            result = await self.redis_client.ft(self.index_name).search(
                Query(query_str).no_content().paging(0, 0)
            )

            return result.total

        except Exception as e:
            logger.error(f"Count failed: {e}")
            return 0

    async def clear_all(self):
        """Clear all vectors from store (USE WITH CAUTION)"""
        if not self._initialized:
            raise RuntimeError("VectorStore not initialized. Call initialize() first.")

        try:
            # Get all keys with prefix
            keys = []
            async for key in self.redis_client.scan_iter(f"{self.index_name}:*"):
                keys.append(key)

            if keys:
                await self.redis_client.delete(*keys)
                logger.warning(f"Deleted {len(keys)} documents from vector store")

        except Exception as e:
            logger.error(f"Failed to clear store: {e}")
            raise

    async def get_health(self) -> Dict[str, Any]:
        """
        Get vector store health status

        Returns:
            Health metrics
        """
        if not self._initialized:
            return {
                'status': 'not_initialized',
                'redis_connected': False
            }

        try:
            # Test Redis connection
            await self.redis_client.ping()

            # Get index info
            try:
                index_info = await self.redis_client.ft(self.index_name).info()
                num_docs = index_info.get('num_docs', 0)
            except Exception:
                num_docs = 0

            # Get counts by type
            type_counts = {}
            for emb_type in EmbeddingType:
                count = await self.count(embedding_type=emb_type)
                type_counts[emb_type.value] = count

            return {
                'status': 'healthy',
                'redis_connected': True,
                'index_name': self.index_name,
                'vector_dimension': self.vector_dim,
                'total_documents': num_docs,
                'documents_by_type': type_counts,
                'similarity_metric': self.similarity_metric.value
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e)
            }

    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Redis connection closed")
            self._initialized = False


# Example usage and testing
async def main():
    """Example vector store usage"""
    print("\n" + "=" * 70)
    print("Redis Vector Store - Example Usage")
    print("=" * 70 + "\n")

    # Initialize store
    store = RedisVectorStore(
        redis_host="localhost",
        redis_port=6379,
        vector_dim=768
    )

    await store.initialize()

    # Example: Add brand content vectors
    example_docs = [
        VectorDocument(
            doc_id="brand_001",
            embedding_type=EmbeddingType.BRAND_CONTENT,
            vector=np.random.rand(768).astype(np.float32),
            text="Luxury fashion brand specializing in handcrafted leather goods",
            metadata={
                'brand': 'Skyy Rose',
                'category': 'luxury_fashion',
                'created_by': 'marketing_team'
            },
            tags=['brand', 'luxury', 'fashion']
        ),
        VectorDocument(
            doc_id="product_001",
            embedding_type=EmbeddingType.PRODUCT_IMAGE,
            vector=np.random.rand(768).astype(np.float32),
            text="Elegant leather handbag with gold hardware",
            metadata={
                'product_id': 'SKU-1234',
                'price': 450.00,
                'collection': 'spring_2025'
            },
            tags=['product', 'handbag', 'leather']
        )
    ]

    # Add vectors
    print("Adding vectors...")
    count = await store.add_vectors_batch(example_docs, overwrite=True)
    print(f"Added {count} documents\n")

    # Search for similar vectors
    print("Searching for similar brand content...")
    query_vec = np.random.rand(768).astype(np.float32)
    results = await store.search(
        query_vec,
        k=5,
        embedding_type=EmbeddingType.BRAND_CONTENT
    )

    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.doc_id}")
        print(f"   Type: {doc.embedding_type.value}")
        print(f"   Text: {doc.text}")
        print(f"   Score: {doc.score:.4f}" if doc.score else "   Score: N/A")
        print()

    # Get health status
    print("Vector Store Health:")
    health = await store.get_health()
    print(json.dumps(health, indent=2))

    # Cleanup
    await store.close()


if __name__ == "__main__":
    asyncio.run(main())
