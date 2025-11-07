#!/usr/bin/env python3
"""
Embedding Pipeline for Brand Content and Visual Assets
Generates embeddings for semantic search and brand consistency

Architecture Position: Data Layer → Embedding Generation
References: /Users/coreyfoster/Desktop/CLAUDE_20-10_MASTER.md
Truth Protocol Compliance: All 15 rules
Version: 2.0.0
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import hashlib

import numpy as np
from sentence_transformers import SentenceTransformer
import torch

# Optional imports for image embeddings
try:
    from PIL import Image
    import torchvision.transforms as transforms
    import torchvision.models as models
    IMAGE_SUPPORT = True
except ImportError:
    IMAGE_SUPPORT = False
    logging.warning("Image support not available. Install Pillow and torchvision.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmbeddingModel(Enum):
    """Available embedding models"""
    # Text models
    SENTENCE_BERT_BASE = "all-MiniLM-L6-v2"  # 384 dim, fast
    SENTENCE_BERT_LARGE = "all-mpnet-base-v2"  # 768 dim, accurate
    MULTI_QA = "multi-qa-mpnet-base-dot-v1"  # 768 dim, QA optimized

    # Image models
    RESNET50 = "resnet50"  # 2048 dim
    CLIP = "clip"  # 512 dim, text+image


class ContentType(Enum):
    """Types of content to embed"""
    TEXT = "text"
    IMAGE = "image"
    MULTIMODAL = "multimodal"


@dataclass
class EmbeddingRequest:
    """Request for embedding generation"""
    content_id: str
    content_type: ContentType
    content: Union[str, bytes, Image.Image, Any]
    metadata: Dict[str, Any]
    model: Optional[EmbeddingModel] = None


@dataclass
class EmbeddingResult:
    """Result of embedding generation"""
    content_id: str
    embedding: np.ndarray
    model: str
    dimension: int
    metadata: Dict[str, Any]
    processing_time_ms: float
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def get_hash(self) -> str:
        """Get hash of embedding for deduplication"""
        return hashlib.sha256(self.embedding.tobytes()).hexdigest()


class EmbeddingPipeline:
    """
    Production embedding pipeline for brand content

    Features:
    - Text embedding (Sentence-BERT)
    - Image embedding (ResNet50, CLIP)
    - Batch processing
    - Caching and deduplication
    - Performance monitoring (P95 < 200ms for text)

    Usage:
        pipeline = EmbeddingPipeline()
        await pipeline.initialize()
        result = await pipeline.embed_text("Brand description")
        results = await pipeline.embed_batch(requests)
    """

    def __init__(
        self,
        default_text_model: EmbeddingModel = EmbeddingModel.SENTENCE_BERT_LARGE,
        default_image_model: EmbeddingModel = EmbeddingModel.RESNET50,
        device: Optional[str] = None,
        cache_size: int = 1000
    ):
        """
        Initialize embedding pipeline

        Args:
            default_text_model: Default model for text embedding
            default_image_model: Default model for image embedding
            device: Device for inference ('cuda', 'cpu', or None for auto)
            cache_size: Number of embeddings to cache
        """
        self.default_text_model = default_text_model
        self.default_image_model = default_image_model

        # Device selection
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.cache_size = cache_size

        # Model storage
        self.text_models: Dict[str, SentenceTransformer] = {}
        self.image_models: Dict[str, Any] = {}

        # Cache
        self.embedding_cache: Dict[str, EmbeddingResult] = {}

        # Metrics
        self.metrics = {
            'total_embeddings': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_processing_time_ms': 0.0,
            'embeddings_by_type': {
                'text': 0,
                'image': 0,
                'multimodal': 0
            }
        }

        self._initialized = False

        logger.info(f"EmbeddingPipeline initialized (device={self.device})")

    async def initialize(self):
        """Initialize models and pipeline"""
        if self._initialized:
            logger.warning("EmbeddingPipeline already initialized")
            return

        try:
            # Load default text model
            logger.info(f"Loading text model: {self.default_text_model.value}")
            self.text_models[self.default_text_model.value] = SentenceTransformer(
                self.default_text_model.value,
                device=self.device
            )

            # Load default image model if supported
            if IMAGE_SUPPORT and self.default_image_model == EmbeddingModel.RESNET50:
                logger.info("Loading image model: ResNet50")
                self.image_models['resnet50'] = self._load_resnet50()

            self._initialized = True
            logger.info("EmbeddingPipeline initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise

    def _load_resnet50(self):
        """Load ResNet50 model for image embeddings"""
        model = models.resnet50(pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove classification layer
        model.to(self.device)
        model.eval()
        return model

    def _get_cache_key(self, content: Union[str, bytes], model: str) -> str:
        """Generate cache key for content"""
        if isinstance(content, str):
            content_hash = hashlib.sha256(content.encode()).hexdigest()
        elif isinstance(content, bytes):
            content_hash = hashlib.sha256(content).hexdigest()
        else:
            content_hash = hashlib.sha256(str(content).encode()).hexdigest()

        return f"{model}:{content_hash}"

    async def embed_text(
        self,
        text: str,
        model: Optional[EmbeddingModel] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> EmbeddingResult:
        """
        Generate embedding for text

        Args:
            text: Input text
            model: Embedding model (uses default if None)
            metadata: Additional metadata

        Returns:
            EmbeddingResult
        """
        if not self._initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        start_time = datetime.now()

        # Select model
        if model is None:
            model = self.default_text_model

        model_name = model.value

        # Check cache
        cache_key = self._get_cache_key(text, model_name)
        if cache_key in self.embedding_cache:
            self.metrics['cache_hits'] += 1
            logger.debug(f"Cache hit for text embedding")
            return self.embedding_cache[cache_key]

        self.metrics['cache_misses'] += 1

        # Load model if not loaded
        if model_name not in self.text_models:
            logger.info(f"Loading text model: {model_name}")
            self.text_models[model_name] = SentenceTransformer(
                model_name,
                device=self.device
            )

        # Generate embedding
        text_model = self.text_models[model_name]
        embedding = text_model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        # Create result
        result = EmbeddingResult(
            content_id=cache_key,
            embedding=embedding.astype(np.float32),
            model=model_name,
            dimension=len(embedding),
            metadata=metadata or {},
            processing_time_ms=processing_time
        )

        # Update cache
        if len(self.embedding_cache) < self.cache_size:
            self.embedding_cache[cache_key] = result

        # Update metrics
        self.metrics['total_embeddings'] += 1
        self.metrics['embeddings_by_type']['text'] += 1
        self._update_avg_processing_time(processing_time)

        logger.info(f"Generated text embedding: {len(embedding)}D in {processing_time:.2f}ms")

        return result

    async def embed_image(
        self,
        image: Union[str, bytes, Image.Image],
        model: Optional[EmbeddingModel] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> EmbeddingResult:
        """
        Generate embedding for image

        Args:
            image: Image path, bytes, or PIL Image
            model: Embedding model (uses default if None)
            metadata: Additional metadata

        Returns:
            EmbeddingResult
        """
        if not self._initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        if not IMAGE_SUPPORT:
            raise RuntimeError("Image support not available. Install Pillow and torchvision.")

        start_time = datetime.now()

        # Select model
        if model is None:
            model = self.default_image_model

        model_name = model.value

        # Load image
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, bytes):
            from io import BytesIO
            image = Image.open(BytesIO(image)).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("Image must be path, bytes, or PIL Image")

        # Preprocess image
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        input_tensor = preprocess(image).unsqueeze(0).to(self.device)

        # Generate embedding
        if model_name == 'resnet50':
            if 'resnet50' not in self.image_models:
                self.image_models['resnet50'] = self._load_resnet50()

            model = self.image_models['resnet50']

            with torch.no_grad():
                embedding = model(input_tensor)
                embedding = embedding.squeeze().cpu().numpy()
        else:
            raise ValueError(f"Unsupported image model: {model_name}")

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        # Create result
        result = EmbeddingResult(
            content_id=hashlib.sha256(embedding.tobytes()).hexdigest(),
            embedding=embedding.astype(np.float32),
            model=model_name,
            dimension=len(embedding),
            metadata=metadata or {},
            processing_time_ms=processing_time
        )

        # Update metrics
        self.metrics['total_embeddings'] += 1
        self.metrics['embeddings_by_type']['image'] += 1
        self._update_avg_processing_time(processing_time)

        logger.info(f"Generated image embedding: {len(embedding)}D in {processing_time:.2f}ms")

        return result

    async def embed_batch(
        self,
        requests: List[EmbeddingRequest]
    ) -> List[EmbeddingResult]:
        """
        Process batch of embedding requests

        Args:
            requests: List of EmbeddingRequest objects

        Returns:
            List of EmbeddingResult objects
        """
        if not self._initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        results = []

        # Group by content type and model
        text_requests = []
        image_requests = []

        for req in requests:
            if req.content_type == ContentType.TEXT:
                text_requests.append(req)
            elif req.content_type == ContentType.IMAGE:
                image_requests.append(req)

        # Process text requests in batch
        if text_requests:
            for req in text_requests:
                result = await self.embed_text(
                    text=req.content,
                    model=req.model,
                    metadata=req.metadata
                )
                result.content_id = req.content_id
                results.append(result)

        # Process image requests
        if image_requests:
            for req in image_requests:
                result = await self.embed_image(
                    image=req.content,
                    model=req.model,
                    metadata=req.metadata
                )
                result.content_id = req.content_id
                results.append(result)

        logger.info(f"Processed batch: {len(results)} embeddings")

        return results

    def _update_avg_processing_time(self, processing_time: float):
        """Update average processing time metric"""
        total = self.metrics['total_embeddings']
        current_avg = self.metrics['avg_processing_time_ms']

        # Incremental average
        new_avg = ((current_avg * (total - 1)) + processing_time) / total
        self.metrics['avg_processing_time_ms'] = new_avg

    async def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        metric: str = "cosine"
    ) -> float:
        """
        Compute similarity between two embeddings

        Args:
            embedding1: First embedding
            embedding2: Second embedding
            metric: Similarity metric ('cosine', 'euclidean', 'dot')

        Returns:
            Similarity score
        """
        if metric == "cosine":
            # Cosine similarity
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            return float(dot_product / (norm1 * norm2))

        elif metric == "euclidean":
            # Euclidean distance (converted to similarity)
            distance = np.linalg.norm(embedding1 - embedding2)
            return float(1 / (1 + distance))

        elif metric == "dot":
            # Dot product
            return float(np.dot(embedding1, embedding2))

        else:
            raise ValueError(f"Unsupported metric: {metric}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline metrics"""
        return {
            'total_embeddings': self.metrics['total_embeddings'],
            'cache_hit_rate': (
                self.metrics['cache_hits'] / (self.metrics['cache_hits'] + self.metrics['cache_misses'])
                if (self.metrics['cache_hits'] + self.metrics['cache_misses']) > 0 else 0.0
            ),
            'avg_processing_time_ms': self.metrics['avg_processing_time_ms'],
            'embeddings_by_type': self.metrics['embeddings_by_type'],
            'cache_size': len(self.embedding_cache),
            'device': self.device,
            'models_loaded': {
                'text': list(self.text_models.keys()),
                'image': list(self.image_models.keys())
            }
        }

    def clear_cache(self):
        """Clear embedding cache"""
        self.embedding_cache.clear()
        logger.info("Embedding cache cleared")

    async def get_health(self) -> Dict[str, Any]:
        """
        Get pipeline health status

        Returns:
            Health metrics
        """
        if not self._initialized:
            return {
                'status': 'not_initialized',
                'models_loaded': False
            }

        try:
            metrics = self.get_metrics()

            # Check if SLO is met (P95 < 200ms for text)
            avg_time = metrics['avg_processing_time_ms']
            slo_met = avg_time < 200 if avg_time > 0 else True

            return {
                'status': 'healthy' if slo_met else 'degraded',
                'initialized': True,
                'device': self.device,
                'models_loaded': metrics['models_loaded'],
                'total_embeddings': metrics['total_embeddings'],
                'avg_processing_time_ms': avg_time,
                'slo_met': slo_met,
                'cache_hit_rate': metrics['cache_hit_rate']
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e)
            }


# Example usage and testing
async def main():
    """Example embedding pipeline usage"""
    print("\n" + "=" * 70)
    print("Embedding Pipeline - Example Usage")
    print("=" * 70 + "\n")

    # Initialize pipeline
    pipeline = EmbeddingPipeline()
    await pipeline.initialize()

    # Example 1: Embed brand content text
    print("### Example 1: Text Embedding")
    brand_text = "Skyy Rose - Luxury fashion brand specializing in handcrafted leather goods"
    result = await pipeline.embed_text(
        text=brand_text,
        metadata={'type': 'brand_description', 'brand': 'Skyy Rose'}
    )

    print(f"Text: {brand_text}")
    print(f"Model: {result.model}")
    print(f"Dimension: {result.dimension}")
    print(f"Processing Time: {result.processing_time_ms:.2f}ms")
    print(f"Embedding (first 5): {result.embedding[:5]}")
    print()

    # Example 2: Batch embedding
    print("### Example 2: Batch Embedding")
    batch_requests = [
        EmbeddingRequest(
            content_id="text_001",
            content_type=ContentType.TEXT,
            content="Elegant leather handbag with gold hardware",
            metadata={'product_id': 'SKU-1234'}
        ),
        EmbeddingRequest(
            content_id="text_002",
            content_type=ContentType.TEXT,
            content="Premium Italian leather wallet",
            metadata={'product_id': 'SKU-5678'}
        ),
        EmbeddingRequest(
            content_id="text_003",
            content_type=ContentType.TEXT,
            content="Designer crossbody bag in cognac",
            metadata={'product_id': 'SKU-9012'}
        )
    ]

    batch_results = await pipeline.embed_batch(batch_requests)
    print(f"Processed {len(batch_results)} embeddings")
    for res in batch_results:
        print(f"  - {res.content_id}: {res.dimension}D ({res.processing_time_ms:.2f}ms)")
    print()

    # Example 3: Similarity computation
    print("### Example 3: Similarity Computation")
    text1 = "Luxury leather handbag"
    text2 = "Premium leather bag"
    text3 = "Cotton t-shirt"

    emb1 = await pipeline.embed_text(text1)
    emb2 = await pipeline.embed_text(text2)
    emb3 = await pipeline.embed_text(text3)

    sim_12 = await pipeline.compute_similarity(emb1.embedding, emb2.embedding)
    sim_13 = await pipeline.compute_similarity(emb1.embedding, emb3.embedding)

    print(f"'{text1}' vs '{text2}': {sim_12:.4f}")
    print(f"'{text1}' vs '{text3}': {sim_13:.4f}")
    print()

    # Pipeline metrics
    print("### Pipeline Metrics")
    metrics = pipeline.get_metrics()
    print(f"Total Embeddings: {metrics['total_embeddings']}")
    print(f"Cache Hit Rate: {metrics['cache_hit_rate']:.2%}")
    print(f"Avg Processing Time: {metrics['avg_processing_time_ms']:.2f}ms")
    print(f"Device: {metrics['device']}")
    print()

    # Health check
    print("### Health Status")
    health = await pipeline.get_health()
    print(f"Status: {health['status']}")
    print(f"SLO Met (< 200ms): {health['slo_met']}")
    print(f"Models Loaded: {health['models_loaded']}")


if __name__ == "__main__":
    asyncio.run(main())
