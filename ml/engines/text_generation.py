#!/usr/bin/env python3
"""
Text Generation Engine - Production Implementation
Multi-model AI text generation with brand voice consistency

Architecture Position: ML Layer → Text Generation Engine
References: /Users/coreyfoster/Desktop/CLAUDE_20-10_MASTER.md
Truth Protocol Compliance: All 15 rules
Version: 2.0.0
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from collections import deque

import numpy as np
from anthropic import Anthropic, AsyncAnthropic, HUMAN_PROMPT, AI_PROMPT
from openai import AsyncOpenAI
import tiktoken

# Import local modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_store import RedisVectorStore, VectorDocument, EmbeddingType
from embedding_pipeline import EmbeddingPipeline, ContentType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AIModel(Enum):
    """Supported AI models for text generation"""
    CLAUDE_SONNET_4_5 = "claude-sonnet-4-5-20250929"
    CLAUDE_SONNET_3_5 = "claude-3-5-sonnet-20241022"
    CLAUDE_OPUS_3 = "claude-3-opus-20240229"
    GPT_4_TURBO = "gpt-4-turbo-preview"
    GPT_4 = "gpt-4"
    GPT_4_O = "gpt-4o"
    GPT_3_5_TURBO = "gpt-3.5-turbo"


class OutputFormat(Enum):
    """Output format options"""
    PLAIN = "plain"
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"


class FinishReason(Enum):
    """Reasons for generation completion"""
    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class TextGenerationRequest:
    """Request for text generation"""
    prompt: str
    system_prompt: Optional[str] = None
    max_tokens: int = 1000
    temperature: float = 0.7
    top_p: float = 1.0
    model: AIModel = AIModel.CLAUDE_SONNET_4_5
    brand_voice_check: bool = True
    output_format: OutputFormat = OutputFormat.PLAIN
    context: Optional[List[Dict[str, str]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: bool = False
    request_id: Optional[str] = None

    def __post_init__(self):
        """Validate request parameters"""
        if self.temperature < 0.0 or self.temperature > 1.0:
            raise ValueError("Temperature must be between 0.0 and 1.0")
        if self.top_p < 0.0 or self.top_p > 1.0:
            raise ValueError("top_p must be between 0.0 and 1.0")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if self.request_id is None:
            self.request_id = self._generate_request_id()

    def _generate_request_id(self) -> str:
        """Generate unique request ID"""
        content = f"{self.prompt}{self.model.value}{time.time()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class TextGenerationResult:
    """Result of text generation"""
    generated_text: str
    model_used: str
    tokens_used: Dict[str, int]  # input, output, total
    generation_time_ms: float
    brand_voice_score: float
    finish_reason: FinishReason
    metadata: Dict[str, Any] = field(default_factory=dict)
    request_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'generated_text': self.generated_text,
            'model_used': self.model_used,
            'tokens_used': self.tokens_used,
            'generation_time_ms': self.generation_time_ms,
            'brand_voice_score': self.brand_voice_score,
            'finish_reason': self.finish_reason.value,
            'metadata': self.metadata,
            'request_id': self.request_id,
            'timestamp': self.timestamp.isoformat()
        }


class ModelSelector:
    """
    Intelligent model selection based on task type

    Selection criteria:
    - Claude Sonnet 4.5: Long-form content, reasoning, analysis, complex tasks
    - Claude Opus: Highest quality, creative writing, critical tasks
    - GPT-4 Turbo: Creative writing, short copy, conversational, faster responses
    - GPT-3.5 Turbo: Simple tasks, high-volume, cost-sensitive
    """

    TASK_KEYWORDS = {
        AIModel.CLAUDE_SONNET_4_5: [
            'analyze', 'reasoning', 'complex', 'detailed', 'comprehensive',
            'technical', 'research', 'explanation', 'comparison', 'strategy'
        ],
        AIModel.CLAUDE_OPUS_3: [
            'creative', 'critical', 'important', 'high-quality', 'nuanced',
            'sophisticated', 'advanced', 'expert', 'professional'
        ],
        AIModel.GPT_4_TURBO: [
            'write', 'create', 'generate', 'copy', 'conversation',
            'chat', 'quick', 'short', 'social', 'marketing'
        ],
        AIModel.GPT_3_5_TURBO: [
            'simple', 'basic', 'quick', 'summary', 'bullet',
            'list', 'short', 'brief', 'fast'
        ]
    }

    @classmethod
    def select_model(
        cls,
        prompt: str,
        preferred_model: Optional[AIModel] = None,
        max_tokens: int = 1000
    ) -> AIModel:
        """
        Select optimal model based on prompt and requirements

        Args:
            prompt: User prompt
            preferred_model: Preferred model (if specified)
            max_tokens: Maximum tokens needed

        Returns:
            Selected AIModel
        """
        if preferred_model:
            return preferred_model

        # Analyze prompt for keywords
        prompt_lower = prompt.lower()
        scores = {model: 0 for model in AIModel}

        for model, keywords in cls.TASK_KEYWORDS.items():
            for keyword in keywords:
                if keyword in prompt_lower:
                    scores[model] += 1

        # Select model with highest score
        if max(scores.values()) > 0:
            selected = max(scores.items(), key=lambda x: x[1])[0]
            logger.info(f"Model selected based on keywords: {selected.value}")
            return selected

        # Default to Claude Sonnet 4.5 for balanced performance
        logger.info("No specific keywords found, using default: Claude Sonnet 4.5")
        return AIModel.CLAUDE_SONNET_4_5


class RateLimiter:
    """Token bucket rate limiter for API calls"""

    def __init__(self, calls_per_minute: int = 50):
        """
        Initialize rate limiter

        Args:
            calls_per_minute: Maximum calls allowed per minute
        """
        self.calls_per_minute = calls_per_minute
        self.interval = 60.0 / calls_per_minute
        self.last_call_times: deque = deque(maxlen=calls_per_minute)

    async def acquire(self):
        """Acquire rate limit token (blocks if necessary)"""
        now = time.time()

        # Remove old timestamps
        while self.last_call_times and self.last_call_times[0] < now - 60:
            self.last_call_times.popleft()

        # Check if we need to wait
        if len(self.last_call_times) >= self.calls_per_minute:
            sleep_time = 60 - (now - self.last_call_times[0])
            if sleep_time > 0:
                logger.warning(f"Rate limit reached, sleeping {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)

        self.last_call_times.append(time.time())


class TextGenerationCache:
    """LRU cache for text generation results"""

    def __init__(self, max_size: int = 1000):
        """
        Initialize cache

        Args:
            max_size: Maximum number of cached results
        """
        self.max_size = max_size
        self.cache: Dict[str, TextGenerationResult] = {}
        self.access_order: deque = deque()

    def _get_cache_key(self, request: TextGenerationRequest) -> str:
        """Generate cache key from request"""
        key_data = {
            'prompt': request.prompt,
            'system_prompt': request.system_prompt,
            'model': request.model.value,
            'temperature': request.temperature,
            'max_tokens': request.max_tokens,
            'top_p': request.top_p
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def get(self, request: TextGenerationRequest) -> Optional[TextGenerationResult]:
        """Get cached result"""
        key = self._get_cache_key(request)
        if key in self.cache:
            # Update access order
            self.access_order.remove(key)
            self.access_order.append(key)
            logger.debug(f"Cache hit for request {request.request_id}")
            return self.cache[key]
        return None

    def set(self, request: TextGenerationRequest, result: TextGenerationResult):
        """Cache result"""
        key = self._get_cache_key(request)

        # Evict oldest if at capacity
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest_key = self.access_order.popleft()
            del self.cache[oldest_key]
            logger.debug(f"Evicted cache entry: {oldest_key}")

        self.cache[key] = result
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
        logger.debug(f"Cached result for request {request.request_id}")

    def clear(self):
        """Clear all cached results"""
        self.cache.clear()
        self.access_order.clear()
        logger.info("Cache cleared")


class TextGenerationEngine:
    """
    Production-grade text generation engine with multi-model support

    Features:
    - Multiple AI models (Claude, GPT-4)
    - Brand voice consistency checking
    - Intelligent model selection
    - Response caching
    - Rate limiting
    - Token budget management
    - Retry logic with exponential backoff
    - Performance monitoring (P95 < 2s)

    Usage:
        engine = TextGenerationEngine(
            anthropic_api_key="...",
            openai_api_key="..."
        )
        await engine.initialize()

        request = TextGenerationRequest(
            prompt="Write a product description",
            model=AIModel.CLAUDE_SONNET_4_5,
            brand_voice_check=True
        )

        result = await engine.generate(request)
        print(result.generated_text)
    """

    def __init__(
        self,
        anthropic_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        enable_cache: bool = True,
        cache_size: int = 1000,
        calls_per_minute: int = 50,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize text generation engine

        Args:
            anthropic_api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            openai_api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            redis_host: Redis host for vector store
            redis_port: Redis port
            enable_cache: Enable response caching
            cache_size: Maximum cached responses
            calls_per_minute: API rate limit
            max_retries: Maximum retry attempts
            retry_delay: Base retry delay in seconds
        """
        # API clients
        self.anthropic_api_key = anthropic_api_key or os.getenv('ANTHROPIC_API_KEY')
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')

        if not self.anthropic_api_key and not self.openai_api_key:
            raise ValueError("At least one API key must be provided (Anthropic or OpenAI)")

        self.anthropic_client: Optional[AsyncAnthropic] = None
        self.openai_client: Optional[AsyncOpenAI] = None

        # Vector store for brand voice
        self.vector_store = RedisVectorStore(
            redis_host=redis_host,
            redis_port=redis_port,
            vector_dim=768
        )

        # Embedding pipeline for brand voice checking
        self.embedding_pipeline = EmbeddingPipeline()

        # Caching and rate limiting
        self.enable_cache = enable_cache
        self.cache = TextGenerationCache(max_size=cache_size) if enable_cache else None
        self.rate_limiter = RateLimiter(calls_per_minute=calls_per_minute)

        # Retry configuration
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Token encoding
        self.token_encoders: Dict[str, Any] = {}

        # Metrics
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'cache_hits': 0,
            'total_tokens_used': 0,
            'total_generation_time_ms': 0.0,
            'requests_by_model': {},
            'errors_by_type': {},
            'brand_voice_scores': []
        }

        self._initialized = False

        logger.info("TextGenerationEngine initialized")

    async def initialize(self):
        """Initialize all components"""
        if self._initialized:
            logger.warning("TextGenerationEngine already initialized")
            return

        try:
            # Initialize API clients
            if self.anthropic_api_key:
                self.anthropic_client = AsyncAnthropic(api_key=self.anthropic_api_key)
                logger.info("Anthropic client initialized")

            if self.openai_api_key:
                self.openai_client = AsyncOpenAI(api_key=self.openai_api_key)
                logger.info("OpenAI client initialized")

            # Initialize token encoders
            try:
                self.token_encoders['gpt-4'] = tiktoken.encoding_for_model("gpt-4")
                self.token_encoders['gpt-3.5-turbo'] = tiktoken.encoding_for_model("gpt-3.5-turbo")
                logger.info("Token encoders loaded")
            except Exception as e:
                logger.warning(f"Failed to load token encoders: {e}")

            # Initialize vector store
            await self.vector_store.initialize()
            logger.info("Vector store initialized")

            # Initialize embedding pipeline
            await self.embedding_pipeline.initialize()
            logger.info("Embedding pipeline initialized")

            self._initialized = True
            logger.info("TextGenerationEngine fully initialized")

        except Exception as e:
            logger.error(f"Failed to initialize TextGenerationEngine: {e}")
            raise

    async def generate(
        self,
        request: TextGenerationRequest
    ) -> TextGenerationResult:
        """
        Generate text using specified AI model

        Args:
            request: TextGenerationRequest object

        Returns:
            TextGenerationResult with generated text and metadata
        """
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        start_time = time.time()
        self.metrics['total_requests'] += 1

        # Check cache
        if self.enable_cache and self.cache:
            cached_result = self.cache.get(request)
            if cached_result:
                self.metrics['cache_hits'] += 1
                logger.info(f"Returning cached result for request {request.request_id}")
                return cached_result

        # Rate limiting
        await self.rate_limiter.acquire()

        # Auto-select model if needed
        if not request.model:
            request.model = ModelSelector.select_model(
                prompt=request.prompt,
                max_tokens=request.max_tokens
            )

        # Generate text with retries
        result = await self._generate_with_retry(request)

        # Brand voice check if enabled
        if request.brand_voice_check:
            brand_score = await self._check_brand_voice(result.generated_text)
            result.brand_voice_score = brand_score
            self.metrics['brand_voice_scores'].append(brand_score)
        else:
            result.brand_voice_score = 1.0

        # Calculate total generation time
        result.generation_time_ms = (time.time() - start_time) * 1000

        # Update metrics
        self._update_metrics(result)

        # Cache result
        if self.enable_cache and self.cache:
            self.cache.set(request, result)

        logger.info(
            f"Generated text: {len(result.generated_text)} chars, "
            f"{result.tokens_used['total']} tokens, "
            f"{result.generation_time_ms:.2f}ms, "
            f"brand_score={result.brand_voice_score:.3f}"
        )

        return result

    async def _generate_with_retry(
        self,
        request: TextGenerationRequest
    ) -> TextGenerationResult:
        """Generate text with exponential backoff retry"""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                if request.model.value.startswith('claude'):
                    return await self._generate_claude(request)
                elif request.model.value.startswith('gpt'):
                    return await self._generate_openai(request)
                else:
                    raise ValueError(f"Unsupported model: {request.model.value}")

            except Exception as e:
                last_error = e
                error_type = type(e).__name__
                self.metrics['errors_by_type'][error_type] = \
                    self.metrics['errors_by_type'].get(error_type, 0) + 1

                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {self.max_retries} attempts failed: {last_error}")

        # All retries failed
        self.metrics['failed_requests'] += 1
        return TextGenerationResult(
            generated_text="",
            model_used=request.model.value,
            tokens_used={'input': 0, 'output': 0, 'total': 0},
            generation_time_ms=0.0,
            brand_voice_score=0.0,
            finish_reason=FinishReason.ERROR,
            metadata={'error': str(last_error)},
            request_id=request.request_id
        )

    async def _generate_claude(
        self,
        request: TextGenerationRequest
    ) -> TextGenerationResult:
        """Generate text using Claude API"""
        if not self.anthropic_client:
            raise RuntimeError("Anthropic client not initialized")

        # Build messages
        messages = []

        # Add context if provided
        if request.context:
            for msg in request.context:
                messages.append({
                    'role': msg['role'],
                    'content': msg['content']
                })

        # Add current prompt
        messages.append({
            'role': 'user',
            'content': request.prompt
        })

        # Call Claude API
        start_time = time.time()

        response = await self.anthropic_client.messages.create(
            model=request.model.value,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            system=request.system_prompt or "",
            messages=messages,
            stop_sequences=request.stop_sequences or []
        )

        generation_time_ms = (time.time() - start_time) * 1000

        # Extract generated text
        generated_text = ""
        for block in response.content:
            if hasattr(block, 'text'):
                generated_text += block.text

        # Apply output formatting
        if request.output_format != OutputFormat.PLAIN:
            generated_text = self._format_output(generated_text, request.output_format)

        # Map finish reason
        finish_reason_map = {
            'end_turn': FinishReason.STOP,
            'max_tokens': FinishReason.LENGTH,
            'stop_sequence': FinishReason.STOP
        }
        finish_reason = finish_reason_map.get(response.stop_reason, FinishReason.STOP)

        # Token usage
        tokens_used = {
            'input': response.usage.input_tokens,
            'output': response.usage.output_tokens,
            'total': response.usage.input_tokens + response.usage.output_tokens
        }

        return TextGenerationResult(
            generated_text=generated_text,
            model_used=request.model.value,
            tokens_used=tokens_used,
            generation_time_ms=generation_time_ms,
            brand_voice_score=0.0,  # Will be calculated later
            finish_reason=finish_reason,
            metadata={
                'stop_reason': response.stop_reason,
                'response_id': response.id
            },
            request_id=request.request_id
        )

    async def _generate_openai(
        self,
        request: TextGenerationRequest
    ) -> TextGenerationResult:
        """Generate text using OpenAI API"""
        if not self.openai_client:
            raise RuntimeError("OpenAI client not initialized")

        # Build messages
        messages = []

        # Add system prompt
        if request.system_prompt:
            messages.append({
                'role': 'system',
                'content': request.system_prompt
            })

        # Add context if provided
        if request.context:
            for msg in request.context:
                messages.append({
                    'role': msg['role'],
                    'content': msg['content']
                })

        # Add current prompt
        messages.append({
            'role': 'user',
            'content': request.prompt
        })

        # Call OpenAI API
        start_time = time.time()

        response = await self.openai_client.chat.completions.create(
            model=request.model.value,
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop_sequences
        )

        generation_time_ms = (time.time() - start_time) * 1000

        # Extract generated text
        generated_text = response.choices[0].message.content or ""

        # Apply output formatting
        if request.output_format != OutputFormat.PLAIN:
            generated_text = self._format_output(generated_text, request.output_format)

        # Map finish reason
        finish_reason_map = {
            'stop': FinishReason.STOP,
            'length': FinishReason.LENGTH,
            'content_filter': FinishReason.CONTENT_FILTER
        }
        finish_reason = finish_reason_map.get(
            response.choices[0].finish_reason,
            FinishReason.STOP
        )

        # Token usage
        usage = response.usage
        tokens_used = {
            'input': usage.prompt_tokens,
            'output': usage.completion_tokens,
            'total': usage.total_tokens
        }

        return TextGenerationResult(
            generated_text=generated_text,
            model_used=request.model.value,
            tokens_used=tokens_used,
            generation_time_ms=generation_time_ms,
            brand_voice_score=0.0,  # Will be calculated later
            finish_reason=finish_reason,
            metadata={
                'finish_reason': response.choices[0].finish_reason,
                'response_id': response.id
            },
            request_id=request.request_id
        )

    def _format_output(self, text: str, output_format: OutputFormat) -> str:
        """Format output text according to specified format"""
        if output_format == OutputFormat.PLAIN:
            return text
        elif output_format == OutputFormat.MARKDOWN:
            # Text is already in markdown format from most models
            return text
        elif output_format == OutputFormat.HTML:
            # Simple markdown to HTML conversion
            # In production, use a proper markdown library
            html_text = text.replace('\n\n', '</p><p>')
            html_text = html_text.replace('\n', '<br>')
            return f'<p>{html_text}</p>'
        elif output_format == OutputFormat.JSON:
            try:
                # Try to parse as JSON
                return json.dumps(json.loads(text), indent=2)
            except:
                # Return as JSON string
                return json.dumps({'text': text}, indent=2)
        else:
            return text

    async def _check_brand_voice(self, generated_text: str) -> float:
        """
        Check brand voice consistency using vector similarity

        Args:
            generated_text: Generated text to check

        Returns:
            Brand voice score (0.0 - 1.0), 0.80+ indicates good match
        """
        try:
            # Generate embedding for output text
            result = await self.embedding_pipeline.embed_text(
                text=generated_text,
                metadata={'type': 'generation_output'}
            )
            output_embedding = result.embedding

            # Search for similar brand voice examples
            similar_docs = await self.vector_store.search(
                query_vector=output_embedding,
                k=5,
                embedding_type=EmbeddingType.BRAND_CONTENT
            )

            if not similar_docs:
                logger.warning("No brand voice examples found in vector store")
                return 0.5  # Neutral score

            # Calculate average similarity score
            scores = [doc.score for doc in similar_docs if doc.score is not None]
            if not scores:
                return 0.5

            # Convert distance to similarity (assuming cosine distance)
            # Lower distance = higher similarity
            avg_score = sum(scores) / len(scores)

            # Normalize to 0-1 range
            brand_voice_score = max(0.0, min(1.0, 1.0 - avg_score))

            logger.info(f"Brand voice score: {brand_voice_score:.3f}")
            return brand_voice_score

        except Exception as e:
            logger.error(f"Brand voice check failed: {e}")
            return 0.5  # Return neutral score on error

    def _update_metrics(self, result: TextGenerationResult):
        """Update performance metrics"""
        self.metrics['successful_requests'] += 1
        self.metrics['total_tokens_used'] += result.tokens_used['total']
        self.metrics['total_generation_time_ms'] += result.generation_time_ms

        # Update model-specific metrics
        model = result.model_used
        if model not in self.metrics['requests_by_model']:
            self.metrics['requests_by_model'][model] = {
                'count': 0,
                'total_tokens': 0,
                'avg_time_ms': 0.0
            }

        model_metrics = self.metrics['requests_by_model'][model]
        model_metrics['count'] += 1
        model_metrics['total_tokens'] += result.tokens_used['total']

        # Update average time
        prev_avg = model_metrics['avg_time_ms']
        count = model_metrics['count']
        new_avg = ((prev_avg * (count - 1)) + result.generation_time_ms) / count
        model_metrics['avg_time_ms'] = new_avg

    async def generate_batch(
        self,
        requests: List[TextGenerationRequest]
    ) -> List[TextGenerationResult]:
        """
        Generate text for multiple requests in parallel

        Args:
            requests: List of TextGenerationRequest objects

        Returns:
            List of TextGenerationResult objects
        """
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        logger.info(f"Processing batch of {len(requests)} requests")

        # Generate all requests concurrently
        tasks = [self.generate(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch request {i} failed: {result}")
                final_results.append(TextGenerationResult(
                    generated_text="",
                    model_used=requests[i].model.value,
                    tokens_used={'input': 0, 'output': 0, 'total': 0},
                    generation_time_ms=0.0,
                    brand_voice_score=0.0,
                    finish_reason=FinishReason.ERROR,
                    metadata={'error': str(result)},
                    request_id=requests[i].request_id
                ))
            else:
                final_results.append(result)

        logger.info(f"Batch processing complete: {len(final_results)} results")
        return final_results

    def count_tokens(self, text: str, model: str = "gpt-4") -> int:
        """
        Count tokens in text for a given model

        Args:
            text: Input text
            model: Model name

        Returns:
            Token count
        """
        # For Claude, approximate (no official tokenizer available)
        if 'claude' in model.lower():
            # Rough approximation: 1 token ≈ 4 characters
            return len(text) // 4

        # For GPT models, use tiktoken
        if model in self.token_encoders:
            encoder = self.token_encoders[model]
            return len(encoder.encode(text))

        # Fallback approximation
        return len(text.split())

    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        total_requests = self.metrics['total_requests']
        successful = self.metrics['successful_requests']

        avg_generation_time = 0.0
        if successful > 0:
            avg_generation_time = self.metrics['total_generation_time_ms'] / successful

        # Calculate P95 from recent requests (simplified)
        p95_time = avg_generation_time * 1.5  # Approximation

        avg_brand_score = 0.0
        if self.metrics['brand_voice_scores']:
            avg_brand_score = sum(self.metrics['brand_voice_scores']) / len(self.metrics['brand_voice_scores'])

        return {
            'total_requests': total_requests,
            'successful_requests': successful,
            'failed_requests': self.metrics['failed_requests'],
            'success_rate': successful / total_requests if total_requests > 0 else 0.0,
            'cache_hits': self.metrics['cache_hits'],
            'cache_hit_rate': self.metrics['cache_hits'] / total_requests if total_requests > 0 else 0.0,
            'total_tokens_used': self.metrics['total_tokens_used'],
            'avg_tokens_per_request': self.metrics['total_tokens_used'] / successful if successful > 0 else 0.0,
            'avg_generation_time_ms': avg_generation_time,
            'p95_generation_time_ms': p95_time,
            'avg_brand_voice_score': avg_brand_score,
            'requests_by_model': self.metrics['requests_by_model'],
            'errors_by_type': self.metrics['errors_by_type']
        }

    async def get_health(self) -> Dict[str, Any]:
        """
        Get health status of text generation engine

        Returns:
            Health metrics and status
        """
        if not self._initialized:
            return {
                'status': 'not_initialized',
                'anthropic_available': False,
                'openai_available': False
            }

        try:
            # Test API connectivity
            anthropic_ok = self.anthropic_client is not None
            openai_ok = self.openai_client is not None

            # Get vector store health
            vector_health = await self.vector_store.get_health()

            # Get embedding pipeline health
            embedding_health = await self.embedding_pipeline.get_health()

            # Get current metrics
            metrics = self.get_metrics()

            # Check if SLO is met (P95 < 2000ms)
            p95_time = metrics['p95_generation_time_ms']
            slo_met = p95_time < 2000 if p95_time > 0 else True

            return {
                'status': 'healthy' if (anthropic_ok or openai_ok) and slo_met else 'degraded',
                'initialized': True,
                'anthropic_available': anthropic_ok,
                'openai_available': openai_ok,
                'vector_store_status': vector_health.get('status'),
                'embedding_pipeline_status': embedding_health.get('status'),
                'total_requests': metrics['total_requests'],
                'success_rate': metrics['success_rate'],
                'avg_generation_time_ms': metrics['avg_generation_time_ms'],
                'p95_generation_time_ms': metrics['p95_generation_time_ms'],
                'slo_met': slo_met,
                'avg_brand_voice_score': metrics['avg_brand_voice_score'],
                'cache_hit_rate': metrics['cache_hit_rate']
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e)
            }

    def clear_cache(self):
        """Clear the response cache"""
        if self.cache:
            self.cache.clear()
            logger.info("Cache cleared")

    async def close(self):
        """Close all connections and cleanup"""
        if self.vector_store:
            await self.vector_store.close()
        logger.info("TextGenerationEngine closed")


# Example usage and testing
async def main():
    """Example text generation usage"""
    print("\n" + "=" * 80)
    print("Text Generation Engine - Production Example")
    print("=" * 80 + "\n")

    # Initialize engine
    # Note: Set ANTHROPIC_API_KEY and OPENAI_API_KEY environment variables
    try:
        engine = TextGenerationEngine(
            enable_cache=True,
            cache_size=100,
            calls_per_minute=50
        )
        await engine.initialize()
        print("Engine initialized successfully\n")
    except Exception as e:
        print(f"Failed to initialize engine: {e}")
        print("Make sure ANTHROPIC_API_KEY or OPENAI_API_KEY is set")
        return

    # Example 1: Simple text generation (Claude)
    print("### Example 1: Product Description Generation (Claude)")
    try:
        request1 = TextGenerationRequest(
            prompt="Write a compelling product description for a luxury leather handbag "
                   "from Skyy Rose brand. Emphasize craftsmanship and elegance.",
            system_prompt="You are a professional marketing copywriter specializing in luxury fashion.",
            max_tokens=300,
            temperature=0.8,
            model=AIModel.CLAUDE_SONNET_4_5,
            brand_voice_check=True
        )

        result1 = await engine.generate(request1)
        print(f"Model: {result1.model_used}")
        print(f"Generated Text:\n{result1.generated_text}\n")
        print(f"Tokens Used: {result1.tokens_used['total']}")
        print(f"Generation Time: {result1.generation_time_ms:.2f}ms")
        print(f"Brand Voice Score: {result1.brand_voice_score:.3f}")
        print(f"Finish Reason: {result1.finish_reason.value}")
        print()
    except Exception as e:
        print(f"Example 1 failed: {e}\n")

    # Example 2: Creative writing (GPT-4)
    print("### Example 2: Social Media Post (GPT-4)")
    try:
        request2 = TextGenerationRequest(
            prompt="Create an Instagram caption for a photo of our new spring collection. "
                   "Keep it engaging and use 2-3 relevant hashtags.",
            max_tokens=150,
            temperature=0.9,
            model=AIModel.GPT_4_TURBO,
            brand_voice_check=True,
            output_format=OutputFormat.PLAIN
        )

        result2 = await engine.generate(request2)
        print(f"Model: {result2.model_used}")
        print(f"Generated Text:\n{result2.generated_text}\n")
        print(f"Tokens Used: {result2.tokens_used['total']}")
        print(f"Generation Time: {result2.generation_time_ms:.2f}ms")
        print(f"Brand Voice Score: {result2.brand_voice_score:.3f}")
        print()
    except Exception as e:
        print(f"Example 2 failed: {e}\n")

    # Example 3: Batch generation
    print("### Example 3: Batch Generation")
    try:
        batch_requests = [
            TextGenerationRequest(
                prompt=f"Write a short tagline for {product}",
                max_tokens=50,
                temperature=0.7,
                model=AIModel.CLAUDE_SONNET_4_5,
                brand_voice_check=False
            )
            for product in ["leather wallet", "designer handbag", "crossbody bag"]
        ]

        batch_results = await engine.generate_batch(batch_requests)
        print(f"Generated {len(batch_results)} taglines:")
        for i, result in enumerate(batch_results, 1):
            print(f"{i}. {result.generated_text.strip()} "
                  f"({result.tokens_used['total']} tokens, {result.generation_time_ms:.0f}ms)")
        print()
    except Exception as e:
        print(f"Example 3 failed: {e}\n")

    # Example 4: Conversation context
    print("### Example 4: Conversation with Context")
    try:
        request4 = TextGenerationRequest(
            prompt="What color options do we have?",
            system_prompt="You are a helpful sales assistant for Skyy Rose luxury fashion brand.",
            max_tokens=200,
            temperature=0.7,
            model=AIModel.GPT_4_TURBO,
            context=[
                {
                    'role': 'user',
                    'content': 'Tell me about your leather handbags'
                },
                {
                    'role': 'assistant',
                    'content': 'Our luxury leather handbags are handcrafted from premium Italian leather. '
                               'They feature gold hardware and come in several styles including totes, '
                               'crossbody bags, and clutches.'
                }
            ]
        )

        result4 = await engine.generate(request4)
        print(f"Model: {result4.model_used}")
        print(f"Generated Text:\n{result4.generated_text}\n")
        print()
    except Exception as e:
        print(f"Example 4 failed: {e}\n")

    # Performance metrics
    print("### Performance Metrics")
    metrics = engine.get_metrics()
    print(f"Total Requests: {metrics['total_requests']}")
    print(f"Success Rate: {metrics['success_rate']:.2%}")
    print(f"Cache Hit Rate: {metrics['cache_hit_rate']:.2%}")
    print(f"Total Tokens Used: {metrics['total_tokens_used']}")
    print(f"Avg Generation Time: {metrics['avg_generation_time_ms']:.2f}ms")
    print(f"P95 Generation Time: {metrics['p95_generation_time_ms']:.2f}ms")
    print(f"Avg Brand Voice Score: {metrics['avg_brand_voice_score']:.3f}")
    print()

    # Requests by model
    print("Requests by Model:")
    for model, stats in metrics['requests_by_model'].items():
        print(f"  {model}: {stats['count']} requests, "
              f"{stats['avg_time_ms']:.2f}ms avg, "
              f"{stats['total_tokens']} tokens")
    print()

    # Health check
    print("### Health Status")
    health = await engine.get_health()
    print(f"Status: {health['status']}")
    print(f"Anthropic Available: {health['anthropic_available']}")
    print(f"OpenAI Available: {health['openai_available']}")
    print(f"Vector Store: {health['vector_store_status']}")
    print(f"Embedding Pipeline: {health['embedding_pipeline_status']}")
    print(f"SLO Met (P95 < 2s): {health['slo_met']}")
    print()

    # Cleanup
    await engine.close()
    print("Engine closed")


if __name__ == "__main__":
    asyncio.run(main())
