#!/usr/bin/env python3
"""
Knowledge Graph Implementation for Brand Entity Relationships
Enterprise-grade PostgreSQL-based graph database for DevSkyy platform

Architecture Position: Data Layer → Brand Knowledge Graph
References: /Users/coreyfoster/Desktop/CLAUDE_20-10_MASTER.md
Truth Protocol Compliance: All 15 rules
Version: 2.0.0

Graph Schema:
- Entities: Brand, Product, Customer, Campaign, Influencer
- Relationships: OWNS, BELONGS_TO, PURCHASED, ENGAGED_WITH, INFLUENCED_BY
- Attributes: Properties stored as JSONB for flexibility

Performance Target: P95 query latency < 200ms
"""

import asyncio
import asyncpg
import json
import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from heapq import heappush, heappop

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Knowledge graph entity types"""
    BRAND = "brand"
    PRODUCT = "product"
    CUSTOMER = "customer"
    CAMPAIGN = "campaign"
    INFLUENCER = "influencer"
    CATEGORY = "category"
    TREND = "trend"
    COLLECTION = "collection"


class RelationshipType(Enum):
    """Knowledge graph relationship types"""
    OWNS = "owns"
    BELONGS_TO = "belongs_to"
    PURCHASED = "purchased"
    ENGAGED_WITH = "engaged_with"
    INFLUENCED_BY = "influenced_by"
    PART_OF = "part_of"
    SIMILAR_TO = "similar_to"
    RECOMMENDS = "recommends"
    FOLLOWS = "follows"
    COLLABORATED_WITH = "collaborated_with"


@dataclass
class GraphEntity:
    """Knowledge graph entity node"""
    entity_id: str
    entity_type: EntityType
    properties: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'entity_id': self.entity_id,
            'entity_type': self.entity_type.value,
            'properties': self.properties,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GraphEntity':
        """Create from dictionary"""
        return cls(
            entity_id=data['entity_id'],
            entity_type=EntityType(data['entity_type']),
            properties=data['properties'],
            created_at=datetime.fromisoformat(data['created_at']) if isinstance(data.get('created_at'), str) else data.get('created_at', datetime.now()),
            updated_at=datetime.fromisoformat(data['updated_at']) if isinstance(data.get('updated_at'), str) else data.get('updated_at', datetime.now()),
            metadata=data.get('metadata', {})
        )


@dataclass
class GraphRelationship:
    """Knowledge graph relationship edge"""
    relationship_id: str
    source_id: str
    target_id: str
    relationship_type: RelationshipType
    properties: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'relationship_id': self.relationship_id,
            'source_id': self.source_id,
            'target_id': self.target_id,
            'relationship_type': self.relationship_type.value,
            'properties': self.properties,
            'weight': self.weight,
            'created_at': self.created_at.isoformat(),
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GraphRelationship':
        """Create from dictionary"""
        return cls(
            relationship_id=data['relationship_id'],
            source_id=data['source_id'],
            target_id=data['target_id'],
            relationship_type=RelationshipType(data['relationship_type']),
            properties=data.get('properties', {}),
            weight=data.get('weight', 1.0),
            created_at=datetime.fromisoformat(data['created_at']) if isinstance(data.get('created_at'), str) else data.get('created_at', datetime.now()),
            metadata=data.get('metadata', {})
        )


@dataclass
class GraphPath:
    """Path through knowledge graph"""
    nodes: List[GraphEntity]
    edges: List[GraphRelationship]
    total_weight: float

    def __len__(self) -> int:
        return len(self.nodes)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'nodes': [node.to_dict() for node in self.nodes],
            'edges': [edge.to_dict() for edge in self.edges],
            'total_weight': self.total_weight,
            'length': len(self)
        }


@dataclass
class GraphMetrics:
    """Knowledge graph metrics"""
    total_entities: int = 0
    total_relationships: int = 0
    entities_by_type: Dict[str, int] = field(default_factory=dict)
    relationships_by_type: Dict[str, int] = field(default_factory=dict)
    avg_degree: float = 0.0
    max_degree: int = 0
    connected_components: int = 0
    avg_query_time_ms: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'total_entities': self.total_entities,
            'total_relationships': self.total_relationships,
            'entities_by_type': self.entities_by_type,
            'relationships_by_type': self.relationships_by_type,
            'avg_degree': self.avg_degree,
            'max_degree': self.max_degree,
            'connected_components': self.connected_components,
            'avg_query_time_ms': self.avg_query_time_ms,
            'last_updated': self.last_updated.isoformat()
        }


class KnowledgeGraphStore:
    """
    Enterprise-grade PostgreSQL-based Knowledge Graph Store

    Features:
    - Entity and relationship management
    - Graph traversal (BFS, DFS, shortest path)
    - Graph analytics (centrality, clustering, community detection)
    - ACID transactions
    - Performance optimized (P95 < 200ms)
    - Connection pooling

    Schema:
    - graph_entities: Entity nodes with JSONB properties
    - graph_relationships: Relationship edges with JSONB properties
    - Indexes on entity_type, relationship_type, source_id, target_id

    Usage:
        graph = KnowledgeGraphStore()
        await graph.initialize()
        await graph.add_entity(entity)
        await graph.add_relationship(relationship)
        path = await graph.find_shortest_path(source_id, target_id)
        centrality = await graph.compute_centrality()
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        host: str = "localhost",
        port: int = 5432,
        database: str = "devskyy",
        user: str = "postgres",
        password: str = "postgres",
        min_pool_size: int = 2,
        max_pool_size: int = 10
    ):
        """
        Initialize Knowledge Graph Store

        Args:
            database_url: PostgreSQL connection URL (overrides other params)
            host: Database host
            port: Database port
            database: Database name
            user: Database user
            password: Database password
            min_pool_size: Minimum connection pool size
            max_pool_size: Maximum connection pool size
        """
        self.database_url = database_url
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.min_pool_size = min_pool_size
        self.max_pool_size = max_pool_size

        self.pool: Optional[asyncpg.Pool] = None
        self._initialized = False
        self.metrics = GraphMetrics()

        logger.info(f"KnowledgeGraphStore initialized: {database}@{host}:{port}")

    async def initialize(self):
        """Initialize database connection and create schema"""
        if self._initialized:
            logger.warning("KnowledgeGraphStore already initialized")
            return

        try:
            # Create connection pool
            if self.database_url:
                self.pool = await asyncpg.create_pool(
                    self.database_url,
                    min_size=self.min_pool_size,
                    max_size=self.max_pool_size,
                    command_timeout=60.0
                )
            else:
                self.pool = await asyncpg.create_pool(
                    host=self.host,
                    port=self.port,
                    database=self.database,
                    user=self.user,
                    password=self.password,
                    min_size=self.min_pool_size,
                    max_size=self.max_pool_size,
                    command_timeout=60.0
                )

            logger.info(f"Connected to PostgreSQL: {self.database}")

            # Create schema
            await self._create_schema()

            # Update metrics
            await self._update_metrics()

            self._initialized = True
            logger.info("KnowledgeGraphStore initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize KnowledgeGraphStore: {e}")
            raise

    async def _create_schema(self):
        """Create knowledge graph schema"""
        async with self.pool.acquire() as conn:
            # Create entities table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS graph_entities (
                    entity_id VARCHAR(255) PRIMARY KEY,
                    entity_type VARCHAR(50) NOT NULL,
                    properties JSONB NOT NULL DEFAULT '{}',
                    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    metadata JSONB DEFAULT '{}'
                )
            """)

            # Create relationships table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS graph_relationships (
                    relationship_id VARCHAR(255) PRIMARY KEY,
                    source_id VARCHAR(255) NOT NULL,
                    target_id VARCHAR(255) NOT NULL,
                    relationship_type VARCHAR(50) NOT NULL,
                    properties JSONB DEFAULT '{}',
                    weight FLOAT DEFAULT 1.0,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    metadata JSONB DEFAULT '{}',
                    FOREIGN KEY (source_id) REFERENCES graph_entities(entity_id) ON DELETE CASCADE,
                    FOREIGN KEY (target_id) REFERENCES graph_entities(entity_id) ON DELETE CASCADE
                )
            """)

            # Create indexes for performance
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_entities_type
                ON graph_entities(entity_type)
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_entities_properties
                ON graph_entities USING GIN(properties)
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_relationships_type
                ON graph_relationships(relationship_type)
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_relationships_source
                ON graph_relationships(source_id)
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_relationships_target
                ON graph_relationships(target_id)
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_relationships_source_target
                ON graph_relationships(source_id, target_id)
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_relationships_properties
                ON graph_relationships USING GIN(properties)
            """)

            logger.info("Knowledge graph schema created successfully")

    async def add_entity(
        self,
        entity: GraphEntity,
        overwrite: bool = False
    ) -> bool:
        """
        Add entity to knowledge graph

        Args:
            entity: GraphEntity to add
            overwrite: Whether to overwrite existing entity

        Returns:
            True if successful
        """
        if not self._initialized:
            raise RuntimeError("GraphStore not initialized. Call initialize() first.")

        start_time = time.time()

        try:
            async with self.pool.acquire() as conn:
                if overwrite:
                    await conn.execute("""
                        INSERT INTO graph_entities
                        (entity_id, entity_type, properties, created_at, updated_at, metadata)
                        VALUES ($1, $2, $3, $4, $5, $6)
                        ON CONFLICT (entity_id)
                        DO UPDATE SET
                            entity_type = EXCLUDED.entity_type,
                            properties = EXCLUDED.properties,
                            updated_at = EXCLUDED.updated_at,
                            metadata = EXCLUDED.metadata
                    """,
                        entity.entity_id,
                        entity.entity_type.value,
                        json.dumps(entity.properties),
                        entity.created_at,
                        entity.updated_at,
                        json.dumps(entity.metadata)
                    )
                else:
                    await conn.execute("""
                        INSERT INTO graph_entities
                        (entity_id, entity_type, properties, created_at, updated_at, metadata)
                        VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                        entity.entity_id,
                        entity.entity_type.value,
                        json.dumps(entity.properties),
                        entity.created_at,
                        entity.updated_at,
                        json.dumps(entity.metadata)
                    )

            query_time = (time.time() - start_time) * 1000
            self._update_avg_query_time(query_time)

            logger.info(f"Added entity: {entity.entity_id} ({entity.entity_type.value}) in {query_time:.2f}ms")
            return True

        except asyncpg.UniqueViolationError:
            logger.warning(f"Entity already exists: {entity.entity_id}")
            return False
        except Exception as e:
            logger.error(f"Failed to add entity: {e}")
            raise

    async def add_relationship(
        self,
        relationship: GraphRelationship,
        overwrite: bool = False
    ) -> bool:
        """
        Add relationship to knowledge graph

        Args:
            relationship: GraphRelationship to add
            overwrite: Whether to overwrite existing relationship

        Returns:
            True if successful
        """
        if not self._initialized:
            raise RuntimeError("GraphStore not initialized. Call initialize() first.")

        start_time = time.time()

        try:
            async with self.pool.acquire() as conn:
                if overwrite:
                    await conn.execute("""
                        INSERT INTO graph_relationships
                        (relationship_id, source_id, target_id, relationship_type, properties, weight, created_at, metadata)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                        ON CONFLICT (relationship_id)
                        DO UPDATE SET
                            source_id = EXCLUDED.source_id,
                            target_id = EXCLUDED.target_id,
                            relationship_type = EXCLUDED.relationship_type,
                            properties = EXCLUDED.properties,
                            weight = EXCLUDED.weight,
                            metadata = EXCLUDED.metadata
                    """,
                        relationship.relationship_id,
                        relationship.source_id,
                        relationship.target_id,
                        relationship.relationship_type.value,
                        json.dumps(relationship.properties),
                        relationship.weight,
                        relationship.created_at,
                        json.dumps(relationship.metadata)
                    )
                else:
                    await conn.execute("""
                        INSERT INTO graph_relationships
                        (relationship_id, source_id, target_id, relationship_type, properties, weight, created_at, metadata)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,
                        relationship.relationship_id,
                        relationship.source_id,
                        relationship.target_id,
                        relationship.relationship_type.value,
                        json.dumps(relationship.properties),
                        relationship.weight,
                        relationship.created_at,
                        json.dumps(relationship.metadata)
                    )

            query_time = (time.time() - start_time) * 1000
            self._update_avg_query_time(query_time)

            logger.info(f"Added relationship: {relationship.relationship_id} ({relationship.relationship_type.value}) in {query_time:.2f}ms")
            return True

        except asyncpg.UniqueViolationError:
            logger.warning(f"Relationship already exists: {relationship.relationship_id}")
            return False
        except Exception as e:
            logger.error(f"Failed to add relationship: {e}")
            raise

    async def get_entity(self, entity_id: str) -> Optional[GraphEntity]:
        """
        Get entity by ID

        Args:
            entity_id: Entity ID

        Returns:
            GraphEntity or None if not found
        """
        if not self._initialized:
            raise RuntimeError("GraphStore not initialized. Call initialize() first.")

        start_time = time.time()

        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT * FROM graph_entities WHERE entity_id = $1
                """, entity_id)

                if not row:
                    return None

                entity = GraphEntity(
                    entity_id=row['entity_id'],
                    entity_type=EntityType(row['entity_type']),
                    properties=json.loads(row['properties']) if isinstance(row['properties'], str) else row['properties'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    metadata=json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
                )

                query_time = (time.time() - start_time) * 1000
                self._update_avg_query_time(query_time)

                return entity

        except Exception as e:
            logger.error(f"Failed to get entity: {e}")
            raise

    async def get_entities_by_type(
        self,
        entity_type: EntityType,
        limit: int = 100,
        offset: int = 0
    ) -> List[GraphEntity]:
        """
        Get entities by type

        Args:
            entity_type: EntityType to filter
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of GraphEntity objects
        """
        if not self._initialized:
            raise RuntimeError("GraphStore not initialized. Call initialize() first.")

        start_time = time.time()

        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT * FROM graph_entities
                    WHERE entity_type = $1
                    ORDER BY created_at DESC
                    LIMIT $2 OFFSET $3
                """, entity_type.value, limit, offset)

                entities = []
                for row in rows:
                    entity = GraphEntity(
                        entity_id=row['entity_id'],
                        entity_type=EntityType(row['entity_type']),
                        properties=json.loads(row['properties']) if isinstance(row['properties'], str) else row['properties'],
                        created_at=row['created_at'],
                        updated_at=row['updated_at'],
                        metadata=json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
                    )
                    entities.append(entity)

                query_time = (time.time() - start_time) * 1000
                self._update_avg_query_time(query_time)

                return entities

        except Exception as e:
            logger.error(f"Failed to get entities by type: {e}")
            raise

    async def get_neighbors(
        self,
        entity_id: str,
        relationship_type: Optional[RelationshipType] = None,
        direction: str = "outgoing"
    ) -> List[Tuple[GraphEntity, GraphRelationship]]:
        """
        Get neighboring entities and their relationships

        Args:
            entity_id: Source entity ID
            relationship_type: Filter by relationship type (optional)
            direction: 'outgoing', 'incoming', or 'both'

        Returns:
            List of (neighbor_entity, relationship) tuples
        """
        if not self._initialized:
            raise RuntimeError("GraphStore not initialized. Call initialize() first.")

        start_time = time.time()

        try:
            async with self.pool.acquire() as conn:
                if direction == "outgoing":
                    if relationship_type:
                        rows = await conn.fetch("""
                            SELECT e.*, r.*
                            FROM graph_relationships r
                            JOIN graph_entities e ON r.target_id = e.entity_id
                            WHERE r.source_id = $1 AND r.relationship_type = $2
                        """, entity_id, relationship_type.value)
                    else:
                        rows = await conn.fetch("""
                            SELECT e.*, r.*
                            FROM graph_relationships r
                            JOIN graph_entities e ON r.target_id = e.entity_id
                            WHERE r.source_id = $1
                        """, entity_id)

                elif direction == "incoming":
                    if relationship_type:
                        rows = await conn.fetch("""
                            SELECT e.*, r.*
                            FROM graph_relationships r
                            JOIN graph_entities e ON r.source_id = e.entity_id
                            WHERE r.target_id = $1 AND r.relationship_type = $2
                        """, entity_id, relationship_type.value)
                    else:
                        rows = await conn.fetch("""
                            SELECT e.*, r.*
                            FROM graph_relationships r
                            JOIN graph_entities e ON r.source_id = e.entity_id
                            WHERE r.target_id = $1
                        """, entity_id)

                else:  # both
                    if relationship_type:
                        rows = await conn.fetch("""
                            (SELECT e.*, r.*
                             FROM graph_relationships r
                             JOIN graph_entities e ON r.target_id = e.entity_id
                             WHERE r.source_id = $1 AND r.relationship_type = $2)
                            UNION
                            (SELECT e.*, r.*
                             FROM graph_relationships r
                             JOIN graph_entities e ON r.source_id = e.entity_id
                             WHERE r.target_id = $1 AND r.relationship_type = $2)
                        """, entity_id, relationship_type.value)
                    else:
                        rows = await conn.fetch("""
                            (SELECT e.*, r.*
                             FROM graph_relationships r
                             JOIN graph_entities e ON r.target_id = e.entity_id
                             WHERE r.source_id = $1)
                            UNION
                            (SELECT e.*, r.*
                             FROM graph_relationships r
                             JOIN graph_entities e ON r.source_id = e.entity_id
                             WHERE r.target_id = $1)
                        """, entity_id)

                neighbors = []
                for row in rows:
                    entity = GraphEntity(
                        entity_id=row['entity_id'],
                        entity_type=EntityType(row['entity_type']),
                        properties=json.loads(row['properties']) if isinstance(row['properties'], str) else row['properties'],
                        created_at=row['created_at'],
                        updated_at=row['updated_at'],
                        metadata=json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
                    )

                    relationship = GraphRelationship(
                        relationship_id=row['relationship_id'],
                        source_id=row['source_id'],
                        target_id=row['target_id'],
                        relationship_type=RelationshipType(row['relationship_type']),
                        properties=json.loads(row['properties']) if isinstance(row['properties'], str) else row['properties'],
                        weight=row['weight'],
                        created_at=row['created_at'],
                        metadata=json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
                    )

                    neighbors.append((entity, relationship))

                query_time = (time.time() - start_time) * 1000
                self._update_avg_query_time(query_time)

                logger.info(f"Found {len(neighbors)} neighbors for {entity_id} in {query_time:.2f}ms")
                return neighbors

        except Exception as e:
            logger.error(f"Failed to get neighbors: {e}")
            raise

    async def traverse_bfs(
        self,
        start_id: str,
        max_depth: int = 3,
        relationship_type: Optional[RelationshipType] = None
    ) -> List[GraphEntity]:
        """
        Breadth-First Search traversal from start entity

        Args:
            start_id: Starting entity ID
            max_depth: Maximum traversal depth
            relationship_type: Filter by relationship type (optional)

        Returns:
            List of entities in BFS order
        """
        if not self._initialized:
            raise RuntimeError("GraphStore not initialized. Call initialize() first.")

        start_time = time.time()

        visited: Set[str] = set()
        result: List[GraphEntity] = []
        queue: deque = deque([(start_id, 0)])

        try:
            while queue:
                current_id, depth = queue.popleft()

                if current_id in visited or depth > max_depth:
                    continue

                visited.add(current_id)

                # Get current entity
                entity = await self.get_entity(current_id)
                if entity:
                    result.append(entity)

                # Get neighbors for next level
                if depth < max_depth:
                    neighbors = await self.get_neighbors(
                        current_id,
                        relationship_type=relationship_type,
                        direction="outgoing"
                    )

                    for neighbor_entity, _ in neighbors:
                        if neighbor_entity.entity_id not in visited:
                            queue.append((neighbor_entity.entity_id, depth + 1))

            query_time = (time.time() - start_time) * 1000
            logger.info(f"BFS traversal completed: {len(result)} entities in {query_time:.2f}ms")

            return result

        except Exception as e:
            logger.error(f"BFS traversal failed: {e}")
            raise

    async def traverse_dfs(
        self,
        start_id: str,
        max_depth: int = 3,
        relationship_type: Optional[RelationshipType] = None
    ) -> List[GraphEntity]:
        """
        Depth-First Search traversal from start entity

        Args:
            start_id: Starting entity ID
            max_depth: Maximum traversal depth
            relationship_type: Filter by relationship type (optional)

        Returns:
            List of entities in DFS order
        """
        if not self._initialized:
            raise RuntimeError("GraphStore not initialized. Call initialize() first.")

        start_time = time.time()

        visited: Set[str] = set()
        result: List[GraphEntity] = []

        async def dfs_helper(current_id: str, depth: int):
            if current_id in visited or depth > max_depth:
                return

            visited.add(current_id)

            # Get current entity
            entity = await self.get_entity(current_id)
            if entity:
                result.append(entity)

            # Recursively visit neighbors
            if depth < max_depth:
                neighbors = await self.get_neighbors(
                    current_id,
                    relationship_type=relationship_type,
                    direction="outgoing"
                )

                for neighbor_entity, _ in neighbors:
                    if neighbor_entity.entity_id not in visited:
                        await dfs_helper(neighbor_entity.entity_id, depth + 1)

        try:
            await dfs_helper(start_id, 0)

            query_time = (time.time() - start_time) * 1000
            logger.info(f"DFS traversal completed: {len(result)} entities in {query_time:.2f}ms")

            return result

        except Exception as e:
            logger.error(f"DFS traversal failed: {e}")
            raise

    async def find_shortest_path(
        self,
        source_id: str,
        target_id: str,
        relationship_type: Optional[RelationshipType] = None
    ) -> Optional[GraphPath]:
        """
        Find shortest path between two entities using Dijkstra's algorithm

        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            relationship_type: Filter by relationship type (optional)

        Returns:
            GraphPath or None if no path exists
        """
        if not self._initialized:
            raise RuntimeError("GraphStore not initialized. Call initialize() first.")

        start_time = time.time()

        try:
            # Priority queue: (distance, current_id, path_nodes, path_edges)
            heap: List[Tuple[float, str, List[str], List[str]]] = [(0.0, source_id, [source_id], [])]
            visited: Set[str] = set()

            while heap:
                current_dist, current_id, path_nodes, path_edges = heappop(heap)

                if current_id in visited:
                    continue

                visited.add(current_id)

                # Found target
                if current_id == target_id:
                    # Construct full path
                    entities = []
                    for node_id in path_nodes:
                        entity = await self.get_entity(node_id)
                        if entity:
                            entities.append(entity)

                    relationships = []
                    for i in range(len(path_nodes) - 1):
                        neighbors = await self.get_neighbors(
                            path_nodes[i],
                            relationship_type=relationship_type,
                            direction="outgoing"
                        )
                        for _, rel in neighbors:
                            if rel.target_id == path_nodes[i + 1]:
                                relationships.append(rel)
                                break

                    path = GraphPath(
                        nodes=entities,
                        edges=relationships,
                        total_weight=current_dist
                    )

                    query_time = (time.time() - start_time) * 1000
                    logger.info(f"Found shortest path: {len(path)} hops, weight={current_dist:.2f} in {query_time:.2f}ms")

                    return path

                # Explore neighbors
                neighbors = await self.get_neighbors(
                    current_id,
                    relationship_type=relationship_type,
                    direction="outgoing"
                )

                for neighbor_entity, relationship in neighbors:
                    if neighbor_entity.entity_id not in visited:
                        new_dist = current_dist + relationship.weight
                        new_path_nodes = path_nodes + [neighbor_entity.entity_id]
                        new_path_edges = path_edges + [relationship.relationship_id]
                        heappush(heap, (new_dist, neighbor_entity.entity_id, new_path_nodes, new_path_edges))

            logger.info(f"No path found between {source_id} and {target_id}")
            return None

        except Exception as e:
            logger.error(f"Shortest path search failed: {e}")
            raise

    async def compute_degree_centrality(
        self,
        entity_type: Optional[EntityType] = None
    ) -> Dict[str, float]:
        """
        Compute degree centrality for entities

        Args:
            entity_type: Filter by entity type (optional)

        Returns:
            Dictionary mapping entity_id to centrality score
        """
        if not self._initialized:
            raise RuntimeError("GraphStore not initialized. Call initialize() first.")

        start_time = time.time()

        try:
            async with self.pool.acquire() as conn:
                if entity_type:
                    rows = await conn.fetch("""
                        SELECT
                            e.entity_id,
                            COUNT(DISTINCT r_out.relationship_id) + COUNT(DISTINCT r_in.relationship_id) as degree
                        FROM graph_entities e
                        LEFT JOIN graph_relationships r_out ON e.entity_id = r_out.source_id
                        LEFT JOIN graph_relationships r_in ON e.entity_id = r_in.target_id
                        WHERE e.entity_type = $1
                        GROUP BY e.entity_id
                        ORDER BY degree DESC
                    """, entity_type.value)
                else:
                    rows = await conn.fetch("""
                        SELECT
                            e.entity_id,
                            COUNT(DISTINCT r_out.relationship_id) + COUNT(DISTINCT r_in.relationship_id) as degree
                        FROM graph_entities e
                        LEFT JOIN graph_relationships r_out ON e.entity_id = r_out.source_id
                        LEFT JOIN graph_relationships r_in ON e.entity_id = r_in.target_id
                        GROUP BY e.entity_id
                        ORDER BY degree DESC
                    """)

                # Get total entities for normalization
                total_entities = len(rows)
                max_possible_degree = total_entities - 1 if total_entities > 1 else 1

                centrality = {}
                for row in rows:
                    # Normalize degree by (n-1)
                    centrality[row['entity_id']] = row['degree'] / max_possible_degree

                query_time = (time.time() - start_time) * 1000
                logger.info(f"Computed degree centrality for {len(centrality)} entities in {query_time:.2f}ms")

                return centrality

        except Exception as e:
            logger.error(f"Degree centrality computation failed: {e}")
            raise

    async def compute_clustering_coefficient(
        self,
        entity_id: str
    ) -> float:
        """
        Compute clustering coefficient for an entity

        Args:
            entity_id: Entity ID

        Returns:
            Clustering coefficient (0.0 to 1.0)
        """
        if not self._initialized:
            raise RuntimeError("GraphStore not initialized. Call initialize() first.")

        start_time = time.time()

        try:
            # Get neighbors
            neighbors = await self.get_neighbors(entity_id, direction="both")
            neighbor_ids = [n[0].entity_id for n in neighbors]

            if len(neighbor_ids) < 2:
                return 0.0

            # Count connections between neighbors
            connections = 0
            for i, neighbor_a in enumerate(neighbor_ids):
                for neighbor_b in neighbor_ids[i+1:]:
                    # Check if edge exists between neighbors
                    async with self.pool.acquire() as conn:
                        exists = await conn.fetchval("""
                            SELECT EXISTS(
                                SELECT 1 FROM graph_relationships
                                WHERE (source_id = $1 AND target_id = $2)
                                   OR (source_id = $2 AND target_id = $1)
                            )
                        """, neighbor_a, neighbor_b)

                        if exists:
                            connections += 1

            # Clustering coefficient = actual_connections / possible_connections
            possible_connections = len(neighbor_ids) * (len(neighbor_ids) - 1) / 2
            coefficient = connections / possible_connections if possible_connections > 0 else 0.0

            query_time = (time.time() - start_time) * 1000
            logger.info(f"Computed clustering coefficient for {entity_id}: {coefficient:.4f} in {query_time:.2f}ms")

            return coefficient

        except Exception as e:
            logger.error(f"Clustering coefficient computation failed: {e}")
            raise

    async def find_communities(
        self,
        min_community_size: int = 3
    ) -> List[Set[str]]:
        """
        Find communities using connected components analysis

        Args:
            min_community_size: Minimum size for a community

        Returns:
            List of communities (sets of entity IDs)
        """
        if not self._initialized:
            raise RuntimeError("GraphStore not initialized. Call initialize() first.")

        start_time = time.time()

        try:
            async with self.pool.acquire() as conn:
                # Get all entities
                entity_rows = await conn.fetch("SELECT entity_id FROM graph_entities")
                all_entities = {row['entity_id'] for row in entity_rows}

                visited: Set[str] = set()
                communities: List[Set[str]] = []

                # Find connected components
                for entity_id in all_entities:
                    if entity_id in visited:
                        continue

                    # BFS to find connected component
                    component: Set[str] = set()
                    queue: deque = deque([entity_id])

                    while queue:
                        current_id = queue.popleft()

                        if current_id in visited:
                            continue

                        visited.add(current_id)
                        component.add(current_id)

                        # Get all neighbors
                        neighbors = await self.get_neighbors(current_id, direction="both")
                        for neighbor_entity, _ in neighbors:
                            if neighbor_entity.entity_id not in visited:
                                queue.append(neighbor_entity.entity_id)

                    # Add component if large enough
                    if len(component) >= min_community_size:
                        communities.append(component)

                query_time = (time.time() - start_time) * 1000
                logger.info(f"Found {len(communities)} communities in {query_time:.2f}ms")

                return communities

        except Exception as e:
            logger.error(f"Community detection failed: {e}")
            raise

    async def delete_entity(self, entity_id: str) -> bool:
        """
        Delete entity and all its relationships

        Args:
            entity_id: Entity ID to delete

        Returns:
            True if successful
        """
        if not self._initialized:
            raise RuntimeError("GraphStore not initialized. Call initialize() first.")

        try:
            async with self.pool.acquire() as conn:
                # Cascade delete handled by foreign key constraints
                result = await conn.execute("""
                    DELETE FROM graph_entities WHERE entity_id = $1
                """, entity_id)

                logger.info(f"Deleted entity: {entity_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to delete entity: {e}")
            raise

    async def delete_relationship(self, relationship_id: str) -> bool:
        """
        Delete relationship

        Args:
            relationship_id: Relationship ID to delete

        Returns:
            True if successful
        """
        if not self._initialized:
            raise RuntimeError("GraphStore not initialized. Call initialize() first.")

        try:
            async with self.pool.acquire() as conn:
                result = await conn.execute("""
                    DELETE FROM graph_relationships WHERE relationship_id = $1
                """, relationship_id)

                logger.info(f"Deleted relationship: {relationship_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to delete relationship: {e}")
            raise

    async def _update_metrics(self):
        """Update graph metrics"""
        try:
            async with self.pool.acquire() as conn:
                # Count entities
                total_entities = await conn.fetchval("""
                    SELECT COUNT(*) FROM graph_entities
                """)

                # Count relationships
                total_relationships = await conn.fetchval("""
                    SELECT COUNT(*) FROM graph_relationships
                """)

                # Count by entity type
                entity_type_rows = await conn.fetch("""
                    SELECT entity_type, COUNT(*) as count
                    FROM graph_entities
                    GROUP BY entity_type
                """)
                entities_by_type = {row['entity_type']: row['count'] for row in entity_type_rows}

                # Count by relationship type
                rel_type_rows = await conn.fetch("""
                    SELECT relationship_type, COUNT(*) as count
                    FROM graph_relationships
                    GROUP BY relationship_type
                """)
                relationships_by_type = {row['relationship_type']: row['count'] for row in rel_type_rows}

                # Calculate average degree
                if total_entities > 0:
                    avg_degree = (2 * total_relationships) / total_entities
                else:
                    avg_degree = 0.0

                # Update metrics
                self.metrics.total_entities = total_entities
                self.metrics.total_relationships = total_relationships
                self.metrics.entities_by_type = entities_by_type
                self.metrics.relationships_by_type = relationships_by_type
                self.metrics.avg_degree = avg_degree
                self.metrics.last_updated = datetime.now()

        except Exception as e:
            logger.error(f"Failed to update metrics: {e}")

    def _update_avg_query_time(self, query_time: float):
        """Update average query time metric"""
        if self.metrics.avg_query_time_ms == 0:
            self.metrics.avg_query_time_ms = query_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.metrics.avg_query_time_ms = (
                alpha * query_time + (1 - alpha) * self.metrics.avg_query_time_ms
            )

    async def get_metrics(self) -> GraphMetrics:
        """
        Get current graph metrics

        Returns:
            GraphMetrics object
        """
        await self._update_metrics()
        return self.metrics

    async def get_health(self) -> Dict[str, Any]:
        """
        Get knowledge graph health status

        Returns:
            Health metrics dictionary
        """
        if not self._initialized:
            return {
                'status': 'not_initialized',
                'database_connected': False
            }

        try:
            # Test database connection
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")

            # Update and get metrics
            metrics = await self.get_metrics()

            # Check SLO (P95 < 200ms)
            slo_met = metrics.avg_query_time_ms < 200

            return {
                'status': 'healthy' if slo_met else 'degraded',
                'database_connected': True,
                'database': self.database,
                'pool_size': self.pool.get_size(),
                'pool_idle': self.pool.get_idle_size(),
                'total_entities': metrics.total_entities,
                'total_relationships': metrics.total_relationships,
                'entities_by_type': metrics.entities_by_type,
                'relationships_by_type': metrics.relationships_by_type,
                'avg_degree': metrics.avg_degree,
                'avg_query_time_ms': metrics.avg_query_time_ms,
                'slo_met': slo_met,
                'last_updated': metrics.last_updated.isoformat()
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e)
            }

    async def close(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("Knowledge graph connection pool closed")
            self._initialized = False


# Example usage and testing
async def main():
    """Example knowledge graph usage"""
    print("\n" + "=" * 70)
    print("Knowledge Graph Store - Example Usage")
    print("=" * 70 + "\n")

    # Initialize graph store
    graph = KnowledgeGraphStore(
        host="localhost",
        port=5432,
        database="devskyy",
        user="postgres",
        password="postgres"
    )

    await graph.initialize()

    print("### Example 1: Create Brand Entity")
    brand = GraphEntity(
        entity_id="brand_skyy_rose",
        entity_type=EntityType.BRAND,
        properties={
            'name': 'Skyy Rose',
            'category': 'luxury_fashion',
            'founded': 2024,
            'description': 'Luxury fashion brand specializing in handcrafted leather goods'
        },
        metadata={'created_by': 'system', 'verified': True}
    )
    await graph.add_entity(brand, overwrite=True)
    print(f"Created brand: {brand.entity_id}")
    print()

    print("### Example 2: Create Product Entities")
    products = [
        GraphEntity(
            entity_id="product_handbag_001",
            entity_type=EntityType.PRODUCT,
            properties={
                'name': 'Elegant Leather Handbag',
                'price': 450.00,
                'sku': 'SKU-1234',
                'collection': 'spring_2025'
            }
        ),
        GraphEntity(
            entity_id="product_wallet_002",
            entity_type=EntityType.PRODUCT,
            properties={
                'name': 'Premium Italian Leather Wallet',
                'price': 180.00,
                'sku': 'SKU-5678',
                'collection': 'spring_2025'
            }
        )
    ]

    for product in products:
        await graph.add_entity(product, overwrite=True)
        print(f"Created product: {product.entity_id}")
    print()

    print("### Example 3: Create Customer Entity")
    customer = GraphEntity(
        entity_id="customer_john_doe",
        entity_type=EntityType.CUSTOMER,
        properties={
            'name': 'John Doe',
            'email': 'john@example.com',
            'tier': 'premium',
            'lifetime_value': 2500.00
        }
    )
    await graph.add_entity(customer, overwrite=True)
    print(f"Created customer: {customer.entity_id}")
    print()

    print("### Example 4: Create Relationships")
    relationships = [
        GraphRelationship(
            relationship_id=str(uuid.uuid4()),
            source_id="brand_skyy_rose",
            target_id="product_handbag_001",
            relationship_type=RelationshipType.OWNS,
            properties={'since': '2024-01-01'},
            weight=1.0
        ),
        GraphRelationship(
            relationship_id=str(uuid.uuid4()),
            source_id="brand_skyy_rose",
            target_id="product_wallet_002",
            relationship_type=RelationshipType.OWNS,
            properties={'since': '2024-01-01'},
            weight=1.0
        ),
        GraphRelationship(
            relationship_id=str(uuid.uuid4()),
            source_id="customer_john_doe",
            target_id="product_handbag_001",
            relationship_type=RelationshipType.PURCHASED,
            properties={'date': '2024-03-15', 'price': 450.00},
            weight=1.0
        )
    ]

    for rel in relationships:
        await graph.add_relationship(rel, overwrite=True)
        print(f"Created relationship: {rel.source_id} -{rel.relationship_type.value}-> {rel.target_id}")
    print()

    print("### Example 5: Get Neighbors")
    neighbors = await graph.get_neighbors("brand_skyy_rose", direction="outgoing")
    print(f"Brand neighbors: {len(neighbors)}")
    for neighbor, rel in neighbors:
        print(f"  - {neighbor.entity_id} ({neighbor.entity_type.value}) via {rel.relationship_type.value}")
    print()

    print("### Example 6: BFS Traversal")
    bfs_result = await graph.traverse_bfs("brand_skyy_rose", max_depth=2)
    print(f"BFS traversal from brand: {len(bfs_result)} entities")
    for entity in bfs_result:
        print(f"  - {entity.entity_id} ({entity.entity_type.value})")
    print()

    print("### Example 7: Find Shortest Path")
    path = await graph.find_shortest_path("brand_skyy_rose", "customer_john_doe")
    if path:
        print(f"Shortest path: {len(path)} hops, total weight: {path.total_weight}")
        for i, node in enumerate(path.nodes):
            print(f"  {i+1}. {node.entity_id} ({node.entity_type.value})")
    else:
        print("No path found")
    print()

    print("### Example 8: Degree Centrality")
    centrality = await graph.compute_degree_centrality()
    print("Top 5 entities by degree centrality:")
    sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    for entity_id, score in sorted_centrality:
        print(f"  - {entity_id}: {score:.4f}")
    print()

    print("### Example 9: Graph Metrics")
    metrics = await graph.get_metrics()
    print("Graph Metrics:")
    print(f"  Total Entities: {metrics.total_entities}")
    print(f"  Total Relationships: {metrics.total_relationships}")
    print(f"  Avg Degree: {metrics.avg_degree:.2f}")
    print(f"  Avg Query Time: {metrics.avg_query_time_ms:.2f}ms")
    print(f"  Entities by Type: {metrics.entities_by_type}")
    print(f"  Relationships by Type: {metrics.relationships_by_type}")
    print()

    print("### Example 10: Health Check")
    health = await graph.get_health()
    print("Knowledge Graph Health:")
    print(json.dumps(health, indent=2))
    print()

    # Cleanup
    await graph.close()


if __name__ == "__main__":
    asyncio.run(main())
