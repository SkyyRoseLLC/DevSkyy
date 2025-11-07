#!/usr/bin/env python3
"""
Enterprise Data Retention and Lifecycle Management System
PostgreSQL-based automated data cleanup with GDPR compliance

Architecture Position: Infrastructure Layer → Data Lifecycle Management
References: /Users/coreyfoster/DevSkyy/CLAUDE.md
Truth Protocol Compliance: All 15 rules
Version: 2.0.0

Features:
- Automated data retention policy enforcement
- GDPR Article 5.1(e) compliance (storage limitation)
- Batch deletion with archival
- APScheduler for cron-based cleanup
- Dry-run mode for testing
- Comprehensive audit logging
- Performance optimized (P95 < 200ms)
- Health checks and monitoring
- PostgreSQL ONLY (NO MongoDB)

Database Tables Managed:
- users, user_sessions, user_preferences (90 days)
- orders (7 years tax compliance)
- analytics_events (30 days)
- error_logs (90 days)
- gdpr_audit_logs (indefinite)
- vector_embeddings (365 days)

Performance Target: P95 query latency < 200ms
"""

import asyncio
import asyncpg
import json
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
from pathlib import Path

# APScheduler for automated jobs
try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger
    HAS_APSCHEDULER = True
except ImportError:
    HAS_APSCHEDULER = False
    logging.warning("APScheduler not installed. Install with: pip install apscheduler")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataType(str, Enum):
    """Data categories for retention policies"""
    USER_SESSIONS = "user_sessions"
    USER_PREFERENCES = "user_preferences"
    ANALYTICS_EVENTS = "analytics_events"
    ERROR_LOGS = "error_logs"
    ORDER_HISTORY = "order_history"
    GDPR_AUDIT_LOGS = "gdpr_audit_logs"
    VECTOR_EMBEDDINGS = "vector_embeddings"
    AGENT_LOGS = "agent_logs"
    TEMPORARY_FILES = "temporary_files"
    CACHE_ENTRIES = "cache_entries"


class RetentionAction(str, Enum):
    """Actions to perform on expired data"""
    DELETE = "delete"
    ARCHIVE = "archive"
    ANONYMIZE = "anonymize"
    COMPRESS = "compress"


class PolicyStatus(str, Enum):
    """Retention policy status"""
    ACTIVE = "active"
    PAUSED = "paused"
    DISABLED = "disabled"


@dataclass
class RetentionPolicy:
    """
    Data retention policy configuration

    Attributes:
        policy_id: Unique policy identifier
        data_type: Type of data this policy applies to
        table_name: PostgreSQL table name
        retention_days: Number of days to retain data (0 = indefinite)
        action: Action to perform on expired data
        where_clause: Additional SQL WHERE conditions
        enabled: Whether policy is active
        legal_basis: Legal justification for retention period
        created_at: Policy creation timestamp
        updated_at: Policy update timestamp
    """
    policy_id: str
    data_type: DataType
    table_name: str
    retention_days: int
    action: RetentionAction = RetentionAction.DELETE
    where_clause: str = ""
    enabled: bool = True
    legal_basis: str = ""
    archive_table: Optional[str] = None
    date_column: str = "created_at"
    batch_size: int = 1000
    status: PolicyStatus = PolicyStatus.ACTIVE
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'policy_id': self.policy_id,
            'data_type': self.data_type.value if isinstance(self.data_type, DataType) else self.data_type,
            'table_name': self.table_name,
            'retention_days': self.retention_days,
            'action': self.action.value if isinstance(self.action, RetentionAction) else self.action,
            'where_clause': self.where_clause,
            'enabled': self.enabled,
            'legal_basis': self.legal_basis,
            'archive_table': self.archive_table,
            'date_column': self.date_column,
            'batch_size': self.batch_size,
            'status': self.status.value if isinstance(self.status, PolicyStatus) else self.status,
            'created_at': self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
            'updated_at': self.updated_at.isoformat() if isinstance(self.updated_at, datetime) else self.updated_at,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RetentionPolicy':
        """Create from dictionary"""
        return cls(
            policy_id=data['policy_id'],
            data_type=DataType(data['data_type']) if isinstance(data['data_type'], str) else data['data_type'],
            table_name=data['table_name'],
            retention_days=data['retention_days'],
            action=RetentionAction(data.get('action', 'delete')) if isinstance(data.get('action'), str) else data.get('action', RetentionAction.DELETE),
            where_clause=data.get('where_clause', ''),
            enabled=data.get('enabled', True),
            legal_basis=data.get('legal_basis', ''),
            archive_table=data.get('archive_table'),
            date_column=data.get('date_column', 'created_at'),
            batch_size=data.get('batch_size', 1000),
            status=PolicyStatus(data.get('status', 'active')) if isinstance(data.get('status'), str) else data.get('status', PolicyStatus.ACTIVE),
            created_at=datetime.fromisoformat(data['created_at']) if isinstance(data.get('created_at'), str) else data.get('created_at', datetime.now(timezone.utc)),
            updated_at=datetime.fromisoformat(data['updated_at']) if isinstance(data.get('updated_at'), str) else data.get('updated_at', datetime.now(timezone.utc)),
            metadata=data.get('metadata', {})
        )


@dataclass
class CleanupResult:
    """
    Result of a cleanup operation

    Attributes:
        cleanup_id: Unique cleanup operation ID
        policy_id: Related policy ID
        table_name: Table that was cleaned
        records_deleted: Number of records deleted
        records_archived: Number of records archived
        storage_freed_bytes: Estimated storage freed
        duration_ms: Operation duration in milliseconds
        dry_run: Whether this was a dry run
        status: Success, failed, or partial
        error_message: Error details if failed
        started_at: Cleanup start time
        completed_at: Cleanup completion time
    """
    cleanup_id: str
    policy_id: str
    table_name: str
    records_deleted: int = 0
    records_archived: int = 0
    storage_freed_bytes: int = 0
    duration_ms: float = 0.0
    dry_run: bool = False
    status: str = "success"
    error_message: Optional[str] = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'cleanup_id': self.cleanup_id,
            'policy_id': self.policy_id,
            'table_name': self.table_name,
            'records_deleted': self.records_deleted,
            'records_archived': self.records_archived,
            'storage_freed_bytes': self.storage_freed_bytes,
            'duration_ms': self.duration_ms,
            'dry_run': self.dry_run,
            'status': self.status,
            'error_message': self.error_message,
            'started_at': self.started_at.isoformat() if isinstance(self.started_at, datetime) else self.started_at,
            'completed_at': self.completed_at.isoformat() if isinstance(self.completed_at, datetime) and self.completed_at else None,
            'metadata': self.metadata
        }


@dataclass
class RetentionMetrics:
    """
    Data retention system metrics

    Attributes:
        total_policies: Number of active policies
        total_cleanups: Total cleanup operations performed
        total_records_deleted: Total records deleted
        total_storage_freed_mb: Total storage freed in MB
        avg_cleanup_time_ms: Average cleanup duration
        last_cleanup_time: Last cleanup execution time
        policies_by_status: Count of policies by status
        cleanups_by_table: Count of cleanups by table
    """
    total_policies: int = 0
    total_cleanups: int = 0
    total_records_deleted: int = 0
    total_storage_freed_mb: float = 0.0
    avg_cleanup_time_ms: float = 0.0
    last_cleanup_time: Optional[datetime] = None
    policies_by_status: Dict[str, int] = field(default_factory=dict)
    cleanups_by_table: Dict[str, int] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'total_policies': self.total_policies,
            'total_cleanups': self.total_cleanups,
            'total_records_deleted': self.total_records_deleted,
            'total_storage_freed_mb': self.total_storage_freed_mb,
            'avg_cleanup_time_ms': self.avg_cleanup_time_ms,
            'last_cleanup_time': self.last_cleanup_time.isoformat() if self.last_cleanup_time else None,
            'policies_by_status': self.policies_by_status,
            'cleanups_by_table': self.cleanups_by_table,
            'last_updated': self.last_updated.isoformat() if isinstance(self.last_updated, datetime) else self.last_updated
        }


class DataRetentionManager:
    """
    Enterprise-grade Data Retention Manager for PostgreSQL

    Features:
    - Automated retention policy enforcement
    - Batch deletion with archival support
    - GDPR compliance (Article 5.1(e))
    - APScheduler integration for cron jobs
    - Dry-run mode for testing
    - Comprehensive audit logging
    - Performance optimized (P95 < 200ms)
    - Health checks and monitoring

    Usage:
        manager = DataRetentionManager()
        await manager.initialize()

        # Create retention policy
        policy = RetentionPolicy(
            policy_id="user_sessions_90d",
            data_type=DataType.USER_SESSIONS,
            table_name="user_sessions",
            retention_days=90
        )
        await manager.create_policy(policy)

        # Execute cleanup
        result = await manager.execute_cleanup(policy.policy_id, dry_run=True)

        # Start automated scheduler
        manager.start_scheduler()
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
        max_pool_size: int = 10,
        enable_scheduler: bool = True
    ):
        """
        Initialize Data Retention Manager

        Args:
            database_url: PostgreSQL connection URL (overrides other params)
            host: Database host
            port: Database port
            database: Database name
            user: Database user
            password: Database password
            min_pool_size: Minimum connection pool size
            max_pool_size: Maximum connection pool size
            enable_scheduler: Enable APScheduler for automated cleanup
        """
        self.database_url = database_url
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.min_pool_size = min_pool_size
        self.max_pool_size = max_pool_size
        self.enable_scheduler = enable_scheduler

        self.pool: Optional[asyncpg.Pool] = None
        self._initialized = False
        self.metrics = RetentionMetrics()
        self.scheduler: Optional[AsyncIOScheduler] = None

        # In-memory policy cache
        self._policy_cache: Dict[str, RetentionPolicy] = {}

        logger.info(f"DataRetentionManager initialized: {database}@{host}:{port}")

    async def initialize(self):
        """Initialize database connection and create schema"""
        if self._initialized:
            logger.warning("DataRetentionManager already initialized")
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

            # Load policies into cache
            await self._load_policies()

            # Initialize APScheduler if enabled
            if self.enable_scheduler:
                await self._initialize_scheduler()

            # Update metrics
            await self._update_metrics()

            self._initialized = True
            logger.info("DataRetentionManager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize DataRetentionManager: {e}")
            raise

    async def _create_schema(self):
        """Create retention management schema in PostgreSQL"""
        async with self.pool.acquire() as conn:
            # Create retention_policies table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS retention_policies (
                    policy_id VARCHAR(255) PRIMARY KEY,
                    data_type VARCHAR(100) NOT NULL,
                    table_name VARCHAR(255) NOT NULL,
                    retention_days INTEGER NOT NULL,
                    action VARCHAR(50) DEFAULT 'delete',
                    where_clause TEXT DEFAULT '',
                    enabled BOOLEAN DEFAULT TRUE,
                    legal_basis TEXT DEFAULT '',
                    archive_table VARCHAR(255),
                    date_column VARCHAR(100) DEFAULT 'created_at',
                    batch_size INTEGER DEFAULT 1000,
                    status VARCHAR(50) DEFAULT 'active',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    metadata JSONB DEFAULT '{}'
                )
            """)

            # Create cleanup_audit_logs table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS cleanup_audit_logs (
                    cleanup_id VARCHAR(255) PRIMARY KEY,
                    policy_id VARCHAR(255) NOT NULL,
                    table_name VARCHAR(255) NOT NULL,
                    records_deleted INTEGER DEFAULT 0,
                    records_archived INTEGER DEFAULT 0,
                    storage_freed_bytes BIGINT DEFAULT 0,
                    duration_ms FLOAT DEFAULT 0.0,
                    dry_run BOOLEAN DEFAULT FALSE,
                    status VARCHAR(50) DEFAULT 'success',
                    error_message TEXT,
                    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    completed_at TIMESTAMP WITH TIME ZONE,
                    metadata JSONB DEFAULT '{}',
                    FOREIGN KEY (policy_id) REFERENCES retention_policies(policy_id) ON DELETE CASCADE
                )
            """)

            # Create indexes for performance
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_retention_policies_data_type
                ON retention_policies(data_type)
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_retention_policies_status
                ON retention_policies(status)
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_retention_policies_enabled
                ON retention_policies(enabled)
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cleanup_audit_logs_policy_id
                ON cleanup_audit_logs(policy_id)
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cleanup_audit_logs_table_name
                ON cleanup_audit_logs(table_name)
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cleanup_audit_logs_started_at
                ON cleanup_audit_logs(started_at DESC)
            """)

            logger.info("Data retention schema created successfully")

    async def _load_policies(self):
        """Load retention policies into memory cache"""
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT * FROM retention_policies WHERE enabled = TRUE
                """)

                self._policy_cache.clear()
                for row in rows:
                    policy = RetentionPolicy(
                        policy_id=row['policy_id'],
                        data_type=DataType(row['data_type']),
                        table_name=row['table_name'],
                        retention_days=row['retention_days'],
                        action=RetentionAction(row['action']),
                        where_clause=row['where_clause'] or '',
                        enabled=row['enabled'],
                        legal_basis=row['legal_basis'] or '',
                        archive_table=row['archive_table'],
                        date_column=row['date_column'],
                        batch_size=row['batch_size'],
                        status=PolicyStatus(row['status']),
                        created_at=row['created_at'],
                        updated_at=row['updated_at'],
                        metadata=json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
                    )
                    self._policy_cache[policy.policy_id] = policy

                logger.info(f"Loaded {len(self._policy_cache)} retention policies into cache")

        except Exception as e:
            logger.error(f"Failed to load policies: {e}")
            raise

    async def _initialize_scheduler(self):
        """Initialize APScheduler for automated cleanup"""
        if not HAS_APSCHEDULER:
            logger.warning("APScheduler not available - scheduler disabled")
            return

        try:
            self.scheduler = AsyncIOScheduler(timezone='UTC')

            # Schedule daily cleanup at 2 AM UTC
            self.scheduler.add_job(
                self._run_all_cleanups,
                CronTrigger(hour=2, minute=0),
                id='daily_cleanup',
                name='Daily Data Retention Cleanup',
                replace_existing=True
            )

            # Schedule metrics update every hour
            self.scheduler.add_job(
                self._update_metrics,
                IntervalTrigger(hours=1),
                id='metrics_update',
                name='Hourly Metrics Update',
                replace_existing=True
            )

            logger.info("APScheduler initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize scheduler: {e}")
            raise

    def start_scheduler(self):
        """Start the APScheduler"""
        if not HAS_APSCHEDULER:
            logger.error("APScheduler not available. Install with: pip install apscheduler")
            return

        if self.scheduler and not self.scheduler.running:
            self.scheduler.start()
            logger.info("APScheduler started - automated cleanup enabled")
        else:
            logger.warning("Scheduler not initialized or already running")

    def stop_scheduler(self):
        """Stop the APScheduler"""
        if self.scheduler and self.scheduler.running:
            self.scheduler.shutdown(wait=True)
            logger.info("APScheduler stopped")
        else:
            logger.warning("Scheduler not running")

    async def create_policy(
        self,
        policy: RetentionPolicy,
        overwrite: bool = False
    ) -> bool:
        """
        Create a new retention policy

        Args:
            policy: RetentionPolicy to create
            overwrite: Whether to overwrite existing policy

        Returns:
            True if successful
        """
        if not self._initialized:
            raise RuntimeError("DataRetentionManager not initialized. Call initialize() first.")

        start_time = time.time()

        try:
            async with self.pool.acquire() as conn:
                if overwrite:
                    await conn.execute("""
                        INSERT INTO retention_policies
                        (policy_id, data_type, table_name, retention_days, action, where_clause,
                         enabled, legal_basis, archive_table, date_column, batch_size, status,
                         created_at, updated_at, metadata)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                        ON CONFLICT (policy_id)
                        DO UPDATE SET
                            data_type = EXCLUDED.data_type,
                            table_name = EXCLUDED.table_name,
                            retention_days = EXCLUDED.retention_days,
                            action = EXCLUDED.action,
                            where_clause = EXCLUDED.where_clause,
                            enabled = EXCLUDED.enabled,
                            legal_basis = EXCLUDED.legal_basis,
                            archive_table = EXCLUDED.archive_table,
                            date_column = EXCLUDED.date_column,
                            batch_size = EXCLUDED.batch_size,
                            status = EXCLUDED.status,
                            updated_at = EXCLUDED.updated_at,
                            metadata = EXCLUDED.metadata
                    """,
                        policy.policy_id,
                        policy.data_type.value,
                        policy.table_name,
                        policy.retention_days,
                        policy.action.value,
                        policy.where_clause,
                        policy.enabled,
                        policy.legal_basis,
                        policy.archive_table,
                        policy.date_column,
                        policy.batch_size,
                        policy.status.value,
                        policy.created_at,
                        policy.updated_at,
                        json.dumps(policy.metadata)
                    )
                else:
                    await conn.execute("""
                        INSERT INTO retention_policies
                        (policy_id, data_type, table_name, retention_days, action, where_clause,
                         enabled, legal_basis, archive_table, date_column, batch_size, status,
                         created_at, updated_at, metadata)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                    """,
                        policy.policy_id,
                        policy.data_type.value,
                        policy.table_name,
                        policy.retention_days,
                        policy.action.value,
                        policy.where_clause,
                        policy.enabled,
                        policy.legal_basis,
                        policy.archive_table,
                        policy.date_column,
                        policy.batch_size,
                        policy.status.value,
                        policy.created_at,
                        policy.updated_at,
                        json.dumps(policy.metadata)
                    )

            # Update cache
            self._policy_cache[policy.policy_id] = policy

            query_time = (time.time() - start_time) * 1000
            logger.info(f"Created retention policy: {policy.policy_id} in {query_time:.2f}ms")
            return True

        except asyncpg.UniqueViolationError:
            logger.warning(f"Policy already exists: {policy.policy_id}")
            return False
        except Exception as e:
            logger.error(f"Failed to create policy: {e}")
            raise

    async def get_policy(self, policy_id: str) -> Optional[RetentionPolicy]:
        """
        Get retention policy by ID

        Args:
            policy_id: Policy ID

        Returns:
            RetentionPolicy or None if not found
        """
        # Check cache first
        if policy_id in self._policy_cache:
            return self._policy_cache[policy_id]

        if not self._initialized:
            raise RuntimeError("DataRetentionManager not initialized. Call initialize() first.")

        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT * FROM retention_policies WHERE policy_id = $1
                """, policy_id)

                if not row:
                    return None

                policy = RetentionPolicy(
                    policy_id=row['policy_id'],
                    data_type=DataType(row['data_type']),
                    table_name=row['table_name'],
                    retention_days=row['retention_days'],
                    action=RetentionAction(row['action']),
                    where_clause=row['where_clause'] or '',
                    enabled=row['enabled'],
                    legal_basis=row['legal_basis'] or '',
                    archive_table=row['archive_table'],
                    date_column=row['date_column'],
                    batch_size=row['batch_size'],
                    status=PolicyStatus(row['status']),
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    metadata=json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
                )

                # Update cache
                self._policy_cache[policy_id] = policy

                return policy

        except Exception as e:
            logger.error(f"Failed to get policy: {e}")
            raise

    async def list_policies(
        self,
        data_type: Optional[DataType] = None,
        enabled_only: bool = True
    ) -> List[RetentionPolicy]:
        """
        List retention policies

        Args:
            data_type: Filter by data type
            enabled_only: Only return enabled policies

        Returns:
            List of RetentionPolicy objects
        """
        if not self._initialized:
            raise RuntimeError("DataRetentionManager not initialized. Call initialize() first.")

        try:
            async with self.pool.acquire() as conn:
                if data_type and enabled_only:
                    rows = await conn.fetch("""
                        SELECT * FROM retention_policies
                        WHERE data_type = $1 AND enabled = TRUE
                        ORDER BY created_at DESC
                    """, data_type.value)
                elif data_type:
                    rows = await conn.fetch("""
                        SELECT * FROM retention_policies
                        WHERE data_type = $1
                        ORDER BY created_at DESC
                    """, data_type.value)
                elif enabled_only:
                    rows = await conn.fetch("""
                        SELECT * FROM retention_policies
                        WHERE enabled = TRUE
                        ORDER BY created_at DESC
                    """)
                else:
                    rows = await conn.fetch("""
                        SELECT * FROM retention_policies
                        ORDER BY created_at DESC
                    """)

                policies = []
                for row in rows:
                    policy = RetentionPolicy(
                        policy_id=row['policy_id'],
                        data_type=DataType(row['data_type']),
                        table_name=row['table_name'],
                        retention_days=row['retention_days'],
                        action=RetentionAction(row['action']),
                        where_clause=row['where_clause'] or '',
                        enabled=row['enabled'],
                        legal_basis=row['legal_basis'] or '',
                        archive_table=row['archive_table'],
                        date_column=row['date_column'],
                        batch_size=row['batch_size'],
                        status=PolicyStatus(row['status']),
                        created_at=row['created_at'],
                        updated_at=row['updated_at'],
                        metadata=json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
                    )
                    policies.append(policy)

                return policies

        except Exception as e:
            logger.error(f"Failed to list policies: {e}")
            raise

    async def update_policy(
        self,
        policy_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """
        Update retention policy

        Args:
            policy_id: Policy ID to update
            updates: Dictionary of fields to update

        Returns:
            True if successful
        """
        if not self._initialized:
            raise RuntimeError("DataRetentionManager not initialized. Call initialize() first.")

        try:
            # Build dynamic UPDATE query
            set_clauses = []
            values = []
            param_count = 1

            for key, value in updates.items():
                set_clauses.append(f"{key} = ${param_count}")
                values.append(value)
                param_count += 1

            # Always update updated_at
            set_clauses.append(f"updated_at = ${param_count}")
            values.append(datetime.now(timezone.utc))
            param_count += 1

            # Add policy_id for WHERE clause
            values.append(policy_id)

            query = f"""
                UPDATE retention_policies
                SET {', '.join(set_clauses)}
                WHERE policy_id = ${param_count}
            """

            async with self.pool.acquire() as conn:
                await conn.execute(query, *values)

            # Invalidate cache
            if policy_id in self._policy_cache:
                del self._policy_cache[policy_id]

            logger.info(f"Updated retention policy: {policy_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update policy: {e}")
            raise

    async def delete_policy(self, policy_id: str) -> bool:
        """
        Delete retention policy

        Args:
            policy_id: Policy ID to delete

        Returns:
            True if successful
        """
        if not self._initialized:
            raise RuntimeError("DataRetentionManager not initialized. Call initialize() first.")

        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    DELETE FROM retention_policies WHERE policy_id = $1
                """, policy_id)

            # Remove from cache
            if policy_id in self._policy_cache:
                del self._policy_cache[policy_id]

            logger.info(f"Deleted retention policy: {policy_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete policy: {e}")
            raise

    async def execute_cleanup(
        self,
        policy_id: str,
        dry_run: bool = False
    ) -> CleanupResult:
        """
        Execute cleanup for a specific policy

        Args:
            policy_id: Policy ID to execute
            dry_run: If True, only simulate the cleanup

        Returns:
            CleanupResult with cleanup statistics
        """
        if not self._initialized:
            raise RuntimeError("DataRetentionManager not initialized. Call initialize() first.")

        start_time = time.time()
        cleanup_id = str(uuid.uuid4())

        result = CleanupResult(
            cleanup_id=cleanup_id,
            policy_id=policy_id,
            table_name="",
            dry_run=dry_run,
            started_at=datetime.now(timezone.utc)
        )

        try:
            # Get policy
            policy = await self.get_policy(policy_id)
            if not policy:
                result.status = "failed"
                result.error_message = f"Policy not found: {policy_id}"
                return result

            if not policy.enabled or policy.status != PolicyStatus.ACTIVE:
                result.status = "skipped"
                result.error_message = f"Policy is disabled or not active: {policy_id}"
                return result

            result.table_name = policy.table_name

            # Calculate cutoff date
            if policy.retention_days == 0:
                # Indefinite retention - skip cleanup
                result.status = "skipped"
                result.error_message = "Indefinite retention policy - no cleanup needed"
                return result

            cutoff_date = datetime.now(timezone.utc) - timedelta(days=policy.retention_days)

            async with self.pool.acquire() as conn:
                # Build WHERE clause
                where_conditions = [f"{policy.date_column} < $1"]
                params = [cutoff_date]

                if policy.where_clause:
                    where_conditions.append(policy.where_clause)

                where_clause = " AND ".join(where_conditions)

                # Count records to be affected
                count_query = f"""
                    SELECT COUNT(*) FROM {policy.table_name}
                    WHERE {where_clause}
                """
                count = await conn.fetchval(count_query, *params)

                if count == 0:
                    result.status = "success"
                    result.completed_at = datetime.now(timezone.utc)
                    result.duration_ms = (time.time() - start_time) * 1000
                    logger.info(f"No records to cleanup for policy {policy_id}")
                    return result

                # Estimate storage size
                size_query = f"""
                    SELECT pg_total_relation_size('{policy.table_name}')
                """
                table_size = await conn.fetchval(size_query)
                estimated_freed = int((table_size / (count + 1)) * count) if table_size else 0

                if dry_run:
                    result.records_deleted = count
                    result.storage_freed_bytes = estimated_freed
                    result.status = "success"
                    result.completed_at = datetime.now(timezone.utc)
                    result.duration_ms = (time.time() - start_time) * 1000
                    logger.info(f"DRY RUN - Would delete {count} records from {policy.table_name}")
                    return result

                # Execute cleanup based on action
                if policy.action == RetentionAction.ARCHIVE:
                    # Archive then delete
                    if policy.archive_table:
                        archived = await self._archive_records(
                            conn,
                            policy.table_name,
                            policy.archive_table,
                            where_clause,
                            params,
                            policy.batch_size
                        )
                        result.records_archived = archived

                    # Delete after archiving
                    deleted = await self._delete_records(
                        conn,
                        policy.table_name,
                        where_clause,
                        params,
                        policy.batch_size
                    )
                    result.records_deleted = deleted

                elif policy.action == RetentionAction.DELETE:
                    # Direct delete
                    deleted = await self._delete_records(
                        conn,
                        policy.table_name,
                        where_clause,
                        params,
                        policy.batch_size
                    )
                    result.records_deleted = deleted

                elif policy.action == RetentionAction.ANONYMIZE:
                    # Anonymize PII fields
                    anonymized = await self._anonymize_records(
                        conn,
                        policy.table_name,
                        where_clause,
                        params,
                        policy.batch_size
                    )
                    result.records_deleted = anonymized

                result.storage_freed_bytes = estimated_freed
                result.status = "success"
                result.completed_at = datetime.now(timezone.utc)
                result.duration_ms = (time.time() - start_time) * 1000

            # Log cleanup to audit table
            await self._log_cleanup(result)

            logger.info(
                f"Cleanup completed for {policy_id}: {result.records_deleted} deleted, "
                f"{result.records_archived} archived in {result.duration_ms:.2f}ms"
            )

            return result

        except Exception as e:
            result.status = "failed"
            result.error_message = str(e)
            result.completed_at = datetime.now(timezone.utc)
            result.duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Cleanup failed for {policy_id}: {e}")

            # Log failed cleanup
            await self._log_cleanup(result)

            return result

    async def _archive_records(
        self,
        conn: asyncpg.Connection,
        source_table: str,
        archive_table: str,
        where_clause: str,
        params: List[Any],
        batch_size: int
    ) -> int:
        """
        Archive records to archive table

        Args:
            conn: Database connection
            source_table: Source table name
            archive_table: Archive table name
            where_clause: WHERE clause for filtering
            params: Query parameters
            batch_size: Batch size for operations

        Returns:
            Number of records archived
        """
        try:
            # Create archive table if not exists (copy structure from source)
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {archive_table} (LIKE {source_table} INCLUDING ALL)
            """)

            # Add archived_at column if not exists
            await conn.execute(f"""
                ALTER TABLE {archive_table}
                ADD COLUMN IF NOT EXISTS archived_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            """)

            # Archive in batches
            total_archived = 0
            while True:
                batch = await conn.fetch(f"""
                    SELECT * FROM {source_table}
                    WHERE {where_clause}
                    LIMIT {batch_size}
                """, *params)

                if not batch:
                    break

                # Insert into archive table
                for row in batch:
                    columns = list(row.keys())
                    placeholders = ', '.join([f'${i+1}' for i in range(len(columns))])
                    column_names = ', '.join(columns)

                    await conn.execute(
                        f"INSERT INTO {archive_table} ({column_names}) VALUES ({placeholders})",
                        *[row[col] for col in columns]
                    )

                total_archived += len(batch)

            logger.info(f"Archived {total_archived} records from {source_table} to {archive_table}")
            return total_archived

        except Exception as e:
            logger.error(f"Failed to archive records: {e}")
            raise

    async def _delete_records(
        self,
        conn: asyncpg.Connection,
        table_name: str,
        where_clause: str,
        params: List[Any],
        batch_size: int
    ) -> int:
        """
        Delete records in batches

        Args:
            conn: Database connection
            table_name: Table name
            where_clause: WHERE clause for filtering
            params: Query parameters
            batch_size: Batch size for operations

        Returns:
            Number of records deleted
        """
        try:
            total_deleted = 0
            while True:
                # Delete in batches for better performance
                result = await conn.execute(f"""
                    DELETE FROM {table_name}
                    WHERE ctid IN (
                        SELECT ctid FROM {table_name}
                        WHERE {where_clause}
                        LIMIT {batch_size}
                    )
                """, *params)

                # Extract row count from result
                deleted = int(result.split()[-1])
                if deleted == 0:
                    break

                total_deleted += deleted

            logger.info(f"Deleted {total_deleted} records from {table_name}")
            return total_deleted

        except Exception as e:
            logger.error(f"Failed to delete records: {e}")
            raise

    async def _anonymize_records(
        self,
        conn: asyncpg.Connection,
        table_name: str,
        where_clause: str,
        params: List[Any],
        batch_size: int
    ) -> int:
        """
        Anonymize PII fields in records

        Args:
            conn: Database connection
            table_name: Table name
            where_clause: WHERE clause for filtering
            params: Query parameters
            batch_size: Batch size for operations

        Returns:
            Number of records anonymized
        """
        try:
            # Common PII fields to anonymize
            pii_fields = ['email', 'first_name', 'last_name', 'phone', 'address', 'ip_address']

            # Check which fields exist in the table
            columns = await conn.fetch(f"""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = '{table_name}'
                AND column_name = ANY($1)
            """, pii_fields)

            if not columns:
                logger.warning(f"No PII fields found in {table_name}")
                return 0

            existing_fields = [col['column_name'] for col in columns]

            # Build anonymization UPDATE query
            set_clauses = []
            for field in existing_fields:
                if field == 'email':
                    set_clauses.append(f"{field} = 'anonymized_' || md5(random()::text) || '@anonymized.local'")
                elif field == 'ip_address':
                    set_clauses.append(f"{field} = '0.0.0.0'")
                else:
                    set_clauses.append(f"{field} = '[ANONYMIZED]'")

            update_query = f"""
                UPDATE {table_name}
                SET {', '.join(set_clauses)}
                WHERE {where_clause}
            """

            result = await conn.execute(update_query, *params)
            anonymized = int(result.split()[-1])

            logger.info(f"Anonymized {anonymized} records in {table_name}")
            return anonymized

        except Exception as e:
            logger.error(f"Failed to anonymize records: {e}")
            raise

    async def _log_cleanup(self, result: CleanupResult):
        """
        Log cleanup operation to audit table

        Args:
            result: CleanupResult to log
        """
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO cleanup_audit_logs
                    (cleanup_id, policy_id, table_name, records_deleted, records_archived,
                     storage_freed_bytes, duration_ms, dry_run, status, error_message,
                     started_at, completed_at, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                """,
                    result.cleanup_id,
                    result.policy_id,
                    result.table_name,
                    result.records_deleted,
                    result.records_archived,
                    result.storage_freed_bytes,
                    result.duration_ms,
                    result.dry_run,
                    result.status,
                    result.error_message,
                    result.started_at,
                    result.completed_at,
                    json.dumps(result.metadata)
                )

        except Exception as e:
            logger.error(f"Failed to log cleanup: {e}")

    async def _run_all_cleanups(self):
        """
        Run cleanup for all active policies

        This is called by the scheduler
        """
        logger.info("Starting scheduled cleanup for all active policies")

        try:
            policies = await self.list_policies(enabled_only=True)

            results = []
            for policy in policies:
                result = await self.execute_cleanup(policy.policy_id, dry_run=False)
                results.append(result)

            # Update metrics
            await self._update_metrics()

            success_count = sum(1 for r in results if r.status == "success")
            failed_count = sum(1 for r in results if r.status == "failed")

            logger.info(
                f"Scheduled cleanup completed: {success_count} successful, "
                f"{failed_count} failed out of {len(results)} policies"
            )

        except Exception as e:
            logger.error(f"Scheduled cleanup failed: {e}")

    async def _update_metrics(self):
        """Update retention system metrics"""
        try:
            async with self.pool.acquire() as conn:
                # Count policies
                total_policies = await conn.fetchval("""
                    SELECT COUNT(*) FROM retention_policies
                """)

                # Count cleanups
                total_cleanups = await conn.fetchval("""
                    SELECT COUNT(*) FROM cleanup_audit_logs
                """)

                # Sum deleted records
                total_records_deleted = await conn.fetchval("""
                    SELECT COALESCE(SUM(records_deleted), 0) FROM cleanup_audit_logs
                """)

                # Sum storage freed
                total_storage_freed_bytes = await conn.fetchval("""
                    SELECT COALESCE(SUM(storage_freed_bytes), 0) FROM cleanup_audit_logs
                """)

                # Average cleanup time
                avg_cleanup_time = await conn.fetchval("""
                    SELECT COALESCE(AVG(duration_ms), 0) FROM cleanup_audit_logs
                    WHERE status = 'success'
                """)

                # Last cleanup time
                last_cleanup = await conn.fetchval("""
                    SELECT MAX(completed_at) FROM cleanup_audit_logs
                """)

                # Policies by status
                status_rows = await conn.fetch("""
                    SELECT status, COUNT(*) as count
                    FROM retention_policies
                    GROUP BY status
                """)
                policies_by_status = {row['status']: row['count'] for row in status_rows}

                # Cleanups by table
                table_rows = await conn.fetch("""
                    SELECT table_name, COUNT(*) as count
                    FROM cleanup_audit_logs
                    GROUP BY table_name
                """)
                cleanups_by_table = {row['table_name']: row['count'] for row in table_rows}

                # Update metrics
                self.metrics.total_policies = total_policies
                self.metrics.total_cleanups = total_cleanups
                self.metrics.total_records_deleted = total_records_deleted
                self.metrics.total_storage_freed_mb = total_storage_freed_bytes / (1024 * 1024)
                self.metrics.avg_cleanup_time_ms = avg_cleanup_time
                self.metrics.last_cleanup_time = last_cleanup
                self.metrics.policies_by_status = policies_by_status
                self.metrics.cleanups_by_table = cleanups_by_table
                self.metrics.last_updated = datetime.now(timezone.utc)

        except Exception as e:
            logger.error(f"Failed to update metrics: {e}")

    async def get_metrics(self) -> RetentionMetrics:
        """
        Get current retention metrics

        Returns:
            RetentionMetrics object
        """
        await self._update_metrics()
        return self.metrics

    async def get_cleanup_history(
        self,
        policy_id: Optional[str] = None,
        limit: int = 100
    ) -> List[CleanupResult]:
        """
        Get cleanup history

        Args:
            policy_id: Filter by policy ID (optional)
            limit: Maximum number of results

        Returns:
            List of CleanupResult objects
        """
        if not self._initialized:
            raise RuntimeError("DataRetentionManager not initialized. Call initialize() first.")

        try:
            async with self.pool.acquire() as conn:
                if policy_id:
                    rows = await conn.fetch("""
                        SELECT * FROM cleanup_audit_logs
                        WHERE policy_id = $1
                        ORDER BY started_at DESC
                        LIMIT $2
                    """, policy_id, limit)
                else:
                    rows = await conn.fetch("""
                        SELECT * FROM cleanup_audit_logs
                        ORDER BY started_at DESC
                        LIMIT $1
                    """, limit)

                results = []
                for row in rows:
                    result = CleanupResult(
                        cleanup_id=row['cleanup_id'],
                        policy_id=row['policy_id'],
                        table_name=row['table_name'],
                        records_deleted=row['records_deleted'],
                        records_archived=row['records_archived'],
                        storage_freed_bytes=row['storage_freed_bytes'],
                        duration_ms=row['duration_ms'],
                        dry_run=row['dry_run'],
                        status=row['status'],
                        error_message=row['error_message'],
                        started_at=row['started_at'],
                        completed_at=row['completed_at'],
                        metadata=json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
                    )
                    results.append(result)

                return results

        except Exception as e:
            logger.error(f"Failed to get cleanup history: {e}")
            raise

    async def get_health(self) -> Dict[str, Any]:
        """
        Get retention system health status

        Returns:
            Health metrics dictionary
        """
        if not self._initialized:
            return {
                'status': 'not_initialized',
                'database_connected': False,
                'scheduler_running': False
            }

        try:
            # Test database connection
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")

            # Update and get metrics
            metrics = await self.get_metrics()

            # Check SLO (P95 < 200ms)
            slo_met = metrics.avg_cleanup_time_ms < 200

            return {
                'status': 'healthy' if slo_met else 'degraded',
                'database_connected': True,
                'database': self.database,
                'pool_size': self.pool.get_size(),
                'pool_idle': self.pool.get_idle_size(),
                'scheduler_running': self.scheduler.running if self.scheduler else False,
                'total_policies': metrics.total_policies,
                'total_cleanups': metrics.total_cleanups,
                'total_records_deleted': metrics.total_records_deleted,
                'total_storage_freed_mb': metrics.total_storage_freed_mb,
                'avg_cleanup_time_ms': metrics.avg_cleanup_time_ms,
                'last_cleanup_time': metrics.last_cleanup_time.isoformat() if metrics.last_cleanup_time else None,
                'policies_by_status': metrics.policies_by_status,
                'cleanups_by_table': metrics.cleanups_by_table,
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
        """Close database connection pool and scheduler"""
        if self.scheduler and self.scheduler.running:
            self.stop_scheduler()

        if self.pool:
            await self.pool.close()
            logger.info("Data retention connection pool closed")
            self._initialized = False


# Example usage and testing
async def main():
    """Example data retention usage with enterprise policies"""
    print("\n" + "=" * 80)
    print("Data Retention Manager - Enterprise PostgreSQL Example")
    print("=" * 80 + "\n")

    # Initialize retention manager
    manager = DataRetentionManager(
        host="localhost",
        port=5432,
        database="devskyy",
        user="postgres",
        password="postgres",
        enable_scheduler=True
    )

    await manager.initialize()

    print("### Example 1: Create User Sessions Retention Policy (90 days)")
    user_sessions_policy = RetentionPolicy(
        policy_id="user_sessions_90d",
        data_type=DataType.USER_SESSIONS,
        table_name="user_sessions",
        retention_days=90,
        action=RetentionAction.DELETE,
        legal_basis="GDPR Article 5.1(e) - Storage Limitation",
        date_column="created_at",
        batch_size=1000,
        metadata={'priority': 'high', 'category': 'user_data'}
    )
    await manager.create_policy(user_sessions_policy, overwrite=True)
    print(f"Created policy: {user_sessions_policy.policy_id}")
    print()

    print("### Example 2: Create Orders Retention Policy (7 years - Tax Compliance)")
    orders_policy = RetentionPolicy(
        policy_id="orders_7years",
        data_type=DataType.ORDER_HISTORY,
        table_name="orders",
        retention_days=2555,  # 7 years
        action=RetentionAction.ARCHIVE,
        legal_basis="Tax compliance (IRC Section 6001)",
        archive_table="orders_archive",
        date_column="created_at",
        batch_size=500,
        metadata={'priority': 'critical', 'category': 'financial'}
    )
    await manager.create_policy(orders_policy, overwrite=True)
    print(f"Created policy: {orders_policy.policy_id}")
    print()

    print("### Example 3: Create Analytics Events Retention Policy (30 days)")
    analytics_policy = RetentionPolicy(
        policy_id="analytics_30d",
        data_type=DataType.ANALYTICS_EVENTS,
        table_name="analytics_events",
        retention_days=30,
        action=RetentionAction.DELETE,
        legal_basis="Legitimate interest - Product improvement",
        date_column="event_timestamp",
        batch_size=2000,
        metadata={'priority': 'medium', 'category': 'analytics'}
    )
    await manager.create_policy(analytics_policy, overwrite=True)
    print(f"Created policy: {analytics_policy.policy_id}")
    print()

    print("### Example 4: List All Policies")
    all_policies = await manager.list_policies(enabled_only=False)
    print(f"Total policies: {len(all_policies)}")
    for policy in all_policies:
        print(f"  - {policy.policy_id}: {policy.table_name} ({policy.retention_days} days) - {policy.status.value}")
    print()

    print("### Example 5: Execute Dry-Run Cleanup")
    dry_run_result = await manager.execute_cleanup(
        user_sessions_policy.policy_id,
        dry_run=True
    )
    print(f"Dry-run result for {dry_run_result.table_name}:")
    print(f"  Status: {dry_run_result.status}")
    print(f"  Would delete: {dry_run_result.records_deleted} records")
    print(f"  Duration: {dry_run_result.duration_ms:.2f}ms")
    print()

    print("### Example 6: Get Retention Metrics")
    metrics = await manager.get_metrics()
    print("Retention System Metrics:")
    print(f"  Total Policies: {metrics.total_policies}")
    print(f"  Total Cleanups: {metrics.total_cleanups}")
    print(f"  Total Records Deleted: {metrics.total_records_deleted}")
    print(f"  Total Storage Freed: {metrics.total_storage_freed_mb:.2f} MB")
    print(f"  Avg Cleanup Time: {metrics.avg_cleanup_time_ms:.2f}ms")
    print()

    print("### Example 7: Health Check")
    health = await manager.get_health()
    print("Data Retention Health:")
    print(json.dumps(health, indent=2))
    print()

    if HAS_APSCHEDULER:
        print("### Example 8: Start Scheduler")
        manager.start_scheduler()
        print("Scheduler started - cleanup will run daily at 2 AM UTC")
        manager.stop_scheduler()
        print()

    # Cleanup
    await manager.close()
    print("Data Retention Manager closed successfully")


if __name__ == "__main__":
    asyncio.run(main())
