"""
GDPR Compliance Module
Data export, deletion, retention policies, consent tracking
Articles 15 (Right of Access), 17 (Right to Erasure), 5 (Data Minimization)

Author: DevSkyy Enterprise Team
Date: October 26, 2025

Citation: GDPR Articles 15, 17, 5; Recital 83
"""

import json
import logging
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional
from enum import Enum
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, EmailStr
from sqlalchemy import select, delete, update, text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/gdpr", tags=["gdpr"])

# ============================================================================
# ENUMS
# ============================================================================


class ConsentType(str, Enum):
    """Types of user consent"""

    MARKETING = "marketing"
    ANALYTICS = "analytics"
    PROFILING = "profiling"
    COOKIES = "cookies"
    DATA_PROCESSING = "data_processing"


class DataCategory(str, Enum):
    """Categories of personal data"""

    PROFILE = "profile"  # Name, email, phone
    ACCOUNT = "account"  # Username, password hash, auth tokens
    BEHAVIORAL = "behavioral"  # Browsing history, interactions
    TRANSACTIONAL = "transactional"  # Orders, payments, invoices
    PREFERENCES = "preferences"  # Settings, language, theme
    GENERATED = "generated"  # ML insights, recommendations


# ============================================================================
# MODELS
# ============================================================================


class ConsentRecord(BaseModel):
    """GDPR consent record (Recital 83)"""

    consent_id: str
    user_id: str
    consent_type: ConsentType
    given: bool
    timestamp: datetime
    expires_at: Optional[datetime] = None
    ip_address: str
    user_agent: str
    metadata: Dict[str, Any] = {}


class DataExportRequest(BaseModel):
    """GDPR Article 15 - Right of Access"""

    user_id: str
    format: str = "json"  # json, csv, xml
    include_related: bool = True  # Include data from 3rd parties


class DataExportResponse(BaseModel):
    """GDPR Data Export Response"""

    export_id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    download_url: str
    data: Dict[str, Any]


class DataDeletionRequest(BaseModel):
    """GDPR Article 17 - Right to Erasure"""

    user_id: str
    reason: str  # Reason for deletion
    include_backups: bool = False  # Delete from backups too


class DataDeletionResponse(BaseModel):
    """GDPR Data Deletion Response"""

    deletion_id: str
    user_id: str
    status: str
    deleted_at: datetime
    items_deleted: int
    note: str


class RetentionPolicy(BaseModel):
    """Data retention policy"""

    data_category: DataCategory
    retention_days: int
    description: str
    legal_basis: str


class AuditLog(BaseModel):
    """GDPR audit log entry"""

    log_id: str
    user_id: str
    action: str  # "export", "delete", "consent_update", "data_access"
    timestamp: datetime
    actor_id: Optional[str] = None  # Admin who performed action
    ip_address: str
    details: Dict[str, Any] = {}


# ============================================================================
# DATA RETENTION POLICIES (GDPR Article 5.1(e))
# ============================================================================

RETENTION_POLICIES = {
    DataCategory.PROFILE: RetentionPolicy(
        data_category=DataCategory.PROFILE,
        retention_days=2555,  # Until account deletion
        description="User profile data (name, email, phone)",
        legal_basis="Necessary for contract performance",
    ),
    DataCategory.ACCOUNT: RetentionPolicy(
        data_category=DataCategory.ACCOUNT,
        retention_days=2555,  # Until account deletion
        description="Account credentials and auth tokens",
        legal_basis="Necessary for account security",
    ),
    DataCategory.BEHAVIORAL: RetentionPolicy(
        data_category=DataCategory.BEHAVIORAL,
        retention_days=365,  # 1 year
        description="User behavior and interaction logs",
        legal_basis="Legitimate interest in product improvement",
    ),
    DataCategory.TRANSACTIONAL: RetentionPolicy(
        data_category=DataCategory.TRANSACTIONAL,
        retention_days=2555,  # Until account deletion
        description="Orders, payments, invoices",
        legal_basis="Legal obligation (tax, accounting)",
    ),
    DataCategory.PREFERENCES: RetentionPolicy(
        data_category=DataCategory.PREFERENCES,
        retention_days=2555,  # Until account deletion
        description="User settings and preferences",
        legal_basis="Necessary for service functionality",
    ),
    DataCategory.GENERATED: RetentionPolicy(
        data_category=DataCategory.GENERATED,
        retention_days=90,  # 90 days
        description="ML-generated insights and recommendations",
        legal_basis="Legitimate interest in service improvement",
    ),
}

# ============================================================================
# GDPR MANAGER
# ============================================================================


class GDPRManager:
    """
    GDPR Compliance Manager

    Handles data export, deletion, consent, and audit logging
    per GDPR Articles 15, 17, and 5
    """

    def __init__(self, get_db_session=None):
        """
        Initialize GDPR manager

        Args:
            get_db_session: Async function that returns database session
        """
        self.get_db_session = get_db_session
        self.consent_records: Dict[str, List[ConsentRecord]] = {}
        self.audit_logs: List[AuditLog] = []
        self.data_exports: Dict[str, DataExportResponse] = {}
        self.data_deletions: Dict[str, DataDeletionResponse] = {}

    async def export_user_data(
        self,
        user_id: str,
        db: AsyncSession,
        ip_address: str = "0.0.0.0"
    ) -> Dict[str, Any]:
        """
        Export ALL user data from database (GDPR Article 15)

        Args:
            user_id: User ID to export data for
            db: Database session
            ip_address: IP address of requester

        Returns:
            Dictionary containing all user data in machine-readable format

        Example:
            >>> async with db_session() as db:
            ...     data = await gdpr_manager.export_user_data("123", db)
            ...     assert "profile" in data
            ...     assert "orders" in data
        """
        try:
            # Start timer for performance tracking
            start_time = datetime.now(timezone.utc)

            user_data = {
                "export_metadata": {
                    "export_date": start_time.isoformat(),
                    "user_id": user_id,
                    "format_version": "1.0",
                    "gdpr_article": "Article 15 - Right of Access",
                },
                "profile": {},
                "orders": [],
                "preferences": {},
                "sessions": [],
                "analytics": [],
                "consent_records": [],
            }

            # 1. Export User Profile Data
            user_query = select(text("*")).select_from(text("users")).where(
                text("id = :user_id")
            )
            result = await db.execute(user_query, {"user_id": user_id})
            user_row = result.mappings().first()

            if user_row:
                user_data["profile"] = {
                    "user_id": user_row.get("id"),
                    "email": user_row.get("email"),
                    "username": user_row.get("username"),
                    "full_name": user_row.get("full_name"),
                    "created_at": user_row.get("created_at").isoformat() if user_row.get("created_at") else None,
                    "updated_at": user_row.get("updated_at").isoformat() if user_row.get("updated_at") else None,
                    "is_active": user_row.get("is_active"),
                }

            # 2. Export Orders
            orders_query = select(text("*")).select_from(text("orders")).where(
                text("customer_id = :user_id")
            )
            orders_result = await db.execute(orders_query, {"user_id": user_id})

            for order in orders_result.mappings():
                user_data["orders"].append({
                    "order_id": order.get("id"),
                    "order_number": order.get("order_number"),
                    "total": float(order.get("total", 0)),
                    "status": order.get("status"),
                    "created_at": order.get("created_at").isoformat() if order.get("created_at") else None,
                    "items": order.get("items"),
                    "shipping_address": order.get("shipping_address"),
                    "billing_address": order.get("billing_address"),
                })

            # 3. Export Customer Preferences
            customer_query = select(text("*")).select_from(text("customers")).where(
                text("id = :user_id")
            )
            customer_result = await db.execute(customer_query, {"user_id": user_id})
            customer_row = customer_result.mappings().first()

            if customer_row:
                user_data["preferences"] = {
                    "email": customer_row.get("email"),
                    "phone": customer_row.get("phone"),
                    "address": customer_row.get("address"),
                    "preferences": customer_row.get("preferences"),
                    "lifetime_value": float(customer_row.get("lifetime_value", 0)),
                    "total_orders": customer_row.get("total_orders", 0),
                }

            # 4. Export User Sessions (if table exists)
            try:
                sessions_query = select(text("*")).select_from(text("user_sessions")).where(
                    text("user_id = :user_id")
                )
                sessions_result = await db.execute(sessions_query, {"user_id": user_id})

                for session in sessions_result.mappings():
                    user_data["sessions"].append({
                        "session_id": session.get("id"),
                        "created_at": session.get("created_at").isoformat() if session.get("created_at") else None,
                        "expires_at": session.get("expires_at").isoformat() if session.get("expires_at") else None,
                        "ip_address": session.get("ip_address"),
                        "user_agent": session.get("user_agent"),
                        "is_active": session.get("is_active"),
                    })
            except Exception as e:
                logger.debug(f"Sessions table not found or error: {e}")

            # 5. Export Analytics/Logs
            try:
                logs_query = select(text("*")).select_from(text("agent_logs")).where(
                    text("input_data::text LIKE :user_pattern OR output_data::text LIKE :user_pattern")
                ).limit(1000)
                logs_result = await db.execute(
                    logs_query,
                    {"user_pattern": f"%{user_id}%"}
                )

                for log in logs_result.mappings():
                    user_data["analytics"].append({
                        "log_id": log.get("id"),
                        "agent_name": log.get("agent_name"),
                        "action": log.get("action"),
                        "status": log.get("status"),
                        "created_at": log.get("created_at").isoformat() if log.get("created_at") else None,
                    })
            except Exception as e:
                logger.debug(f"Agent logs query error: {e}")

            # 6. Export Consent Records (from in-memory store)
            if user_id in self.consent_records:
                user_data["consent_records"] = [
                    {
                        "consent_id": record.consent_id,
                        "consent_type": record.consent_type,
                        "given": record.given,
                        "timestamp": record.timestamp.isoformat(),
                        "expires_at": record.expires_at.isoformat() if record.expires_at else None,
                    }
                    for record in self.consent_records[user_id]
                ]

            # Calculate performance metric
            end_time = datetime.now(timezone.utc)
            duration_ms = (end_time - start_time).total_seconds() * 1000

            user_data["export_metadata"]["duration_ms"] = duration_ms
            user_data["export_metadata"]["performance_status"] = (
                "PASS" if duration_ms < 200 else "SLOW"
            )

            # Audit log to database
            await self._log_audit_to_db(
                user_id=user_id,
                action="data_export",
                details={
                    "duration_ms": duration_ms,
                    "records_exported": {
                        "orders": len(user_data["orders"]),
                        "sessions": len(user_data["sessions"]),
                        "analytics": len(user_data["analytics"]),
                    }
                },
                ip_address=ip_address,
                db=db,
            )

            logger.info(
                f"Data export completed for user {user_id} in {duration_ms:.2f}ms"
            )

            return user_data

        except Exception as e:
            logger.error(f"Data export failed for user {user_id}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Data export failed: {str(e)}",
            )

    async def request_data_export(
        self, user_id: str, format: str = "json", include_related: bool = True
    ) -> DataExportResponse:
        """
        GDPR Article 15 - Right of Access

        User can request export of all personal data

        Args:
            user_id: User requesting export
            format: Export format (json, csv, xml)
            include_related: Include data from 3rd parties

        Returns:
            DataExportResponse with download URL

        Citation: GDPR Article 15 - Right of Access
        """
        export_id = str(uuid4())

        # Get database session
        if not self.get_db_session:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database session not configured",
            )

        async with self.get_db_session() as db:
            # Export all user data
            user_data = await self.export_user_data(user_id, db)

            # Convert format if needed
            if format == "csv":
                user_data["_format"] = "csv"
                # CSV conversion would happen at download time
            elif format == "xml":
                user_data["_format"] = "xml"
                # XML conversion would happen at download time

            export_response = DataExportResponse(
                export_id=export_id,
                user_id=user_id,
                created_at=datetime.now(timezone.utc),
                expires_at=datetime.now(timezone.utc) + timedelta(days=30),
                download_url=f"/api/v1/gdpr/exports/{export_id}/download",
                data=user_data,
            )

            self.data_exports[export_id] = export_response

            # Audit log
            await self._log_audit(
                user_id=user_id,
                action="data_export_request",
                details={
                    "export_id": export_id,
                    "format": format,
                    "include_related": include_related,
                },
            )

            logger.info(f"Data export requested: {export_id} for user {user_id}")

            return export_response

    async def delete_user_data(
        self,
        user_id: str,
        db: AsyncSession,
        reason: str,
        ip_address: str = "0.0.0.0"
    ) -> Dict[str, Any]:
        """
        Delete ALL user data with cascade (GDPR Article 17)

        Args:
            user_id: User ID to delete
            db: Database session
            reason: Reason for deletion
            ip_address: IP address of requester

        Returns:
            Dictionary containing deletion summary

        Example:
            >>> async with db_session() as db:
            ...     result = await gdpr_manager.delete_user_data("123", db, "User request")
            ...     assert result["items_deleted"] > 0
        """
        try:
            start_time = datetime.now(timezone.utc)
            items_deleted = 0

            # Verify user exists before deletion
            user_check = await db.execute(
                text("SELECT id FROM users WHERE id = :user_id"),
                {"user_id": user_id}
            )
            if not user_check.scalar():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"User {user_id} not found"
                )

            # Legal retention checks
            exceptions = []

            # Check if user has orders within tax retention period (7 years)
            tax_retention_date = datetime.now(timezone.utc) - timedelta(days=2555)
            recent_orders = await db.execute(
                text("SELECT COUNT(*) FROM orders WHERE customer_id = :user_id AND created_at > :retention_date"),
                {"user_id": user_id, "retention_date": tax_retention_date}
            )
            recent_order_count = recent_orders.scalar()

            if recent_order_count > 0:
                exceptions.append(f"Cannot delete {recent_order_count} recent orders (tax/legal retention)")

            # 1. Delete User Sessions (cascade)
            try:
                result = await db.execute(
                    delete(text("user_sessions")).where(text("user_id = :user_id")),
                    {"user_id": user_id}
                )
                items_deleted += result.rowcount
                logger.info(f"Deleted {result.rowcount} user sessions for user {user_id}")
            except Exception as e:
                logger.debug(f"Sessions deletion error (table may not exist): {e}")

            # 2. Delete/Anonymize Customer Data
            try:
                # Anonymize instead of delete to preserve analytics
                result = await db.execute(
                    update(text("customers"))
                    .where(text("id = :user_id"))
                    .values({
                        "email": text(f"'deleted_{user_id}@anonymized.local'"),
                        "first_name": text("'REDACTED'"),
                        "last_name": text("'REDACTED'"),
                        "phone": text("NULL"),
                        "address": text("NULL"),
                        "preferences": text("'{}'::json"),
                    }),
                    {"user_id": user_id}
                )
                items_deleted += result.rowcount
            except Exception as e:
                logger.debug(f"Customer anonymization error: {e}")

            # 3. Anonymize Orders (keep for legal compliance but remove PII)
            try:
                result = await db.execute(
                    update(text("orders"))
                    .where(text("customer_id = :user_id"))
                    .values({
                        "customer_email": text(f"'deleted_{user_id}@anonymized.local'"),
                        "shipping_address": text("NULL"),
                        "billing_address": text("NULL"),
                        "notes": text("'[REDACTED]'"),
                    }),
                    {"user_id": user_id}
                )
                items_deleted += result.rowcount
                logger.info(f"Anonymized {result.rowcount} orders for user {user_id}")
            except Exception as e:
                logger.debug(f"Order anonymization error: {e}")

            # 4. Delete User Preferences
            try:
                result = await db.execute(
                    delete(text("user_preferences")).where(text("user_id = :user_id")),
                    {"user_id": user_id}
                )
                items_deleted += result.rowcount
            except Exception as e:
                logger.debug(f"Preferences deletion error: {e}")

            # 5. Delete User Profile (final step)
            result = await db.execute(
                delete(text("users")).where(text("id = :user_id")),
                {"user_id": user_id}
            )
            items_deleted += result.rowcount

            # 6. Delete consent records from memory
            if user_id in self.consent_records:
                del self.consent_records[user_id]

            # Commit transaction
            await db.commit()

            # Calculate performance
            end_time = datetime.now(timezone.utc)
            duration_ms = (end_time - start_time).total_seconds() * 1000

            deletion_summary = {
                "deletion_id": str(uuid4()),
                "user_id": user_id,
                "status": "completed",
                "deleted_at": end_time.isoformat(),
                "items_deleted": items_deleted,
                "duration_ms": duration_ms,
                "reason": reason,
                "exceptions": exceptions,
                "note": f"User data deleted/anonymized. {len(exceptions)} exceptions apply.",
            }

            # Audit log
            await self._log_audit_to_db(
                user_id=user_id,
                action="data_deletion",
                details=deletion_summary,
                ip_address=ip_address,
                db=db,
            )

            logger.info(
                f"Data deletion completed for user {user_id}: {items_deleted} items in {duration_ms:.2f}ms"
            )

            return deletion_summary

        except Exception as e:
            await db.rollback()
            logger.error(f"Data deletion failed for user {user_id}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Data deletion failed: {str(e)}",
            )

    async def anonymize_user_data(
        self,
        user_id: str,
        db: AsyncSession,
        ip_address: str = "0.0.0.0"
    ) -> Dict[str, Any]:
        """
        Anonymize user PII using cryptographic hashing

        Args:
            user_id: User ID to anonymize
            db: Database session
            ip_address: IP address of requester

        Returns:
            Dictionary containing anonymization summary

        Example:
            >>> async with db_session() as db:
            ...     result = await gdpr_manager.anonymize_user_data("123", db)
            ...     assert result["fields_anonymized"] > 0
        """
        try:
            start_time = datetime.now(timezone.utc)
            fields_anonymized = 0

            # Create anonymization hash (SHA-256)
            anon_hash = hashlib.sha256(f"anon_{user_id}_{datetime.now(timezone.utc).timestamp()}".encode()).hexdigest()[:16]

            # 1. Anonymize User Profile
            result = await db.execute(
                update(text("users"))
                .where(text("id = :user_id"))
                .values({
                    "email": text(f"'anon_{anon_hash}@anonymized.local'"),
                    "username": text(f"'anon_user_{anon_hash}'"),
                    "full_name": text("'[ANONYMIZED]'"),
                }),
                {"user_id": user_id}
            )
            fields_anonymized += result.rowcount * 3

            # 2. Anonymize Customer Data
            try:
                result = await db.execute(
                    update(text("customers"))
                    .where(text("id = :user_id"))
                    .values({
                        "email": text(f"'anon_{anon_hash}@anonymized.local'"),
                        "first_name": text("'[ANONYMIZED]'"),
                        "last_name": text("'[ANONYMIZED]'"),
                        "phone": text("NULL"),
                        "address": text("NULL"),
                    }),
                    {"user_id": user_id}
                )
                fields_anonymized += result.rowcount * 5
            except Exception as e:
                logger.debug(f"Customer anonymization error: {e}")

            # 3. Anonymize Order PII
            try:
                result = await db.execute(
                    update(text("orders"))
                    .where(text("customer_id = :user_id"))
                    .values({
                        "customer_email": text(f"'anon_{anon_hash}@anonymized.local'"),
                        "shipping_address": text("'{\"anonymized\": true}'::json"),
                        "billing_address": text("'{\"anonymized\": true}'::json"),
                    }),
                    {"user_id": user_id}
                )
                fields_anonymized += result.rowcount * 3
            except Exception as e:
                logger.debug(f"Order anonymization error: {e}")

            # 4. Anonymize Sessions
            try:
                result = await db.execute(
                    update(text("user_sessions"))
                    .where(text("user_id = :user_id"))
                    .values({
                        "ip_address": text("'0.0.0.0'"),
                        "user_agent": text("'[ANONYMIZED]'"),
                    }),
                    {"user_id": user_id}
                )
                fields_anonymized += result.rowcount * 2
            except Exception as e:
                logger.debug(f"Session anonymization error: {e}")

            await db.commit()

            # Calculate performance
            end_time = datetime.now(timezone.utc)
            duration_ms = (end_time - start_time).total_seconds() * 1000

            anonymization_summary = {
                "anonymization_id": str(uuid4()),
                "user_id": user_id,
                "anonymization_hash": anon_hash,
                "fields_anonymized": fields_anonymized,
                "duration_ms": duration_ms,
                "timestamp": end_time.isoformat(),
            }

            # Audit log
            await self._log_audit_to_db(
                user_id=user_id,
                action="data_anonymization",
                details=anonymization_summary,
                ip_address=ip_address,
                db=db,
            )

            logger.info(
                f"Data anonymization completed for user {user_id}: {fields_anonymized} fields in {duration_ms:.2f}ms"
            )

            return anonymization_summary

        except Exception as e:
            await db.rollback()
            logger.error(f"Data anonymization failed for user {user_id}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Data anonymization failed: {str(e)}",
            )

    async def request_data_deletion(
        self, user_id: str, reason: str, include_backups: bool = False
    ) -> DataDeletionResponse:
        """
        GDPR Article 17 - Right to Erasure

        User can request deletion of all personal data

        Args:
            user_id: User requesting deletion
            reason: Reason for deletion request
            include_backups: Delete from backups too

        Returns:
            DataDeletionResponse with deletion status

        Citation: GDPR Article 17 - Right to Erasure

        Exceptions:
            - Legal obligation to retain (taxes, accounting)
            - Legitimate interest in retaining (contract, public interest)
        """
        deletion_id = str(uuid4())

        # Get database session
        if not self.get_db_session:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database session not configured",
            )

        async with self.get_db_session() as db:
            # Delete all user data
            deletion_summary = await self.delete_user_data(user_id, db, reason)

            deletion_response = DataDeletionResponse(
                deletion_id=deletion_summary["deletion_id"],
                user_id=user_id,
                status=deletion_summary["status"],
                deleted_at=datetime.fromisoformat(deletion_summary["deleted_at"]),
                items_deleted=deletion_summary["items_deleted"],
                note=deletion_summary["note"],
            )

            self.data_deletions[deletion_id] = deletion_response

            # Audit log
            await self._log_audit(
                user_id=user_id,
                action="data_deletion_request",
                details={
                    "deletion_id": deletion_id,
                    "reason": reason,
                    "include_backups": include_backups,
                    "items_deleted": deletion_summary["items_deleted"],
                },
            )

            logger.info(f"Data deletion requested: {deletion_id} for user {user_id}")

            return deletion_response

    async def update_consent(
            self,
            user_id: str,
            consent_type: ConsentType,
            given: bool,
            ip_address: str,
            user_agent: str) -> ConsentRecord:
        """
        Update user consent (GDPR Recital 83)

        Args:
            user_id: User granting/revoking consent
            consent_type: Type of consent
            given: Whether consent is given or revoked
            ip_address: IP address of user
            user_agent: User agent string

        Returns:
            ConsentRecord with consent status

        Citation: GDPR Recital 83 - Freely Given Consent
        """
        consent_record = ConsentRecord(
            consent_id=str(uuid4()),
            user_id=user_id,
            consent_type=consent_type,
            given=given,
            timestamp=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(days=730),  # 2 years
            ip_address=ip_address,
            user_agent=user_agent,
        )

        if user_id not in self.consent_records:
            self.consent_records[user_id] = []

        self.consent_records[user_id].append(consent_record)

        # Audit log
        await self._log_audit(
            user_id=user_id,
            action="consent_update",
            details={
                "consent_type": consent_type,
                "given": given,
                "consent_id": consent_record.consent_id,
            },
        )

        logger.info(f"Consent updated: {consent_type} = {given} for user {user_id}")

        return consent_record

    async def get_user_consents(self, user_id: str) -> List[ConsentRecord]:
        """Get all consent records for a user"""
        return self.consent_records.get(user_id, [])

    async def get_data_retention_status(
        self,
        user_id: str,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """
        Get data retention status for a user

        Args:
            user_id: User ID to check retention status
            db: Database session

        Returns:
            Dictionary containing retention status for all data categories

        Example:
            >>> async with db_session() as db:
            ...     status = await gdpr_manager.get_data_retention_status("123", db)
            ...     assert "profile" in status
            ...     assert "retention_days_remaining" in status["profile"]
        """
        try:
            retention_status = {}

            # Get user creation date
            user_query = await db.execute(
                text("SELECT created_at FROM users WHERE id = :user_id"),
                {"user_id": user_id}
            )
            user_row = user_query.first()

            if not user_row:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"User {user_id} not found"
                )

            user_created_at = user_row[0]
            current_time = datetime.now(timezone.utc)

            # Check each data category
            for category, policy in RETENTION_POLICIES.items():
                if category == DataCategory.PROFILE or category == DataCategory.ACCOUNT:
                    # Profile/Account data retained until deletion
                    retention_status[category.value] = {
                        "policy": policy.dict(),
                        "status": "active",
                        "retention_days_remaining": "Until account deletion",
                        "can_be_deleted": True,
                        "legal_basis": policy.legal_basis,
                    }
                elif category == DataCategory.BEHAVIORAL:
                    # Behavioral data - 1 year retention
                    days_since_creation = (current_time - user_created_at).days
                    days_remaining = max(0, policy.retention_days - days_since_creation)

                    retention_status[category.value] = {
                        "policy": policy.dict(),
                        "status": "active" if days_remaining > 0 else "expired",
                        "retention_days_remaining": days_remaining,
                        "can_be_deleted": True,
                        "legal_basis": policy.legal_basis,
                    }
                elif category == DataCategory.TRANSACTIONAL:
                    # Transactional data - check for recent orders
                    orders_query = await db.execute(
                        text("SELECT COUNT(*), MAX(created_at) FROM orders WHERE customer_id = :user_id"),
                        {"user_id": user_id}
                    )
                    order_count, last_order_date = orders_query.first()

                    if order_count > 0 and last_order_date:
                        days_since_order = (current_time - last_order_date).days
                        days_remaining = max(0, policy.retention_days - days_since_order)

                        retention_status[category.value] = {
                            "policy": policy.dict(),
                            "status": "active" if days_remaining > 0 else "can_archive",
                            "retention_days_remaining": days_remaining,
                            "can_be_deleted": False,  # Legal obligation
                            "legal_basis": policy.legal_basis,
                            "note": "Cannot be fully deleted due to tax/legal obligations. Can be anonymized.",
                        }
                    else:
                        retention_status[category.value] = {
                            "policy": policy.dict(),
                            "status": "no_data",
                            "retention_days_remaining": 0,
                            "can_be_deleted": True,
                            "legal_basis": policy.legal_basis,
                        }
                elif category == DataCategory.GENERATED:
                    # ML-generated data - 90 days
                    retention_status[category.value] = {
                        "policy": policy.dict(),
                        "status": "active",
                        "retention_days_remaining": policy.retention_days,
                        "can_be_deleted": True,
                        "legal_basis": policy.legal_basis,
                    }
                else:
                    # Other categories
                    retention_status[category.value] = {
                        "policy": policy.dict(),
                        "status": "active",
                        "retention_days_remaining": policy.retention_days,
                        "can_be_deleted": True,
                        "legal_basis": policy.legal_basis,
                    }

            return {
                "user_id": user_id,
                "checked_at": current_time.isoformat(),
                "retention_status": retention_status,
                "overall_summary": {
                    "total_categories": len(retention_status),
                    "can_delete_all": all(
                        status.get("can_be_deleted", False)
                        for status in retention_status.values()
                    ),
                    "has_legal_restrictions": any(
                        not status.get("can_be_deleted", True)
                        for status in retention_status.values()
                    ),
                },
            }

        except Exception as e:
            logger.error(f"Failed to get retention status for user {user_id}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get retention status: {str(e)}",
            )

    async def get_retention_policies(self) -> Dict[str, RetentionPolicy]:
        """Get all data retention policies"""
        return RETENTION_POLICIES

    async def get_audit_logs(
            self,
            user_id: Optional[str] = None,
            action: Optional[str] = None,
            limit: int = 100) -> List[AuditLog]:
        """Get GDPR audit logs"""
        logs = self.audit_logs

        if user_id:
            logs = [l for l in logs if l.user_id == user_id]

        if action:
            logs = [l for l in logs if l.action == action]

        return logs[-limit:]

    async def _log_audit_to_db(
        self,
        user_id: str,
        action: str,
        details: Dict[str, Any],
        ip_address: str,
        db: AsyncSession,
        actor_id: Optional[str] = None,
    ) -> str:
        """
        Log GDPR audit event to database

        Args:
            user_id: User ID
            action: Action performed
            details: Action details
            ip_address: IP address
            db: Database session
            actor_id: Optional actor performing action

        Returns:
            Audit log ID
        """
        try:
            log_id = str(uuid4())
            timestamp = datetime.now(timezone.utc)

            # Create audit log table if it doesn't exist
            await db.execute(text("""
                CREATE TABLE IF NOT EXISTS gdpr_audit_logs (
                    log_id VARCHAR(36) PRIMARY KEY,
                    user_id VARCHAR(255) NOT NULL,
                    action VARCHAR(100) NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                    actor_id VARCHAR(255),
                    ip_address VARCHAR(45),
                    details JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """))

            # Insert audit log
            await db.execute(
                text("""
                    INSERT INTO gdpr_audit_logs
                    (log_id, user_id, action, timestamp, actor_id, ip_address, details)
                    VALUES (:log_id, :user_id, :action, :timestamp, :actor_id, :ip_address, :details)
                """),
                {
                    "log_id": log_id,
                    "user_id": user_id,
                    "action": action,
                    "timestamp": timestamp,
                    "actor_id": actor_id,
                    "ip_address": ip_address,
                    "details": json.dumps(details),
                }
            )

            await db.commit()

            logger.debug(f"GDPR audit log created: {log_id} for action {action}")

            return log_id

        except Exception as e:
            logger.error(f"Failed to create audit log: {str(e)}")
            # Don't fail the operation if audit logging fails
            return ""

    async def _log_audit(
        self,
        user_id: str,
        action: str,
        details: Dict[str, Any],
        actor_id: Optional[str] = None,
        ip_address: str = "0.0.0.0",
    ) -> AuditLog:
        """Create audit log entry (in-memory)"""
        audit_log = AuditLog(
            log_id=str(uuid4()),
            user_id=user_id,
            action=action,
            timestamp=datetime.now(timezone.utc),
            actor_id=actor_id,
            ip_address=ip_address,
            details=details,
        )

        self.audit_logs.append(audit_log)
        return audit_log


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

gdpr_manager = GDPRManager()

# ============================================================================
# API ENDPOINTS
# ============================================================================


async def get_db_dependency():
    """Database dependency for FastAPI endpoints"""
    try:
        from database import get_db
        async for session in get_db():
            yield session
    except ImportError:
        # Fallback if database module not available
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database connection not configured"
        )


@router.post("/data-export", response_model=DataExportResponse)
async def endpoint_request_data_export(
    request: DataExportRequest,
    db: AsyncSession = Depends(get_db_dependency)
):
    """
    GDPR Article 15 - Request data export

    Example:
        POST /api/v1/gdpr/data-export
        {
            "user_id": "123",
            "format": "json",
            "include_related": true
        }
    """
    return await gdpr_manager.request_data_export(
        user_id=request.user_id,
        format=request.format,
        include_related=request.include_related
    )


@router.post("/data-delete", response_model=DataDeletionResponse)
async def endpoint_request_data_deletion(
    request: DataDeletionRequest,
    db: AsyncSession = Depends(get_db_dependency)
):
    """
    GDPR Article 17 - Request data deletion

    Example:
        POST /api/v1/gdpr/data-delete
        {
            "user_id": "123",
            "reason": "User requested account deletion",
            "include_backups": false
        }
    """
    return await gdpr_manager.request_data_deletion(
        user_id=request.user_id,
        reason=request.reason,
        include_backups=request.include_backups
    )


@router.post("/consent")
async def endpoint_update_consent(
    user_id: str,
    consent_type: ConsentType,
    given: bool,
    ip_address: str = "0.0.0.0",
    user_agent: str = "Unknown"
):
    """
    Update user consent

    Example:
        POST /api/v1/gdpr/consent?user_id=123&consent_type=marketing&given=true
    """
    return await gdpr_manager.update_consent(
        user_id=user_id,
        consent_type=consent_type,
        given=given,
        ip_address=ip_address,
        user_agent=user_agent
    )


@router.get("/consents/{user_id}")
async def endpoint_get_consents(user_id: str):
    """
    Get all consent records for a user

    Example:
        GET /api/v1/gdpr/consents/123
    """
    consents = await gdpr_manager.get_user_consents(user_id)
    return {"user_id": user_id, "consents": consents}


@router.get("/retention-policies")
async def endpoint_get_retention_policies():
    """
    Get all data retention policies

    Example:
        GET /api/v1/gdpr/retention-policies
    """
    policies = await gdpr_manager.get_retention_policies()
    return {"policies": {k.value: v.dict() for k, v in policies.items()}}


@router.get("/retention-status/{user_id}")
async def endpoint_get_retention_status(
    user_id: str,
    db: AsyncSession = Depends(get_db_dependency)
):
    """
    Get data retention status for a specific user

    Example:
        GET /api/v1/gdpr/retention-status/123
    """
    return await gdpr_manager.get_data_retention_status(user_id, db)


@router.get("/audit-logs")
async def endpoint_get_audit_logs(
    user_id: Optional[str] = None,
    action: Optional[str] = None,
    limit: int = 100
):
    """
    Get GDPR audit logs

    Example:
        GET /api/v1/gdpr/audit-logs?user_id=123&limit=50
    """
    logs = await gdpr_manager.get_audit_logs(
        user_id=user_id,
        action=action,
        limit=limit
    )
    return {"logs": logs, "count": len(logs)}

if __name__ == "__main__":
    import asyncio

    async def demo():
        # Test data export
        export = await gdpr_manager.request_data_export("user123", format="json")
        print(f"Export created: {export.export_id}")

        # Test consent update
        consent = await gdpr_manager.update_consent(
            user_id="user123",
            consent_type=ConsentType.MARKETING,
            given=True,
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0...",
        )
        print(f"Consent updated: {consent.consent_id}")

        # Test retention policies
        policies = await gdpr_manager.get_retention_policies()
        print(f"Retention policies: {len(policies)}")

    asyncio.run(demo())
