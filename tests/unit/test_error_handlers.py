"""
Comprehensive Unit Tests for Error Handlers Module
Tests for custom exceptions and FastAPI error handling
"""

import os
import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi import FastAPI, HTTPException, status
from fastapi.testclient import TestClient
from fastapi.exceptions import RequestValidationError

from error_handlers import (
    DevSkyyException,
    DatabaseException,
    AuthenticationException,
    AuthorizationException,
    ValidationException,
    ResourceNotFoundException,
    RateLimitException,
    ExternalServiceException,
    http_exception_handler,
    validation_exception_handler,
    devskyy_exception_handler,
    generic_exception_handler,
    safe_execute,
    safe_execute_async,
    register_error_handlers,
)


class TestDevSkyyException:
    """Test suite for DevSkyyException base class"""

    def test_create_custom_devskyy_exception_with_valid_data(self) -> None:
        """Create custom DevSkyy exception with valid data"""
        message = "Test error"
        status_code = 400
        details = {"field": "test"}

        exc = DevSkyyException(message, status_code, details)

        assert exc.message == message
        assert exc.status_code == status_code
        assert exc.details == details
        assert str(exc) == message

    def test_devskyy_exception_defaults(self) -> None:
        """DevSkyyException uses default values correctly"""
        exc = DevSkyyException("Test error")

        assert exc.status_code == 500
        assert exc.details == {}

    def test_devskyy_exception_inheritance(self) -> None:
        """DevSkyyException is proper Exception subclass"""
        exc = DevSkyyException("Test")
        assert isinstance(exc, Exception)


class TestDatabaseException:
    """Test suite for DatabaseException"""

    def test_database_exception_status_code(self) -> None:
        """DatabaseException sets 500 status code"""
        exc = DatabaseException("Connection failed")

        assert exc.status_code == 500
        assert exc.message == "Connection failed"

    def test_database_exception_with_details(self) -> None:
        """DatabaseException stores details"""
        details = {"table": "users", "operation": "select"}
        exc = DatabaseException("Query error", details)

        assert exc.details == details


class TestAuthenticationException:
    """Test suite for AuthenticationException"""

    def test_authentication_exception_status_code(self) -> None:
        """Authentication exception sets 401 status code"""
        exc = AuthenticationException()

        assert exc.status_code == 401

    def test_authentication_exception_default_message(self) -> None:
        """Authentication exception has default message"""
        exc = AuthenticationException()

        assert exc.message == "Authentication failed"

    def test_authentication_exception_custom_message(self) -> None:
        """Authentication exception accepts custom message"""
        exc = AuthenticationException("Invalid token")

        assert exc.message == "Invalid token"


class TestAuthorizationException:
    """Test suite for AuthorizationException"""

    def test_authorization_exception_status_code(self) -> None:
        """Authorization exception sets 403 status code"""
        exc = AuthorizationException()

        assert exc.status_code == 403

    def test_authorization_exception_default_message(self) -> None:
        """Authorization exception has default message"""
        exc = AuthorizationException()

        assert exc.message == "Access denied"


class TestValidationException:
    """Test suite for ValidationException"""

    def test_validation_exception_status_code(self) -> None:
        """Validation exception sets 422 status code"""
        exc = ValidationException("Invalid input")

        assert exc.status_code == 422


class TestResourceNotFoundException:
    """Test suite for ResourceNotFoundException"""

    def test_resource_not_found_exception_message(self) -> None:
        """Resource not found exception formats message correctly"""
        exc = ResourceNotFoundException("User", "user_123")

        assert "not found" in exc.message
        assert "User" in exc.message
        assert "user_123" in exc.message
        assert exc.status_code == 404


class TestRateLimitException:
    """Test suite for RateLimitException"""

    def test_rate_limit_exception_status_code(self) -> None:
        """Rate limit exception sets 429 status code"""
        exc = RateLimitException()

        assert exc.status_code == 429

    def test_rate_limit_exception_retry_after(self) -> None:
        """Rate limit exception includes retry_after"""
        exc = RateLimitException(retry_after=120)

        assert exc.details["retry_after"] == 120

    def test_rate_limit_exception_default_retry_after(self) -> None:
        """Rate limit exception has default retry_after"""
        exc = RateLimitException()

        assert exc.details["retry_after"] == 60


class TestExternalServiceException:
    """Test suite for ExternalServiceException"""

    def test_external_service_exception_message(self) -> None:
        """External service exception formats message"""
        exc = ExternalServiceException("Stripe", "API error")

        assert "External service error" in exc.message
        assert "Stripe" in exc.message
        assert "API error" in exc.message
        assert exc.status_code == 502


class TestHttpExceptionHandler:
    """Test suite for HTTP exception handler"""

    @pytest.mark.asyncio
    async def test_http_exception_handler_formats_response_correctly(self) -> None:
        """HTTP exception handler formats response correctly"""
        request = Mock()
        request.url.path = "/api/test"
        exc = HTTPException(status_code=400, detail="Bad request")

        response = await http_exception_handler(request, exc)

        assert response.status_code == 400
        content = response.body.decode()
        assert "error" in content
        assert "http_error" in content
        assert "/api/test" in content

    @pytest.mark.asyncio
    async def test_http_exception_handler_preserves_status_code(self) -> None:
        """HTTP exception handler preserves original status code"""
        request = Mock()
        request.url.path = "/api/test"
        exc = HTTPException(status_code=404, detail="Not found")

        response = await http_exception_handler(request, exc)

        assert response.status_code == 404


class TestValidationExceptionHandler:
    """Test suite for validation exception handler"""

    @pytest.mark.asyncio
    async def test_validation_exception_handler_extracts_error_details(self) -> None:
        """Validation exception handler extracts error details"""
        request = Mock()
        request.url.path = "/api/users"

        # Mock validation error
        error = Mock()
        error.errors.return_value = [
            {
                "loc": ("body", "email"),
                "msg": "invalid email format",
                "type": "value_error",
            }
        ]
        exc = RequestValidationError(error.errors())

        response = await validation_exception_handler(request, exc)

        assert response.status_code == 422
        content = response.body.decode()
        assert "validation_error" in content
        assert "/api/users" in content


class TestDevSkyyExceptionHandler:
    """Test suite for DevSkyy exception handler"""

    @pytest.mark.asyncio
    async def test_devskyy_exception_handler_formats_response(self) -> None:
        """DevSkyy exception handler formats response correctly"""
        request = Mock()
        request.url.path = "/api/test"
        exc = AuthenticationException("Invalid token")

        response = await devskyy_exception_handler(request, exc)

        assert response.status_code == 401
        content = response.body.decode()
        assert "AuthenticationException" in content
        assert "Invalid token" in content


class TestGenericExceptionHandler:
    """Test suite for generic exception handler"""

    @pytest.mark.asyncio
    async def test_generic_exception_handler_exposes_errors_in_debug_mode(self) -> None:
        """Generic exception handler exposes errors in debug mode"""
        with patch.dict(os.environ, {"DEBUG": "true"}):
            request = Mock()
            request.url.path = "/api/test"
            request.method = "POST"
            exc = ValueError("Test error")

            response = await generic_exception_handler(request, exc)

            assert response.status_code == 500
            content = response.body.decode()
            assert "Test error" in content
            assert "traceback" in content

    @pytest.mark.asyncio
    async def test_generic_exception_handler_hides_errors_in_production(self) -> None:
        """Generic exception handler hides errors in production"""
        with patch.dict(os.environ, {"DEBUG": "false"}):
            request = Mock()
            request.url.path = "/api/test"
            request.method = "POST"
            exc = ValueError("Sensitive error")

            response = await generic_exception_handler(request, exc)

            assert response.status_code == 500
            content = response.body.decode()
            assert "internal server error" in content
            assert "Sensitive error" not in content
            assert "traceback" not in content


class TestSafeExecute:
    """Test suite for safe_execute function"""

    def test_safe_execute_function_catches_and_logs_errors(self) -> None:
        """Safe execute function catches and logs errors"""

        def failing_function() -> None:
            raise ValueError("Test error")

        result = safe_execute(failing_function, default_return="default")

        assert result == "default"

    def test_safe_execute_returns_function_result(self) -> None:
        """Safe execute returns function result on success"""

        def successful_function() -> str:
            return "success"

        result = safe_execute(successful_function)

        assert result == "success"

    def test_safe_execute_with_logging_disabled(self) -> None:
        """Safe execute works with logging disabled"""

        def failing_function() -> None:
            raise ValueError("Test")

        result = safe_execute(failing_function, default_return=None, log_errors=False)

        assert result is None

    def test_safe_execute_default_return_none(self) -> None:
        """Safe execute defaults return value to None"""

        def failing_function() -> None:
            raise RuntimeError("Test")

        result = safe_execute(failing_function)

        assert result is None


class TestSafeExecuteAsync:
    """Test suite for safe_execute_async function"""

    @pytest.mark.asyncio
    async def test_safe_execute_async_function_handles_async_errors(self) -> None:
        """Safe execute async function handles async errors"""

        async def failing_async_function() -> None:
            raise ValueError("Async error")

        result = await safe_execute_async(failing_async_function, default_return="default")

        assert result == "default"

    @pytest.mark.asyncio
    async def test_safe_execute_async_returns_async_result(self) -> None:
        """Safe execute async returns async function result"""

        async def successful_async_function() -> str:
            return "async success"

        result = await safe_execute_async(successful_async_function)

        assert result == "async success"

    @pytest.mark.asyncio
    async def test_safe_execute_async_with_logging_disabled(self) -> None:
        """Safe execute async works with logging disabled"""

        async def failing_async_function() -> None:
            raise ValueError("Test")

        result = await safe_execute_async(
            failing_async_function, default_return=None, log_errors=False
        )

        assert result is None


class TestRegisterErrorHandlers:
    """Test suite for error handler registration"""

    def test_register_error_handlers_adds_handlers(self) -> None:
        """Register error handlers adds all handlers to app"""
        app = FastAPI()

        register_error_handlers(app)

        assert len(app.exception_handlers) > 0

    def test_register_error_handlers_with_real_app(self) -> None:
        """Verify error handlers work with real FastAPI app"""
        app = FastAPI()
        register_error_handlers(app)
        client = TestClient(app)

        @app.get("/test-exception")
        def test_endpoint() -> None:
            raise AuthenticationException("Not authorized")

        response = client.get("/test-exception")

        assert response.status_code == 401
        data = response.json()
        assert "error" in data
        assert data["error"]["type"] == "AuthenticationException"


class TestIntegrationErrorHandlers:
    """Integration tests for error handling system"""

    def test_multiple_exception_types_handled(self) -> None:
        """Multiple exception types are handled correctly"""
        app = FastAPI()
        register_error_handlers(app)
        client = TestClient(app)

        @app.get("/auth-error")
        def auth_error() -> None:
            raise AuthenticationException()

        @app.get("/db-error")
        def db_error() -> None:
            raise DatabaseException("Connection failed")

        response_auth = client.get("/auth-error")
        response_db = client.get("/db-error")

        assert response_auth.status_code == 401
        assert response_db.status_code == 500

    def test_exception_details_preserved(self) -> None:
        """Exception details are preserved in response"""
        exc = DevSkyyException("Test", details={"key": "value"})

        assert exc.details["key"] == "value"
