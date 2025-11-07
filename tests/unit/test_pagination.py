"""
Comprehensive Unit Tests for Pagination Module
Tests for pagination utilities and response formatting
"""

import pytest
from typing import List

from api.pagination import (
    PaginationParams,
    PaginatedResponse,
    create_paginated_response,
    CursorPaginationParams,
    CursorPaginatedResponse,
)


class TestPaginationParams:
    """Test suite for PaginationParams"""

    def test_pagination_params_defaults(self) -> None:
        """PaginationParams uses default values"""
        params = PaginationParams()

        assert params.page == 1
        assert params.page_size == 20

    def test_pagination_params_custom_values(self) -> None:
        """PaginationParams accepts custom values"""
        params = PaginationParams(page=2, page_size=50)

        assert params.page == 2
        assert params.page_size == 50

    def test_pagination_params_offset_calculation(self) -> None:
        """PaginationParams calculates offset correctly"""
        params = PaginationParams(page=1, page_size=20)

        assert params.offset == 0

    def test_pagination_params_offset_calculation_page_2(self) -> None:
        """PaginationParams calculates offset for page 2"""
        params = PaginationParams(page=2, page_size=20)

        assert params.offset == 20

    def test_pagination_params_offset_calculation_page_5(self) -> None:
        """PaginationParams calculates offset for page 5"""
        params = PaginationParams(page=5, page_size=50)

        assert params.offset == 200

    def test_pagination_params_limit_property(self) -> None:
        """PaginationParams limit is alias for page_size"""
        params = PaginationParams(page_size=25)

        assert params.limit == 25

    def test_pagination_params_min_page_validation(self) -> None:
        """PaginationParams page must be >= 1"""
        with pytest.raises(ValueError):
            PaginationParams(page=0)

    def test_pagination_params_max_page_size_validation(self) -> None:
        """PaginationParams page_size must be <= 100"""
        with pytest.raises(ValueError):
            PaginationParams(page_size=101)

    def test_pagination_params_min_page_size_validation(self) -> None:
        """PaginationParams page_size must be >= 1"""
        with pytest.raises(ValueError):
            PaginationParams(page_size=0)


class TestPaginatedResponse:
    """Test suite for PaginatedResponse"""

    def test_paginated_response_creation(self) -> None:
        """PaginatedResponse can be created with items"""
        items = ["item1", "item2"]
        response = PaginatedResponse(
            items=items,
            total=100,
            page=1,
            page_size=20,
            total_pages=5,
            has_next=True,
            has_previous=False,
        )

        assert response.items == items
        assert response.total == 100
        assert response.page == 1

    def test_paginated_response_with_generic_type(self) -> None:
        """PaginatedResponse works with different types"""
        response = PaginatedResponse[int](
            items=[1, 2, 3],
            total=100,
            page=1,
            page_size=20,
            total_pages=5,
            has_next=True,
            has_previous=False,
        )

        assert len(response.items) == 3

    def test_paginated_response_metadata(self) -> None:
        """PaginatedResponse includes all metadata"""
        response = PaginatedResponse(
            items=[],
            total=150,
            page=2,
            page_size=30,
            total_pages=5,
            has_next=True,
            has_previous=True,
        )

        assert response.total == 150
        assert response.page == 2
        assert response.page_size == 30
        assert response.total_pages == 5
        assert response.has_next is True
        assert response.has_previous is True


class TestCreatePaginatedResponse:
    """Test suite for create_paginated_response helper function"""

    def test_create_paginated_response_first_page(self) -> None:
        """create_paginated_response for first page"""
        items = ["a", "b", "c"]
        response = create_paginated_response(items, total=100, page=1, page_size=20)

        assert response.items == items
        assert response.total == 100
        assert response.page == 1
        assert response.has_next is True
        assert response.has_previous is False

    def test_create_paginated_response_middle_page(self) -> None:
        """create_paginated_response for middle page"""
        items = ["a", "b", "c"]
        response = create_paginated_response(items, total=100, page=3, page_size=20)

        assert response.page == 3
        assert response.has_next is True
        assert response.has_previous is True

    def test_create_paginated_response_last_page(self) -> None:
        """create_paginated_response for last page"""
        items = ["a", "b"]
        response = create_paginated_response(items, total=42, page=3, page_size=20)

        assert response.page == 3
        assert response.has_next is False
        assert response.has_previous is True

    def test_create_paginated_response_total_pages_calculation(self) -> None:
        """create_paginated_response calculates total_pages correctly"""
        response = create_paginated_response([], total=100, page=1, page_size=20)

        assert response.total_pages == 5

    def test_create_paginated_response_odd_total(self) -> None:
        """create_paginated_response handles odd total counts"""
        response = create_paginated_response([], total=105, page=1, page_size=20)

        assert response.total_pages == 6

    def test_create_paginated_response_empty_items(self) -> None:
        """create_paginated_response handles empty items"""
        response = create_paginated_response([], total=0, page=1, page_size=20)

        assert response.items == []
        assert response.total == 0
        assert response.total_pages == 0
        assert response.has_next is False

    def test_create_paginated_response_single_item(self) -> None:
        """create_paginated_response handles single item"""
        response = create_paginated_response(["item"], total=1, page=1, page_size=20)

        assert response.total_pages == 1
        assert response.has_next is False

    def test_create_paginated_response_exact_page_size(self) -> None:
        """create_paginated_response when total equals page_size"""
        response = create_paginated_response([], total=20, page=1, page_size=20)

        assert response.total_pages == 1
        assert response.has_next is False


class TestCursorPaginationParams:
    """Test suite for CursorPaginationParams"""

    def test_cursor_pagination_params_defaults(self) -> None:
        """CursorPaginationParams uses default values"""
        params = CursorPaginationParams()

        assert params.cursor is None
        assert params.limit == 20

    def test_cursor_pagination_params_with_cursor(self) -> None:
        """CursorPaginationParams accepts cursor"""
        cursor = "eyJpZCI6MTAwfQ=="
        params = CursorPaginationParams(cursor=cursor, limit=50)

        assert params.cursor == cursor
        assert params.limit == 50

    def test_cursor_pagination_params_max_limit(self) -> None:
        """CursorPaginationParams enforces max limit"""
        with pytest.raises(ValueError):
            CursorPaginationParams(limit=101)

    def test_cursor_pagination_params_min_limit(self) -> None:
        """CursorPaginationParams enforces min limit"""
        with pytest.raises(ValueError):
            CursorPaginationParams(limit=0)


class TestCursorPaginatedResponse:
    """Test suite for CursorPaginatedResponse"""

    def test_cursor_paginated_response_creation(self) -> None:
        """CursorPaginatedResponse can be created"""
        items = ["item1", "item2"]
        response = CursorPaginatedResponse(
            items=items,
            next_cursor="eyJpZCI6MTAwfQ==",
            has_more=True,
        )

        assert response.items == items
        assert response.next_cursor == "eyJpZCI6MTAwfQ=="
        assert response.has_more is True

    def test_cursor_paginated_response_no_more_items(self) -> None:
        """CursorPaginatedResponse handles no more items"""
        response = CursorPaginatedResponse(
            items=["item"],
            next_cursor=None,
            has_more=False,
        )

        assert response.next_cursor is None
        assert response.has_more is False

    def test_cursor_paginated_response_with_generic_type(self) -> None:
        """CursorPaginatedResponse works with different types"""
        response = CursorPaginatedResponse[dict](
            items=[{"id": 1}, {"id": 2}],
            next_cursor="cursor123",
            has_more=True,
        )

        assert len(response.items) == 2
        assert response.items[0]["id"] == 1

    def test_cursor_paginated_response_empty_items(self) -> None:
        """CursorPaginatedResponse handles empty items"""
        response = CursorPaginatedResponse(
            items=[],
            next_cursor=None,
            has_more=False,
        )

        assert response.items == []
        assert response.has_more is False


class TestPaginationIntegration:
    """Integration tests for pagination functionality"""

    def test_offset_pagination_workflow(self) -> None:
        """Complete offset-based pagination workflow"""
        # Simulate paginating through large dataset
        total_items = 105
        page_size = 20

        all_items = []

        for page in range(1, 10):
            params = PaginationParams(page=page, page_size=page_size)
            offset = params.offset

            # Simulate database fetch
            page_items = list(range(offset, min(offset + page_size, total_items)))

            response = create_paginated_response(
                items=page_items,
                total=total_items,
                page=page,
                page_size=page_size,
            )

            all_items.extend(response.items)

            if not response.has_next:
                break

        assert len(all_items) == 105

    def test_cursor_pagination_workflow(self) -> None:
        """Complete cursor-based pagination workflow"""
        items_db = list(range(1, 101))
        cursor = None
        all_fetched = []

        for _ in range(10):  # Max iterations
            params = CursorPaginationParams(cursor=cursor, limit=10)

            # Simulate cursor-based fetch
            if cursor:
                cursor_idx = int(cursor)
                page_items = items_db[cursor_idx: cursor_idx + params.limit]
            else:
                page_items = items_db[: params.limit]

            has_more = len(page_items) == params.limit
            next_cursor = str(len(all_fetched) + len(page_items)) if has_more else None

            response = CursorPaginatedResponse(
                items=page_items,
                next_cursor=next_cursor,
                has_more=has_more,
            )

            all_fetched.extend(response.items)

            if not response.has_more:
                break

            cursor = response.next_cursor

        assert len(all_fetched) == 100
