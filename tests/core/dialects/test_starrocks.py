"""StarRocks dialect specifics.

StarRocks speaks the MySQL wire protocol but is not MySQL: its parser resolves
``"x"`` as an identifier, its database and table names are case-sensitive, and
its statement cap is a session variable. These pin the choices the dialect
makes, each one against what the server was observed to do.
"""

from __future__ import annotations

from typing import cast

import pytest

from dbt_charts.core.dialects import get_dialect
from dbt_charts.core.dialects.starrocks import StarRocksDialect


@pytest.fixture
def dialect() -> StarRocksDialect:
    """The registered instance — the one every lookup returns."""
    return cast(StarRocksDialect, get_dialect("starrocks"))


class TestParameters:
    def test_param_is_the_mysql_positional_placeholder(
        self, dialect: StarRocksDialect
    ) -> None:
        """mysql-connector-python binds positionally, with no index in the placeholder."""
        assert dialect.param(1) == "%s"
        assert dialect.param(7) == "%s"
        assert dialect.params(3) == ["%s", "%s", "%s"]

    def test_the_placeholder_is_not_a_named_one(
        self, dialect: StarRocksDialect
    ) -> None:
        """A named-param dialect would be handed a dict the MySQL driver cannot bind."""
        assert dialect.uses_named_params is False


class TestStringLiterals:
    def test_backslashes_are_escaped(self, dialect: StarRocksDialect) -> None:
        """StarRocks' '...' constant takes backslash escapes, so a backslash in a
        value has to be doubled or it escapes the character after it."""
        assert dialect.escapes_backslashes is True
        assert dialect.string_literal("a\\b") == "'a\\\\b'"

    def test_an_embedded_quote_is_backslash_escaped(
        self, dialect: StarRocksDialect
    ) -> None:
        assert dialect.string_literal("it's") == r"'it\'s'"


class TestQuoting:
    def test_identifiers_are_backtick_quoted(self, dialect: StarRocksDialect) -> None:
        """Backticks are what dbt-starrocks itself emits, so a relation is not
        spelled two ways for one warehouse."""
        assert dialect.quote("orders") == "`orders`"

    def test_an_embedded_backtick_is_doubled(self, dialect: StarRocksDialect) -> None:
        assert dialect.quote("we`ird") == "`we``ird`"


class TestIdentifierCase:
    def test_database_and_table_names_are_case_sensitive(
        self, dialect: StarRocksDialect
    ) -> None:
        """StarRocks resolves `GAMING_OPS` and `gaming_ops` to different
        databases, so a schema filter must compare them as written."""
        assert dialect.folds_identifier_case is False

    def test_a_schema_filter_is_verbatim(self, dialect: StarRocksDialect) -> None:
        sql = dialect.bulk_tables_sql("", ["Analytics", "raw"])
        assert "table_schema IN ('Analytics', 'raw')" in sql
        assert "LOWER(" not in sql


class TestStatementTimeout:
    def test_the_cap_is_a_session_variable(self, dialect: StarRocksDialect) -> None:
        assert dialect.statement_timeout_sql(120) == "SET query_timeout = 120"

    def test_the_symbolic_timeout_message_is_recognized(
        self, dialect: StarRocksDialect
    ) -> None:
        assert dialect.is_statement_timeout_error(
            Exception(
                "5024 (HY000): Query timeout. Query exceeded time limit of 1 seconds"
            )
        )

    def test_the_numbered_timeout_form_is_recognized(
        self, dialect: StarRocksDialect
    ) -> None:
        """The driver leads with the code and its SQLSTATE; a message that kept
        only the code must still classify as a timeout."""
        assert dialect.is_statement_timeout_error(Exception("5024 (HY000): ..."))

    def test_an_ordinary_error_is_not_a_timeout(
        self, dialect: StarRocksDialect
    ) -> None:
        assert not dialect.is_statement_timeout_error(
            Exception("1064 (HY000): Column 'nope' cannot be resolved.")
        )


class TestBulkSchema:
    def test_the_whole_catalog_is_one_query(self, dialect: StarRocksDialect) -> None:
        sql = dialect.bulk_schema_sql()
        assert "FROM information_schema.columns" in sql
        assert "ordinal_position" in sql

    def test_the_server_schemas_are_excluded(self, dialect: StarRocksDialect) -> None:
        """`information_schema`, `sys` and `_statistics_` hold server metadata,
        not user tables."""
        sql = dialect.bulk_schema_sql()
        assert (
            "table_schema NOT IN ('information_schema', 'sys', '_statistics_')" in sql
        )

    def test_no_scope_is_needed(self, dialect: StarRocksDialect) -> None:
        """The session's catalog is the one the connection was opened on."""
        assert dialect.bulk_schema_sql("default_catalog") == dialect.bulk_schema_sql()


class TestBulkTables:
    def test_the_table_listing_reads_no_column(self, dialect: StarRocksDialect) -> None:
        sql = dialect.bulk_tables_sql("", ["analytics"])
        assert "FROM information_schema.tables" in sql
        assert "column_name" not in sql

    def test_no_schemas_matches_nothing(self, dialect: StarRocksDialect) -> None:
        assert "1 = 0" in dialect.bulk_tables_sql("", [])
