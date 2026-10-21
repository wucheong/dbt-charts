"""StarRocks dialect implementation.

StarRocks speaks the MySQL wire protocol on the query port (9030), and
dbt-starrocks drives it with mysql-connector-python, so:
- %s positional parameters (mysql-connector's pyformat style)
- backslash escape sequences in string literals
- backtick-quoted identifiers

StarRocks' SQL is not MySQL's, though it shares the wire protocol. The two
differences that matter here are documented on the methods they change.
"""

import re
from collections.abc import Sequence

from dbt_charts.core.dialects.base import SQLDialect

# StarRocks cancels a statement that runs past ``query_timeout`` with error
# 5024: "Query timeout. Query exceeded time limit of N seconds". The
# mysql-connector error text leads with the code and its SQLSTATE
# ("5024 (HY000): Query timeout. ..."), so the numbered form is matched
# together with the symbolic message — a bare "5024" would be too loose to
# anchor on, since the server echoes the query back in several messages.
_TIMEOUT_RE = re.compile(r"\bQuery timeout\b|\b5024 \(HY000\)")

# StarRocks' own schemas, which hold server metadata rather than user
# tables. ``information_schema`` is the ANSI view itself, ``sys`` the system
# catalog, and ``_statistics_`` the internal statistics database.
_SYSTEM_SCHEMAS = ("information_schema", "sys", "_statistics_")


class StarRocksDialect(SQLDialect):
    """StarRocks dialect with %s parameter style.

    Example:
        >>> dialect = StarRocksDialect()
        >>> dialect.param(1)
        '%s'
    """

    name = "starrocks"
    # A '...' constant takes backslash escape sequences (\\', \\\\, \\n, ...),
    # so a backslash in a value has to be doubled before it is inlined.
    # https://docs.starrocks.io/docs/sql-reference/sql-literals/
    escapes_backslashes = True
    # mysql-connector-python's cursor implements fetchmany, so the row cap
    # rides the driver's own limit kwarg rather than a post-fetch slice.
    cursor_supports_driver_limit = True
    # Database and table names are case-sensitive on StarRocks (``GAMING_OPS``
    # is not ``gaming_ops``), so a schema filter must compare them as written.
    # Column names are the exception — they fold case — but a schema filter
    # never compares one.
    folds_identifier_case = False

    def param(self, index: int) -> str:
        """Generate StarRocks parameter placeholder.

        mysql-connector-python binds positionally with ``%s``; the index is
        not part of the placeholder.

        Args:
            index: 1-based parameter index (ignored for StarRocks)

        Returns:
            Parameter placeholder '%s'
        """
        return "%s"

    def quote(self, identifier: str) -> str:
        """Backtick-quote an identifier.

        StarRocks accepts both backticks and double quotes for identifiers —
        its parser resolves ``"x"`` as a column reference, not a string
        literal, the way ANSI SQL does. Backticks are what dbt-starrocks
        itself emits (``StarRocksAdapter.quote``), and a relation must not be
        spelled two ways for the same warehouse depending on which path
        opened it.
        """
        return "`" + identifier.replace("`", "``") + "`"

    def statement_timeout_sql(self, seconds: int) -> str:
        """Generate the session-level ``query_timeout`` SET statement.

        StarRocks caps a statement with a session variable, in whole seconds,
        and 0 would mean *no* limit — so the cap is always spelled as a
        positive integer. The variable is set once per connection, where it
        binds to the session and outlives the statement.
        """
        return f"SET query_timeout = {seconds}"

    def is_statement_timeout_error(self, exc: Exception) -> bool:
        """Return True when StarRocks cancelled the statement at ``query_timeout``."""
        return _TIMEOUT_RE.search(str(exc)) is not None

    def bulk_schema_sql(self, scope: str = "") -> str:
        """Whole-catalog columns from ``information_schema.columns``.

        StarRocks' ``information_schema`` spans the session's catalog — the one
        the connection was opened on — so no scope qualifier is needed. The
        server's own schemas are excluded so the result is user tables only.
        """
        excluded = ", ".join(self.string_literal(name) for name in _SYSTEM_SCHEMAS)
        return (
            "SELECT table_schema, table_name, column_name, data_type "
            "FROM information_schema.columns "
            f"WHERE table_schema NOT IN ({excluded}) "
            "ORDER BY table_schema, table_name, ordinal_position"
        )

    def bulk_tables_sql(self, scope: str, schemas: Sequence[str]) -> str:
        """Whole-catalog tables from ``information_schema.tables``, user tables only.

        The schema filter is always present: an empty *schemas* matches nothing
        (``schema_condition``'s contract), and dropping the clause would list
        the whole catalog instead.
        """
        excluded = ", ".join(self.string_literal(name) for name in _SYSTEM_SCHEMAS)
        condition = self.schema_condition(schemas, "table_schema")
        return (
            "SELECT table_schema, table_name "
            "FROM information_schema.tables "
            f"WHERE table_schema NOT IN ({excluded}) "
            f"AND {condition} "
            "ORDER BY table_schema, table_name"
        )
