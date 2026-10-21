"""StarRocks, for real. Skipped unless `DCT_STARROCKS_HOST` is set.

Any StarRocks answers these. Start one with
`docker run -p 9030:9030 starrocks/allin1-ubuntu:3.3.9` and set
`DCT_STARROCKS_HOST=localhost`.

The fixture creates a scratch database and a SELECT-only user on it, so the
read-only posture dbt charts documents is the one actually exercised: every
query below runs as a user that cannot write.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from dbt_charts.cli.filesystem_project import FilesystemProject
from dbt_charts.core.compile.models.query.normalized import SqlQuery
from dbt_charts.core.compile.models.source import StarRocksSourceConfig

# Aliased: pytest collects a module-level name starting with `test_` as a
# test, and `test_connection` is the connection probe, not a case.
from dbt_charts.core.connections import (
    bulk_schema_for_config,
    test_connection as probe_connection,
)
from dbt_charts.core.dialects import get_dialect
from dbt_charts.core.execute.adapters.adapter_registry import build_adapter_registry

pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("DCT_STARROCKS_HOST"),
        reason="set DCT_STARROCKS_HOST to run against a StarRocks FE",
    ),
    # One worker for the whole module. xdist's default `load` scheduler hands
    # out single tests, so a module-scoped fixture is set up once *per worker
    # that runs a test from the module* — and each setup drops and recreates
    # the same scratch database, which then collides with the next worker's
    # CREATE. The group pins every test here to one worker, so the fixture is
    # built exactly once.
    pytest.mark.xdist_group("starrocks_server"),
]

_HOST = os.environ.get("DCT_STARROCKS_HOST", "")
_PORT = int(os.environ.get("DCT_STARROCKS_PORT", "9030"))
_USER = os.environ.get("DCT_STARROCKS_USER", "root")
_PASSWORD = os.environ.get("DCT_STARROCKS_PASSWORD", "")
_CATALOG = os.environ.get("DCT_STARROCKS_CATALOG", "default_catalog")

_DATABASE = "dct_charts_test"
_READER = "dct_charts_reader"
_READER_PASSWORD = "dct_charts_reader_pw"


def _admin() -> object:
    import mysql.connector

    return mysql.connector.connect(
        host=_HOST, port=_PORT, username=_USER, password=_PASSWORD
    )


def _wait_for_database_gone(name: str, timeout: float = 60.0) -> None:
    """Block until *name* is gone from the server's catalog.

    A database left behind by an earlier run is dropped here, and the catalog
    can keep listing the name for a moment while the server reclaims it.
    Creating the scratch database while the old one is still listed is refused
    with "database exists", so the create waits for the name to be released
    rather than assuming the drop statement was the last word.
    """
    import time

    deadline = time.monotonic() + timeout
    while True:
        admin = _admin()
        try:
            cursor = admin.cursor()
            cursor.execute(
                "SELECT schema_name FROM information_schema.schemata "
                "WHERE schema_name = %s",
                (name,),
            )
            present = cursor.fetchall()
            cursor.close()
        finally:
            admin.close()
        if not present:
            return
        if time.monotonic() >= deadline:
            raise AssertionError(
                f"database {name!r} was still present {timeout}s after its drop"
            )
        time.sleep(0.5)


@pytest.fixture(scope="module")
def reader_source() -> Iterator[StarRocksSourceConfig]:
    """A SELECT-only user on a scratch database, and the source that uses it."""

    admin = _admin()
    cursor = admin.cursor()
    cursor.execute(f"DROP DATABASE IF EXISTS {_DATABASE}")
    cursor.execute(f"DROP USER IF EXISTS '{_READER}'@'%'")
    cursor.close()
    admin.close()
    # The drop above is asynchronous — the name is only free once the server
    # says so.
    _wait_for_database_gone(_DATABASE)

    admin = _admin()
    cursor = admin.cursor()
    for statement in (
        f"CREATE DATABASE {_DATABASE}",
        f"CREATE TABLE {_DATABASE}.orders ("
        "id INT, customer_id VARCHAR(64), amount DECIMAL(18,2), ordered_on DATE"
        ") DUPLICATE KEY(id) DISTRIBUTED BY HASH(id) BUCKETS 1"
        " PROPERTIES ('replication_num' = '1')",
        f"INSERT INTO {_DATABASE}.orders VALUES "
        "(1, 'a', 10.50, '2024-01-15'), (2, 'b', 20.50, '2024-02-20'),"
        " (3, 'c', 30.00, '2024-03-25')",
        f"CREATE USER '{_READER}'@'%' IDENTIFIED BY '{_READER_PASSWORD}'",
        f"GRANT SELECT ON {_DATABASE}.orders TO '{_READER}'@'%'",
    ):
        cursor.execute(statement)
    cursor.close()
    admin.close()

    source = StarRocksSourceConfig(
        type="starrocks",
        host=_HOST,
        port=_PORT,
        catalog=_CATALOG,
        schema=_DATABASE,
        username=_READER,
        password=_READER_PASSWORD,
    )
    try:
        yield source
    finally:
        admin = _admin()
        cursor = admin.cursor()
        for statement in (
            f"DROP USER IF EXISTS '{_READER}'@'%'",
            f"DROP DATABASE IF EXISTS {_DATABASE}",
        ):
            cursor.execute(statement)
        cursor.close()
        admin.close()


def _rows(source: StarRocksSourceConfig, sql: str) -> list[dict[str, object]]:
    """Run *sql* as the source's own reader user, and return the rows."""
    import mysql.connector

    connection = mysql.connector.connect(
        host=source.host,
        port=source.port,
        username=source.username,
        password=source.password,
        database=f"{source.catalog}.{source.schema_}",
        buffered=True,
    )
    try:
        cursor = connection.cursor()
        cursor.execute(sql)
        names = [column[0] for column in cursor.description or []]
        return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]
    finally:
        connection.close()


class TestConnection:
    def test_the_configured_credentials_reach_the_server(
        self, reader_source: StarRocksSourceConfig
    ) -> None:
        ok, error = probe_connection(reader_source)
        assert ok, error
        assert error is None

    def test_a_wrong_password_is_a_classified_failure(
        self, reader_source: StarRocksSourceConfig
    ) -> None:
        ok, error = probe_connection(
            reader_source.model_copy(update={"password": "not-the-password"})
        )
        assert not ok
        assert error is not None


class TestQueryExecution:
    def test_a_select_returns_its_rows(
        self, reader_source: StarRocksSourceConfig
    ) -> None:
        rows = _rows(
            reader_source,
            f"SELECT id, amount FROM {_DATABASE}.orders ORDER BY id",
        )
        assert [row["id"] for row in rows] == [1, 2, 3]
        assert str(rows[0]["amount"]) == "10.50"

    def test_the_column_is_compared_as_a_date(
        self, reader_source: StarRocksSourceConfig
    ) -> None:
        """`CAST(x AS DATE)` is what the filter helper spells for this dialect,
        so the server has to accept it — and to compare on the day, not the
        instant."""
        rows = _rows(
            reader_source,
            f"SELECT CAST(ordered_on AS DATE) AS d FROM {_DATABASE}.orders "
            "WHERE CAST(ordered_on AS DATE) >= '2024-02-01' ORDER BY d",
        )
        assert [str(row["d"]) for row in rows] == ["2024-02-20", "2024-03-25"]

    def test_a_backslash_survives_the_round_trip(
        self, reader_source: StarRocksSourceConfig
    ) -> None:
        """The dialect escapes backslashes in a string literal, so a value
        carrying one must come back as it went in."""
        value = "a\\b'c"
        rows = _rows(
            reader_source,
            "SELECT " + _literal(value) + " AS v",
        )
        assert rows[0]["v"] == value


def _literal(value: str) -> str:
    """The dialect's own string-literal spelling of *value*."""
    return get_dialect("starrocks").string_literal(value)


class TestBulkSchema:
    def test_the_whole_catalog_is_one_query(
        self, reader_source: StarRocksSourceConfig
    ) -> None:
        tree = bulk_schema_for_config(reader_source)
        assert _DATABASE in tree
        assert "orders" in tree[_DATABASE]
        columns = tree[_DATABASE]["orders"]
        assert columns["customer_id"] == "varchar"
        assert columns["amount"] == "decimal"
        assert columns["ordered_on"] == "date"

    def test_the_server_schemas_are_not_in_the_tree(
        self, reader_source: StarRocksSourceConfig
    ) -> None:
        tree = bulk_schema_for_config(reader_source)
        assert "sys" not in tree
        assert "information_schema" not in tree


class TestReadOnlyPosture:
    def test_the_reader_cannot_write(
        self, reader_source: StarRocksSourceConfig
    ) -> None:
        """The posture dbt charts documents: a SELECT-only user, with the
        refusal coming from the warehouse."""
        import mysql.connector

        with pytest.raises(mysql.connector.Error) as excinfo:
            _rows(
                reader_source,
                f"INSERT INTO {_DATABASE}.orders VALUES (4, 'd', 1.00, '2024-04-01')",
            )
        assert "privilege" in str(excinfo.value).lower()


class TestProjectRegistry:
    """The path a board render actually takes: a project's source registry
    resolved by name, executed through the adapter registry."""

    def test_a_query_runs_through_the_registry(
        self,
        reader_source: StarRocksSourceConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("DCT_TEST_STARROCKS_PASSWORD", reader_source.password)
        (tmp_path / "dbt_charts.yml").write_text(
            "sources:\n"
            "  sr:\n"
            "    type: starrocks\n"
            f"    host: {reader_source.host}\n"
            f"    port: {reader_source.port}\n"
            f"    catalog: {reader_source.catalog}\n"
            f"    schema: {reader_source.schema_}\n"
            f"    username: {reader_source.username}\n"
            "    password: \"{{ env_var('DCT_TEST_STARROCKS_PASSWORD') }}\"\n"
        )
        project = FilesystemProject(tmp_path)
        registry = build_adapter_registry(project)
        try:
            result = registry.execute(
                SqlQuery(
                    sql=f"SELECT id, amount FROM {_DATABASE}.orders ORDER BY id",
                    source="sr",
                )
            )
        finally:
            registry.close()
        assert not result.error, result.error
        assert [row["id"] for row in result.data] == [1, 2, 3]

    def test_the_row_cap_is_enforced(
        self,
        reader_source: StarRocksSourceConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The cap rides the driver's own limit kwarg, so the fetch itself is
        bounded rather than the result sliced afterwards."""
        monkeypatch.setenv("DCT_TEST_STARROCKS_PASSWORD", reader_source.password)
        (tmp_path / "dbt_charts.yml").write_text(
            "sources:\n"
            "  sr:\n"
            "    type: starrocks\n"
            f"    host: {reader_source.host}\n"
            f"    port: {reader_source.port}\n"
            f"    catalog: {reader_source.catalog}\n"
            f"    schema: {reader_source.schema_}\n"
            f"    username: {reader_source.username}\n"
            "    password: \"{{ env_var('DCT_TEST_STARROCKS_PASSWORD') }}\"\n"
        )
        project = FilesystemProject(tmp_path)
        registry = build_adapter_registry(project)
        try:
            result = registry.execute(
                SqlQuery(
                    sql=f"SELECT id FROM {_DATABASE}.orders ORDER BY id",
                    source="sr",
                    limit=2,
                )
            )
        finally:
            registry.close()
        assert not result.error, result.error
        assert len(result.data) == 2
