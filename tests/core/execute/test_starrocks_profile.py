"""A `type: starrocks` profiles.yml target, read through dbt.

StarRocks has a dbt adapter, so dbt — not dbt charts — owns the target's
schema: the target is validated by dbt-starrocks' own credentials class and
carried through untouched. What dbt charts adds is the routing: the resolved
source's type must name a dialect and an adapter, or the connection is
refused with "unsupported adapter type" long after the author was told the
source was fine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from dbt_charts.cli.filesystem_project import FilesystemProject
from dbt_charts.core.compile.config import ProjectSourcesConfig
from dbt_charts.core.compile.models.source import ResolvedSourceConfig
from dbt_charts.core.dialects import get_dialect
from dbt_charts.core.execute.adapters.dbt_adapter_factory import build_adapter
from dbt_charts.core.execute.source_resolver import DbtContext, DefaultSourceResolver

_TARGET = {
    "type": "starrocks",
    "host": "fe.example.com",
    "port": 9030,
    "catalog": "default_catalog",
    "schema": "gaming_ops",
    "username": "reader",
    "password": "s3cret",
    "threads": 4,
}


def _write_profile(project_dir: Path, target: dict[str, Any]) -> None:
    (project_dir / "profiles.yml").write_text(
        yaml.dump({"analytics": {"target": "dev", "outputs": {"dev": target}}})
    )


def _resolve(
    project_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> ResolvedSourceConfig:
    monkeypatch.delenv("DBT_PROFILES_DIR", raising=False)
    _write_profile(project_dir, _TARGET)
    result = DefaultSourceResolver().resolve(
        authored={"type": "dbt_profile", "profile": "analytics"},
        board_sources={},
        project_sources=ProjectSourcesConfig(),
        dbt_context=DbtContext(dbt_project_path=project_dir),
    )
    assert result is not None
    return result


def test_a_starrocks_target_resolves_to_a_starrocks_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _resolve(tmp_path, monkeypatch)
    assert source.type == "starrocks"
    # A dbt_profile expands into DbtTargetSourceConfig, which is extra="allow":
    # dbt owns profiles.yml, so the target's own field names (schema, not
    # schema_) reach the model as extras rather than declared fields.
    dumped = source.model_dump()
    assert dumped["host"] == "fe.example.com"
    assert dumped["schema"] == "gaming_ops"
    assert dumped["catalog"] == "default_catalog"
    assert dumped["username"] == "reader"


def test_the_resolved_type_names_a_dialect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A type with no dialect would fail at the first query, not at resolution."""
    source = _resolve(tmp_path, monkeypatch)
    assert get_dialect(source.type).name == "starrocks"


def test_the_resolved_target_builds_a_starrocks_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The credentials dbt validated are the ones the adapter is built from."""
    source = _resolve(tmp_path, monkeypatch)
    creds = source.model_dump(by_alias=True)
    adapter = build_adapter(creds, register_macros=False)
    assert type(adapter).__name__ == "StarRocksAdapter"
    assert adapter.config.credentials.host == "fe.example.com"
    assert adapter.config.credentials.username == "reader"
    assert adapter.config.credentials.schema == "gaming_ops"


def test_a_project_whose_own_target_is_starrocks_builds_a_starrocks_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dbt project whose profile points at StarRocks is reached through its
    own target, the same posture as every other dbt-adapter-backed warehouse.

    Only the driver-seam warehouses (lakecompute, trino) are refused here with
    ERR-DBT-TARGET-NEEDS-SOURCE, because dbt charts reaches them only through a
    declared source. StarRocks has a dbt adapter, so the project's own target is
    a valid connection — refusing it would push authors to a source they do not
    need.
    """
    from dbt_charts.core.execute.adapters.dbt_adapter import DbtAdapter

    monkeypatch.delenv("DBT_PROFILES_DIR", raising=False)
    _write_profile(tmp_path, _TARGET)
    (tmp_path / "dbt_project.yml").write_text("name: analytics\nprofile: analytics\n")
    adapter = DbtAdapter(
        project=FilesystemProject(tmp_path),
        dbt_project_path=tmp_path,
        target_name="dev",
    )
    assert adapter.resolve_target_type() == "starrocks"
    built = adapter._get_dbt_adapter()
    assert type(built).__name__ == "StarRocksAdapter"
    # The dialect the target names is the one the query is written in; if it
    # were unresolved the query would fail at its first statement, not here.
    assert adapter._dialect == "starrocks"
    assert get_dialect(adapter._dialect).name == "starrocks"
