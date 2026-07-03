"""Unit tests for neo4j_client: factory + n10s init."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from neo4j.exceptions import ClientError

from src.graphrag.config import Neo4jSettings
from src.graphrag.neo4j_client import (
    ensure_n10s_initialized,
    make_driver,
    session_scope,
)


def _settings() -> Neo4jSettings:
    return Neo4jSettings(
        uri="bolt://x:7687",
        username="neo4j",
        password="x",
        database="neo4j",
    )


def test_make_driver_uses_settings():
    with patch("src.graphrag.neo4j_client.GraphDatabase.driver") as mock_drv:
        mock_drv.return_value = "fake_driver"
        result = make_driver(_settings())
    assert result == "fake_driver"
    args, kwargs = mock_drv.call_args
    assert args[0] == "bolt://x:7687"
    # auth tuple
    assert kwargs.get("auth") == ("neo4j", "x") or args[1] == ("neo4j", "x")


def test_make_driver_disables_unrecognized_notifications():
    """Driver is built with the UNRECOGNIZED notification class disabled.

    Retrieval Cypher unions multiple ontology relationship types
    (e.g. SUBCLASSOF, BROADER); without this flag every absent type
    fires a stderr warning that drowns the user's status lines.
    """
    with patch("src.graphrag.neo4j_client.GraphDatabase.driver") as mock_drv:
        mock_drv.return_value = "fake_driver"
        make_driver(_settings())
    _, kwargs = mock_drv.call_args
    assert kwargs.get("notifications_disabled_classifications") == ["UNRECOGNIZED"]


def test_make_driver_falls_back_to_env_when_no_settings():
    """Calling `make_driver()` with no argument resolves settings from
    `Neo4jSettings.from_env` and forwards them to the driver factory.
    """
    fake_settings = Neo4jSettings(
        uri="bolt://from-env:7687", username="u", password="p", database="neo4j"
    )
    with (
        patch(
            "src.graphrag.neo4j_client.Neo4jSettings.from_env",
            return_value=fake_settings,
        ),
        patch("src.graphrag.neo4j_client.GraphDatabase.driver") as mock_drv,
    ):
        mock_drv.return_value = "fake_driver"
        make_driver()
    args, _ = mock_drv.call_args
    assert args[0] == "bolt://from-env:7687"


def test_session_scope_uses_explicit_database():
    """`session_scope` opens a session bound to the requested database."""
    fake_session = MagicMock()
    driver = MagicMock()
    driver.session.return_value.__enter__.return_value = fake_session
    driver.session.return_value.__exit__.return_value = False

    with session_scope(driver, database="my_db") as s:
        assert s is fake_session
    driver.session.assert_called_once_with(database="my_db")


def test_session_scope_falls_back_to_env_database():
    """Without an explicit `database` argument, the env default is used."""
    fake_session = MagicMock()
    driver = MagicMock()
    driver.session.return_value.__enter__.return_value = fake_session
    driver.session.return_value.__exit__.return_value = False

    fake_settings = Neo4jSettings(
        uri="bolt://x:7687", username="u", password="p", database="env_db"
    )
    with patch(
        "src.graphrag.neo4j_client.Neo4jSettings.from_env",
        return_value=fake_settings,
    ):
        with session_scope(driver) as s:
            assert s is fake_session
    driver.session.assert_called_once_with(database="env_db")


def test_ensure_n10s_initialized_first_run_calls_init():
    session = MagicMock()
    ensure_n10s_initialized(session)
    queries = [c.args[0] for c in session.run.call_args_list]
    assert any("CREATE CONSTRAINT" in q for q in queries)
    assert any("graphconfig.init" in q for q in queries)


def test_ensure_n10s_initialized_swallows_non_empty_error():
    session = MagicMock()
    constraint_result = MagicMock()

    def run_side_effect(query, *args, **kwargs):
        if "graphconfig.init" in query:
            raise ClientError("The graph is non-empty. Config cannot be changed.")
        return constraint_result

    session.run.side_effect = run_side_effect
    # Should not raise.
    ensure_n10s_initialized(session)


def test_ensure_n10s_initialized_propagates_unrelated_errors():
    session = MagicMock()

    def run_side_effect(query, *args, **kwargs):
        if "graphconfig.init" in query:
            raise ClientError("Something completely different happened.")
        return MagicMock()

    session.run.side_effect = run_side_effect
    with pytest.raises(ClientError):
        ensure_n10s_initialized(session)
