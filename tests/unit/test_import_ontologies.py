"""Unit tests for src.scripts.import_ontologies."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from neo4j.exceptions import ClientError

from src.graphrag.ontology_metadata import ONTOLOGIES
from src.scripts.import_ontologies import import_one, main


def _spec():
    return ONTOLOGIES[0]


def test_import_one_calls_n10s_and_metadata():
    session = MagicMock()
    fetch_result = MagicMock()
    fetch_result.single.return_value = {
        "terminationStatus": "OK",
        "triplesParsed": 100,
        "triplesLoaded": 100,
        "extraInfo": "",
    }
    session.run.side_effect = [fetch_result, MagicMock()]

    import_one(session, _spec())

    queries = [c.args[0] for c in session.run.call_args_list]
    assert any("n10s.rdf.import.fetch" in q for q in queries)
    assert any("MERGE (o:Ontology" in q for q in queries)


def test_import_one_handles_n10s_clienterror():
    session = MagicMock()
    session.run.side_effect = ClientError("n10s blew up")
    # Must not raise.
    import_one(session, _spec())


def test_import_one_handles_no_result():
    session = MagicMock()
    fetch_result = MagicMock()
    fetch_result.single.return_value = None
    session.run.return_value = fetch_result
    # Must not raise.
    import_one(session, _spec())


def test_import_one_warns_on_non_ok_status():
    session = MagicMock()
    fetch_result = MagicMock()
    fetch_result.single.return_value = {
        "terminationStatus": "ERROR",
        "triplesParsed": 0,
        "triplesLoaded": 0,
        "extraInfo": "parse error",
    }
    session.run.side_effect = [fetch_result, MagicMock()]
    import_one(session, _spec())


def test_main_iterates_all_ontologies():
    """When all on-disk files are present, every spec is forwarded to import_one."""
    with patch("src.scripts.import_ontologies.make_driver") as mock_drv, patch(
        "src.scripts.import_ontologies.session_scope"
    ) as mock_ss, patch(
        "src.scripts.import_ontologies.ensure_n10s_initialized"
    ) as mock_init, patch(
        "src.scripts.import_ontologies.import_one"
    ) as mock_import_one, patch(
        "pathlib.Path.exists", return_value=True
    ):

        mock_session = MagicMock()
        mock_ss.return_value.__enter__.return_value = mock_session
        mock_ss.return_value.__exit__.return_value = None

        mock_driver = MagicMock()
        mock_drv.return_value = mock_driver

        rc = main()

    assert rc == 0
    mock_driver.verify_connectivity.assert_called_once()
    mock_driver.close.assert_called_once()
    mock_init.assert_called_once_with(mock_session)
    assert mock_import_one.call_count == len(ONTOLOGIES)


def test_main_skips_local_when_file_missing(caplog):
    """spec.local with no on-disk file must skip import_one for that spec."""
    with patch("src.scripts.import_ontologies.make_driver") as mock_drv, patch(
        "src.scripts.import_ontologies.session_scope"
    ) as mock_ss, patch("src.scripts.import_ontologies.ensure_n10s_initialized"), patch(
        "src.scripts.import_ontologies.import_one"
    ) as mock_import_one, patch(
        "pathlib.Path.exists", return_value=False
    ), caplog.at_level(
        "WARNING"
    ):

        mock_session = MagicMock()
        mock_ss.return_value.__enter__.return_value = mock_session
        mock_ss.return_value.__exit__.return_value = None
        mock_drv.return_value = MagicMock()

        rc = main()

    assert rc == 0
    n_local = sum(1 for s in ONTOLOGIES if s.local)
    n_non_local = len(ONTOLOGIES) - n_local
    # Non-local specs still flow through; local ones are skipped.
    assert mock_import_one.call_count == n_non_local

    called_keys = {c.args[1].key for c in mock_import_one.call_args_list}
    for spec in ONTOLOGIES:
        if spec.local:
            assert (
                spec.key not in called_keys
            ), f"Missing-local spec {spec.key} should have been skipped"

    # And we should have at least one warning per skipped local spec.
    skip_warnings = [r for r in caplog.records if "Skipping" in r.message]
    assert len(skip_warnings) >= n_local
