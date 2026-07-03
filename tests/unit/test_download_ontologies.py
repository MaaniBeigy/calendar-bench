"""Unit tests for download_ontologies: URL resolution + curl invocation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.graphrag.ontology_metadata import OntologySpec
from src.scripts.download_ontologies import download_one, main, resolve_url


def _spec(url: str, file: str = "x.xrdf") -> OntologySpec:
    return OntologySpec(
        key="X",
        file=file,
        title="X",
        rdf_format="RDF/XML",
        description="x",
        download_url=url,
    )


def test_resolve_url_substitutes_apikey():
    spec = _spec("https://example.com/x?apikey={apikey}")
    assert resolve_url(spec, "secret") == "https://example.com/x?apikey=secret"


def test_resolve_url_returns_none_when_apikey_required_but_absent():
    spec = _spec("https://example.com/x?apikey={apikey}")
    assert resolve_url(spec, None) is None


def test_resolve_url_no_placeholder_passes_through():
    spec = _spec("https://example.com/x.owl")
    assert resolve_url(spec, None) == "https://example.com/x.owl"
    assert resolve_url(spec, "ignored") == "https://example.com/x.owl"


def test_download_one_skips_when_cached(tmp_path):
    spec = _spec("https://x", file="cached.owl")
    target = tmp_path / "cached.owl"
    target.write_bytes(b"already here")

    with patch("src.scripts.download_ontologies.ONTOLOGIES_DIR", tmp_path):
        assert download_one(spec, "key", force=False) is True


def test_download_one_skips_without_apikey(tmp_path):
    spec = _spec("https://example.com/x?apikey={apikey}", file="needs-key.xrdf")
    with patch("src.scripts.download_ontologies.ONTOLOGIES_DIR", tmp_path):
        assert download_one(spec, None, force=False) is False


def test_download_one_runs_curl_and_renames_part(tmp_path):
    spec = _spec("https://example.com/x", file="new.owl")

    def fake_run(cmd, **kw):
        out_idx = cmd.index("-o")
        out_path = Path(cmd[out_idx + 1])
        out_path.write_bytes(b"hello world")
        return MagicMock(returncode=0)

    with patch("src.scripts.download_ontologies.ONTOLOGIES_DIR", tmp_path), patch(
        "src.scripts.download_ontologies.subprocess.run", side_effect=fake_run
    ):
        assert download_one(spec, "key", force=False) is True
    assert (tmp_path / "new.owl").read_bytes() == b"hello world"
    assert not (tmp_path / "new.owl.part").exists()


def test_download_one_handles_curl_failure(tmp_path):
    spec = _spec("https://example.com/x", file="fail.owl")

    def fake_run(cmd, **kw):
        # Create the .part file, then fail, the function should clean it up.
        out_idx = cmd.index("-o")
        Path(cmd[out_idx + 1]).write_bytes(b"partial")
        raise subprocess.CalledProcessError(22, cmd, stderr="404")

    with patch("src.scripts.download_ontologies.ONTOLOGIES_DIR", tmp_path), patch(
        "src.scripts.download_ontologies.subprocess.run", side_effect=fake_run
    ):
        assert download_one(spec, "key", force=False) is False
    assert not (tmp_path / "fail.owl").exists()
    assert not (tmp_path / "fail.owl.part").exists()


def test_download_one_handles_empty_response(tmp_path):
    spec = _spec("https://example.com/x", file="empty.owl")

    def fake_run(cmd, **kw):
        out_idx = cmd.index("-o")
        Path(cmd[out_idx + 1]).write_bytes(b"")
        return MagicMock(returncode=0)

    with patch("src.scripts.download_ontologies.ONTOLOGIES_DIR", tmp_path), patch(
        "src.scripts.download_ontologies.subprocess.run", side_effect=fake_run
    ):
        assert download_one(spec, "key", force=False) is False
    assert not (tmp_path / "empty.owl").exists()


def test_download_one_force_re_fetches(tmp_path):
    spec = _spec("https://example.com/x", file="redo.owl")
    (tmp_path / "redo.owl").write_bytes(b"old")

    def fake_run(cmd, **kw):
        out_idx = cmd.index("-o")
        Path(cmd[out_idx + 1]).write_bytes(b"fresh")
        return MagicMock(returncode=0)

    with patch("src.scripts.download_ontologies.ONTOLOGIES_DIR", tmp_path), patch(
        "src.scripts.download_ontologies.subprocess.run", side_effect=fake_run
    ):
        assert download_one(spec, "key", force=True) is True
    assert (tmp_path / "redo.owl").read_bytes() == b"fresh"


def test_main_only_filter(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["x", "--only", "BCIO"])
    monkeypatch.setenv("BIOPORTAL_API_KEY", "key")
    with patch("src.scripts.download_ontologies.ONTOLOGIES_DIR", tmp_path), patch(
        "src.scripts.download_ontologies.download_one"
    ) as mock_dl:
        mock_dl.return_value = True
        rc = main()
    assert rc == 0
    assert mock_dl.call_count == 1
    assert mock_dl.call_args.args[0].key == "BCIO"


def test_main_only_unknown_key_returns_2(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["x", "--only", "NOT_A_REAL_KEY"])
    with patch("src.scripts.download_ontologies.ONTOLOGIES_DIR", tmp_path), patch(
        "src.scripts.download_ontologies.download_one"
    ) as mock_dl:
        rc = main()
    assert rc == 2
    mock_dl.assert_not_called()


def test_main_returns_1_on_any_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["x", "--only", "BCIO"])
    monkeypatch.setenv("BIOPORTAL_API_KEY", "key")
    with patch("src.scripts.download_ontologies.ONTOLOGIES_DIR", tmp_path), patch(
        "src.scripts.download_ontologies.download_one"
    ) as mock_dl:
        mock_dl.return_value = False
        rc = main()
    assert rc == 1


# -------------------------------------------------------------------------------------
# ---- Local-spec branch: skip the curl path entirely for author-developed files. -----
# -------------------------------------------------------------------------------------


def _local_spec(file: str = "local.ttl") -> OntologySpec:
    return OntologySpec(
        key="LOCAL_TEST",
        file=file,
        title="Local",
        rdf_format="Turtle",
        description="x",
        download_url="",
        local=True,
    )


def test_download_one_local_spec_present(tmp_path):
    spec = _local_spec(file="present.ttl")
    (tmp_path / "present.ttl").write_bytes(b"@prefix : <#> .")
    with patch("src.scripts.download_ontologies.ONTOLOGIES_DIR", tmp_path):
        assert download_one(spec, "any-key", force=False) is True


def test_download_one_local_spec_missing(tmp_path):
    spec = _local_spec(file="missing.ttl")
    with patch("src.scripts.download_ontologies.ONTOLOGIES_DIR", tmp_path):
        assert download_one(spec, "any-key", force=False) is False


# -------------------------------------------------------------------------------------
# -------- Curl edge cases: stale .part cleanup + curl-succeeded-but-no-output. -------
# -------------------------------------------------------------------------------------


def test_download_one_clears_stale_part_file(tmp_path):
    """A leftover .part from a previous aborted run should be deleted before curl."""
    spec = _spec("https://example.com/x", file="reuse.owl")
    stale = tmp_path / "reuse.owl.part"
    stale.write_bytes(b"stale partial")

    captured: dict = {}

    def fake_run(cmd, **kw):
        # Record whether the stale file was already cleared at the moment curl runs.
        captured["stale_existed_at_curl_start"] = stale.exists()
        out_idx = cmd.index("-o")
        Path(cmd[out_idx + 1]).write_bytes(b"fresh")
        return MagicMock(returncode=0)

    with patch("src.scripts.download_ontologies.ONTOLOGIES_DIR", tmp_path), patch(
        "src.scripts.download_ontologies.subprocess.run", side_effect=fake_run
    ):
        assert download_one(spec, "key", force=False) is True

    # The stale .part should have been unlinked before curl was invoked.
    assert captured["stale_existed_at_curl_start"] is False
    assert (tmp_path / "reuse.owl").read_bytes() == b"fresh"


def test_download_one_curl_succeeds_but_no_output(tmp_path):
    """curl returns 0 yet writes nothing: script must reject and clean up."""
    spec = _spec("https://example.com/x", file="ghost.owl")

    def fake_run(cmd, **kw):
        # Deliberately do NOT write the -o target.
        return MagicMock(returncode=0)

    with patch("src.scripts.download_ontologies.ONTOLOGIES_DIR", tmp_path), patch(
        "src.scripts.download_ontologies.subprocess.run", side_effect=fake_run
    ):
        assert download_one(spec, "key", force=False) is False
    assert not (tmp_path / "ghost.owl").exists()


# -------------------------------------------------------------------------------------
# ----------------------------- main() bootstrap branches. ----------------------------
# -------------------------------------------------------------------------------------


def test_main_creates_ontologies_dir_when_missing(monkeypatch, tmp_path):
    """If ONTOLOGIES_DIR doesn't exist on disk, main() must mkdir -p it."""
    cache = tmp_path / "does_not_exist_yet"
    assert not cache.exists()

    monkeypatch.setattr(sys, "argv", ["x", "--only", "BCIO"])
    monkeypatch.setenv("BIOPORTAL_API_KEY", "key")
    with patch("src.scripts.download_ontologies.ONTOLOGIES_DIR", cache), patch(
        "src.scripts.download_ontologies.download_one", return_value=True
    ):
        rc = main()

    assert rc == 0
    assert cache.exists() and cache.is_dir()


def test_download_one_curl_failure_without_partial(tmp_path):
    """curl fails before writing anything: covers the False arm of `if tmp_target.exists()`
    inside the except block (no partial file to clean up)."""
    spec = _spec("https://example.com/x", file="fail-empty.owl")

    def fake_run(cmd, **kw):
        # Don't write the -o target: simulate curl failing on connect.
        raise subprocess.CalledProcessError(7, cmd, stderr="connection refused")

    with patch("src.scripts.download_ontologies.ONTOLOGIES_DIR", tmp_path), patch(
        "src.scripts.download_ontologies.subprocess.run", side_effect=fake_run
    ):
        assert download_one(spec, "key", force=False) is False
    assert not (tmp_path / "fail-empty.owl").exists()
    assert not (tmp_path / "fail-empty.owl.part").exists()


def test_main_without_only_processes_all_specs(monkeypatch, tmp_path):
    """No `--only` flag: covers the False arm of `if args.only:` in main()."""
    from src.graphrag.ontology_metadata import ONTOLOGIES

    monkeypatch.setattr(sys, "argv", ["x"])  # no --only
    monkeypatch.setenv("BIOPORTAL_API_KEY", "key")
    with patch("src.scripts.download_ontologies.ONTOLOGIES_DIR", tmp_path), patch(
        "src.scripts.download_ontologies.download_one", return_value=True
    ) as mock_dl:
        rc = main()
    assert rc == 0
    assert mock_dl.call_count == len(ONTOLOGIES)


def test_main_warns_when_no_apikey(monkeypatch, tmp_path, caplog):
    """Without BIOPORTAL_API_KEY, main() emits a warning and continues."""
    monkeypatch.setattr(sys, "argv", ["x", "--only", "BCIO"])
    monkeypatch.delenv("BIOPORTAL_API_KEY", raising=False)
    with caplog.at_level("WARNING"), patch(
        "src.scripts.download_ontologies.ONTOLOGIES_DIR", tmp_path
    ), patch("src.scripts.download_ontologies.download_one", return_value=True):
        rc = main()

    assert rc == 0
    assert any(
        "BIOPORTAL_API_KEY not set" in record.message for record in caplog.records
    ), [r.message for r in caplog.records]
