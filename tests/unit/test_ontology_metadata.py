"""Unit tests for the ontology catalog."""

from __future__ import annotations

from src.graphrag.ontology_metadata import (
    _CATALOG_PATH,
    ONTOLOGIES,
    OntologySpec,
    _load_catalog,
    by_key,
    uri_prefixes_for_keys,
)


def test_catalog_yaml_file_exists():
    assert _CATALOG_PATH.is_file()
    assert _CATALOG_PATH.name == "ontology_metadata.yaml"


def test_load_catalog_round_trips_to_ontologies():
    # The module builds ONTOLOGIES from the YAML at import; reloading the
    # catalog yields the same frozen specs.
    assert _load_catalog() == ONTOLOGIES


def test_catalog_is_non_empty():
    assert len(ONTOLOGIES) >= 17
    assert all(isinstance(s, OntologySpec) for s in ONTOLOGIES)


def test_by_key_finds_existing():
    spec = by_key("BCIO")
    assert spec is not None
    assert spec.key == "BCIO"
    assert spec.title.startswith("Behaviour")


def test_by_key_returns_none_for_unknown():
    assert by_key("DEFINITELY_NOT_AN_ONTOLOGY") is None


def test_keys_are_unique():
    keys = [s.key for s in ONTOLOGIES]
    assert len(keys) == len(set(keys)), f"Duplicate keys: {keys}"


def test_files_are_unique():
    files = [s.file for s in ONTOLOGIES]
    assert len(files) == len(set(files)), f"Duplicate filenames: {files}"


def test_every_spec_has_download_url():
    for spec in ONTOLOGIES:
        if spec.local:
            assert (
                spec.download_url == ""
            ), f"{spec.key} is local but has a non-empty download_url"
            continue
        assert spec.download_url, f"{spec.key} has empty download_url"
        assert spec.download_url.startswith("http"), spec.key


def test_local_specs_use_turtle_or_owl_format():
    for spec in ONTOLOGIES:
        if spec.local:
            assert spec.rdf_format in (
                "Turtle",
                "RDF/XML",
                "JSON-LD",
                "N-Triples",
            ), f"{spec.key} has unsupported rdf_format {spec.rdf_format!r}"


def test_bioportal_urls_have_apikey_placeholder():
    bioportal_specs = [
        s for s in ONTOLOGIES if "data.bioontology.org" in s.download_url
    ]
    assert bioportal_specs, "Expected at least one BioPortal-hosted ontology"
    for spec in bioportal_specs:
        assert (
            "{apikey}" in spec.download_url
        ), f"{spec.key} BioPortal URL missing {{apikey}} placeholder"


def test_obo_purls_have_no_apikey():
    obo_specs = [s for s in ONTOLOGIES if "purl.obolibrary.org" in s.download_url]
    assert obo_specs, "Expected at least one OBO Foundry-hosted ontology"
    for spec in obo_specs:
        assert (
            "{apikey}" not in spec.download_url
        ), f"{spec.key} OBO purl shouldn't need an API key"


class TestUriPrefixesForKeys:
    """The scenario YAML's `task_generation.ontologies` list is
    resolved to URI prefixes via `uri_prefixes_for_keys` and fed
    into the GraphRAG retriever's `uri_prefixes` filter.  Without
    correct resolution the retriever falls back to the unfiltered
    multi-ontology pool and fetch retries exhaust on off-domain
    candidates."""

    def test_healthtasks_returns_instance_namespace(self):
        prefixes = uri_prefixes_for_keys(["HealthTasks"])
        assert prefixes == ["https://w3id.org/calendar-bench/health/task/"]

    def test_humanactivities_returns_namespace(self):
        prefixes = uri_prefixes_for_keys(["HumanActivities"])
        assert prefixes == ["https://w3id.org/calendar-bench/human-activities/"]

    def test_both_local_ontologies_combine(self):
        prefixes = uri_prefixes_for_keys(["HealthTasks", "HumanActivities"])
        assert prefixes == [
            "https://w3id.org/calendar-bench/health/task/",
            "https://w3id.org/calendar-bench/human-activities/",
        ]

    def test_caller_order_preserved(self):
        prefixes = uri_prefixes_for_keys(["HumanActivities", "HealthTasks"])
        assert prefixes[0].endswith("human-activities/")
        assert prefixes[1].endswith("health/task/")

    def test_unknown_key_silently_dropped(self):
        prefixes = uri_prefixes_for_keys(["DEFINITELY_NOT_AN_ONTOLOGY"])
        assert prefixes == []

    def test_known_key_with_no_namespaces_dropped(self):
        """BioPortal-hosted ontologies have no `uri_namespaces` declared
        so they contribute background context only; they must NOT
        narrow the retrieval gate."""
        prefixes = uri_prefixes_for_keys(["BCIO"])  # background-only
        assert prefixes == []

    def test_mixed_known_unknown_keys(self):
        prefixes = uri_prefixes_for_keys(["HealthTasks", "MADE_UP", "BCIO"])
        assert prefixes == ["https://w3id.org/calendar-bench/health/task/"]

    def test_duplicate_keys_deduplicated(self):
        prefixes = uri_prefixes_for_keys(["HealthTasks", "HealthTasks"])
        assert prefixes == ["https://w3id.org/calendar-bench/health/task/"]

    def test_empty_list_returns_empty(self):
        assert uri_prefixes_for_keys([]) == []


class TestUriNamespacesField:
    """The two locally-authored ontologies must declare URI namespaces
    so the scenario YAML can gate retrieval on them.  Catalog ontologies
    sourced from BioPortal default to no namespaces; they remain
    available as background context but never narrow the retriever."""

    def test_healthtasks_declares_instance_namespace(self):
        spec = by_key("HealthTasks")
        assert spec is not None
        assert spec.uri_namespaces == ("https://w3id.org/calendar-bench/health/task/",)

    def test_humanactivities_declares_root_namespace(self):
        spec = by_key("HumanActivities")
        assert spec is not None
        assert spec.uri_namespaces == (
            "https://w3id.org/calendar-bench/human-activities/",
        )

    def test_bioportal_specs_default_to_no_namespaces(self):
        for spec in ONTOLOGIES:
            if not spec.local:
                assert spec.uri_namespaces == (), (
                    f"{spec.key} declares uri_namespaces but is sourced from a "
                    "third-party catalog; set them only for ontologies you "
                    "want the scenario YAML to be able to gate retrieval on."
                )
