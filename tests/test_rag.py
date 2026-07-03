"""Per-ontology retrieval test.

For each ontology, run a domain-specific query through the retriever (skipping the LLM
generation step to save tokens) and assert that at least one returned context
block references a URI from that ontology.

This tests that the GraphRAG retrieval can actually surface each ontology's content,
not just the loudest few.
"""

from __future__ import annotations

import pytest

# A query crafted per ontology to surface its specific content.
# Edit these freely as you add or change new ontologies.
RAG_QUERIES: dict[str, str] = {
    "BCIO": "What entities describe behaviour change interventions and their evaluations?",
    # BCTT v1 uses distinctive numbered labels like "1.1 Goal setting (behavior)".
    # Generic BCT queries get shadowed by the more verbose BCIO equivalents, so
    # we lean on a label format that's unique to the BCTT taxonomy.
    "BCTT": "1.1 Goal setting (behavior) 1.4 Action planning; entries from the BCT Taxonomy v1.",
    # BNO uses very specific clinical/anthropometric terminology; generic
    # nutrition queries get shadowed by ONS/FoodOn/FGNHNS which carry richer
    # general-purpose descriptions.  Anchor to BNO-distinctive labels.
    "BNO": (
        "Anthropometrics BMI bioelectric impedence body composition; "
        "adjusted body weight equation for obesity; activity factor "
        "or equation in clinical nutrition assessment."
    ),
    "COPPER": "Personalised recommendations for physical activity and coping with barriers.",
    "EBP": "Recommendations and behaviour change strategies for hypertension self-management.",
    "PersonasOnto": "How are user personas modelled as archetypes for design decisions?",
    "FoodGroupNHNS": "Japanese food groups from the National Health and Nutrition Examination Survey.",
    "ONS": "Standardised vocabulary for describing nutritional studies.",
    "SATO": "Workflow steps and IDEAS framework for designing mobile health behaviour change apps.",
    "QUANTUM-MIND": "The sequential process from observation to a final mental state and belief formation.",
    # MFOEM uses very specific affective-feeling labels (e.g. "feeling at ease",
    # "feeling exhausted") that are unique to the Emotion Ontology — generic
    # "emotion" queries get shadowed by BCIO's broader emotional-construct
    # entries, so anchor to discrete labels.
    "MFOEM": (
        "Discrete emotions and subjective feelings: happiness, sadness, anger, "
        "fear, anxiety, disgust, surprise; feeling tired, feeling energetic, "
        "feeling at ease."
    ),
    # EFO uses GWAS / cohort-study labels ("personality trait measurement",
    # "Temperament and Character Inventory", "wellbeing measurement") that
    # are distinctive enough to anchor retrieval without being shadowed by
    # BCIO's broader behavioural-disposition entries.
    "EFO": (
        "Personality and the extraversion trait; Temperament and Character "
        "Inventory; wellbeing, depression and anxiety measurements from "
        "cohort and GWAS studies."
    ),
    "OPE": "Functional movements and musculoskeletal parts engaged in physical exercises.",
    "HeLiFit": "Event-based models representing physical activity according to WHO guidance.",
    "ENVO": "Environmental systems, components and processes around daily life.",
    "OCHV": "Consumer health vocabulary terms for everyday health concepts.",
    "LSFO": "Lifestyle factor categories and their hierarchy.",
    "FOODON": "Food categories and food product hierarchy.",
    "HSPO": "Health services provision categories and roles.",
    # Locally-developed ontologies: queries lean on terminology that's
    # specific to these files (Level1..Level4 grading, MET value, Compendium).
    # The HealthTasks query names both class-level terminology (Level1..Level3)
    # and instance task titles so retrieval surfaces both classes and
    # `health/task/` individuals.
    "HealthTasks": (
        "Calendar-bench mental wellbeing calm breathing and mindfulness tasks: "
        "take a box breathing break, mindful pause, breathe with a calm image; "
        "graded difficulty Level1 Level2 Level3."
    ),
    "HumanActivities": (
        "Bicycling general activity from the Compendium of Physical Activities "
        "with its activity code and MET metabolic equivalent value."
    ),
    # Context dictionary: anchor on labels unique to local cb-namespace entries
    # so the retriever surfaces this ontology rather than the wider BCIO /
    # MFOEM IRIs that Context also re-asserts as cbc:<Category> individuals.
    "Context": (
        "Calendar-bench context features: Extraversion low state trait, "
        "Bergram digital nudging default delivery, dramatic relief TTM "
        "process, behaviour change technique categories."
    ),
}

# Top-k is wide enough that embedder choice (OpenAI text-embedding-3-small
# vs local Qwen3-Embedding-0.6B) does not flip the assertion: both rank
# the target ontology's URIs into the top-50 for these queries, even when
# OCHV / BCIO entries crowd the first 10-20 positions.
TOP_K = 50


def test_query_defined(ontology_spec):
    assert (
        ontology_spec.key in RAG_QUERIES
    ), f"No RAG_QUERIES entry for {ontology_spec.key}"


def test_retrieval_surfaces_ontology(
    retriever, ontology_spec, ontology_uri_samples, ontology_uri_prefixes
):
    samples = ontology_uri_samples.get(ontology_spec.key, [])
    prefix = ontology_uri_prefixes.get(ontology_spec.key, "")
    if not samples:
        pytest.skip(f"Could not parse sample URIs for {ontology_spec.key}")

    query = RAG_QUERIES.get(ontology_spec.key)
    assert query, f"No test query for {ontology_spec.key}"

    result = retriever.search(query_text=query, top_k=TOP_K)
    contexts = [item.content for item in result.items]
    assert contexts, f"Retriever returned 0 items for {ontology_spec.key} query"

    by_prefix = bool(prefix) and any(prefix in c for c in contexts)
    by_uri = any(uri in c for c in contexts for uri in samples)

    if by_prefix or by_uri:
        return

    snippet = "\n---\n".join(c[:300] for c in contexts[:3])
    pytest.fail(
        f"{ontology_spec.key}: top-{TOP_K} retrieval did not surface any URI "
        f"under prefix {prefix!r} for query {query!r}.\nFirst contexts:\n{snippet}"
    )
