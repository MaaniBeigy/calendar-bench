"""Shared snippets reused inside multiple task-generation prompts."""

from __future__ import annotations

VERBATIM_CONTENT_RULES = """\
CONTENT RULES — ontology_uri, display_name and description MUST come from
the ontology context retrieved by the GraphRAG knowledge graph.  Do NOT
paraphrase, do NOT invent emojis, do NOT remove emojis.

  ontology_uri  Look at the ``uri:`` line at the bottom of every
                context block above.  Each task you return MUST cite a
                URI you SEE in those context blocks.  Copy the URI
                byte-for-byte from the retrieval context.  Do NOT
                extrapolate the URI shape, do NOT construct one from a
                label, do NOT invent a URI even if it would "make
                sense", and do NOT reuse a URI you have seen in a
                previous prompt or in your training data — only the
                URIs printed inside this prompt's context blocks are
                valid.

                Spread your picks across DIFFERENT context blocks.  If
                the retrieval context contains 10 candidate instance
                URIs, pick 5 distinct ones — do not return 5 copies of
                the most prominent one.  Different personas in this run
                should end up with different URI sets, so prefer URIs
                that match this persona's profile and scenario filters
                rather than always picking the first / shortest one.

                Reject any URI whose local name ends in ``Task``,
                ``LevelN``, ``Activity`` or another class-name suffix
                — those are OWL class nodes, not authored task
                instances; the validator drops them.

  display_name  Copy it VERBATIM from the GraphRAG retrieval context (the
                ontology label / displayName / dcterms:title / prefLabel
                for the chosen instance URI).  PRESERVE any emojis exactly
                as they appear — do NOT add emojis the context does not
                contain, and do NOT remove emojis the context does contain.
                When the retrieval context has no display name for this
                URI, derive a short Title-Case title from the snake_case
                ``label`` and add no emoji.

  description   Copy it VERBATIM from the GraphRAG retrieval context (the
                ontology dcterms:description / rdfs:comment / dedicated
                description property for the chosen instance URI).
                PRESERVE any emojis exactly as they appear.  Do NOT
                append the URI in square brackets to the description
                — the URI lives in its own ``ontology_uri`` field; the
                downstream pipeline appends a citation automatically
                when it is needed for display.
"""
