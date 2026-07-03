"""LLM prompts consumed by the metrics stage.

Currently hosts the semantic-compatibility judge used by the LLM-backed
`SemanticCompatibility` scorer. Each module exports a single `TEMPLATE`
string so prompt wording can be iterated on without touching the
renderer or registry.
"""
