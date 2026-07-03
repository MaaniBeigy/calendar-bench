"""LLM prompts consumed by the augmentation stage.

The LLM-agent augmenter picks one of these via the YAML
`augmentation.llm_agent.prompt_template` field. Each module exports a
single `TEMPLATE` string so researchers can iterate on prompt wording
without touching the renderer or registry.
"""
