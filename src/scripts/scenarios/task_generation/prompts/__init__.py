"""Per-prompt LLM templates used by the task-generation stage.

Each module here exports a single `TEMPLATE` string. Researchers and
developers iterating on prompt wording should edit the file that
matches the prompt name. the central registry in
`task_generation/prompt_templates.py` only wires the constants into
the `render()` API and is rarely the place to make wording changes.
"""
