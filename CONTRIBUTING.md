# Contributing

Thanks for improving CalendarBench. This page covers how to add a model,
augmenter, prompt, or benchmark example, then how to run the tests and coverage.
For the runnable stack and pipelines see [README.md](README.md); for the
experiment configuration schema see [CONFIG.md](CONFIG.md).

## Adding a model, augmenter, or prompt

- **A new LLM model** for an existing method needs no code: add an
  [`augmentation`](CONFIG.md) entry to a scenario with its `provider` / `model`
  (optionally pin `task_generator_model` / `evaluator_model`).
- **A new prompt template** is a Python module referenced by name in
  `prompt_template:`. Put an augmenter prompt beside
  [augment_oneshot.py](src/scripts/scenarios/augmentation/prompts/augment_oneshot.py)
  and a task-generation prompt beside
  [health_improvement.py](src/scripts/scenarios/task_generation/prompts/health_improvement.py).
  Prompt-component ablations register blocks in
  [ablation_designs.py](src/scripts/scenarios/augmentation/prompts/ablation_designs.py)
  with placebo text in
  [augment_oneshot_placebos.py](src/scripts/scenarios/augmentation/prompts/augment_oneshot_placebos.py).
- **A new augmenter or technique** implements the `Augmenter` interface in
  [augmentation/base.py](src/scripts/scenarios/augmentation/base.py), lands as a
  module next to [greedy.py](src/scripts/scenarios/augmentation/greedy.py) /
  [llm_agent.py](src/scripts/scenarios/augmentation/llm_agent.py) /
  [ptime.py](src/scripts/scenarios/augmentation/ptime.py) /
  [rl/](src/scripts/scenarios/augmentation/rl/), gets its method string
  registered in `_build_augmenter` in
  [scenarios/cli.py](src/scripts/scenarios/cli.py), and allowed in the config
  schema [config/schema.py](src/scripts/scenarios/config/schema.py). Add the
  method to a scenario's `augmentation:` list to run it.

> **Pull-request policy.** A PR that adds a model, augmenter, or prompt must only
> **add** to a scenario's `augmentation:` list (or a new prompt module). Do not
> change the shared experiment-design files (`environment.yaml`,
> `persona_config.yaml`, `event_config.yaml`, `temporal_relation_rules.yaml`) or
> the `observation:` block of existing augmenters: those define what every method
> sees, so editing them breaks cross-method comparability and may cause the PR to
> be rejected.

## Adding a benchmark example

More examples than
[example_experiment](src/experiments/persona/example_experiment) and
[healthy_lifestyle_promotion](src/experiments/persona/healthy_lifestyle_promotion)
are welcome. Copy one to `src/experiments/persona/<name>/` and edit its five
YAMLs; the files are wired by content, not filename. See [CONFIG.md](CONFIG.md)
for the schema and [README.md](README.md) for running it end to end.

## Testing

The suite verifies, per ontology, that:

- (a) the source file's URIs are imported into Neo4j,
- (b) embeddings exist for that ontology,
- (c) a domain-specific query through the retriever brings nodes from that ontology.

These are **integration** tests: they need a running Neo4j with ontologies
imported and embeddings built (README steps 3, 4, and 5). A full run takes 5-10
minutes depending on the host.

Run the full suite:

```bash
docker compose run --rm app pytest
```

Run a single test file:

```bash
docker compose run --rm app pytest tests/test_imports.py
docker compose run --rm app pytest tests/test_embeddings.py
docker compose run --rm app pytest tests/test_rag.py
```

Run only the per-ontology slice for one ontology (the parametrize id matches the
spec key in [src/graphrag/ontology_metadata.py](src/graphrag/ontology_metadata.py)):

```bash
docker compose run --rm app pytest -k "BCIO"
docker compose run --rm app pytest -k "OPE or HeLiFit"
```

The retrieval test calls the retriever directly (it skips the LLM), so it spends
only one OpenAI embedding call per ontology query and is cheap to re-run.

### Fast unit tests (no Neo4j, no LLM)

```bash
docker compose run --rm app pytest tests/unit/
```

Pure unit tests with mocks cover config, factories (LLM / embedder / retriever /
pipeline), n10s init, the curl-based downloader, and every script's `main()`
flow. They run in seconds and need neither a database nor an API key.

### End-to-end honesty test (cheap OpenRouter model)

`tests/test_e2e_fake_ontology.py` imports a tiny synthetic ontology fixture into
Neo4j, embeds it, and asks a question that is only answerable from that fixture
via OpenRouter. The default model is `openai/gpt-4o-mini`. Free models were tried
first but are essentially unusable due to aggressive rate limits (8 RPM, frequent
429s). The fixture defines deliberately fictional surgical / hospital terms
(*Quibble bypass surgery*, *Throckmorton operating theatre*, *Vexil-7 clamp*,
*Zorblax registered nurse*), chosen to sit completely outside the behavior-change
/ nutrition / physical-activity domain of the imported ontologies, so retrieval
grounding can be verified without false positives. The test asserts the answer
cites a URI from the fake namespace, proving the LLM did not fall back to its
training prior or to a sibling ontology.

```bash
# Requires OPENROUTER_API_KEY and OPENAI_API_KEY in .env. Skipped otherwise.
docker compose run --rm app pytest tests/test_e2e_fake_ontology.py -v
```

Override the model with `OPENROUTER_TEST_MODEL` (default: `openai/gpt-4o-mini`).
Other cheap options: `anthropic/claude-haiku-4-5`, `google/gemini-2.5-flash-lite`,
`mistralai/mistral-small`.

```bash
OPENROUTER_TEST_MODEL=anthropic/claude-haiku-4-5 \
  docker compose run --rm app pytest tests/test_e2e_fake_ontology.py -v
```

## Coverage

Generate a terminal coverage report (saved to `.logs/coverage.txt`), an HTML
report (browseable at `htmlcov/index.html`), and refresh the badge in
[`src/assets/images/coverage.svg`](src/assets/images/coverage.svg) using
[`genbadge`](https://github.com/smarie/python-genbadge):

```bash
mkdir -p .logs htmlcov src/assets/images
docker compose run --rm app sh -c \
  "pytest --cov-report=term-missing --cov-report=html --cov-report=xml:coverage.xml --cov=src tests/ && \
   genbadge coverage -i coverage.xml -o src/assets/images/coverage.svg" \
  | tee .logs/coverage.txt
```

Open the HTML report afterwards:

```bash
# Linux
xdg-open htmlcov/index.html
# macOS
open htmlcov/index.html
# Windows
explorer.exe ".\htmlcov\index.html"
```

The HTML view colours each source line: green = covered, red = missed, yellow =
partial branch. Click any file to drill in. `htmlcov/` is gitignored.

Notes:

- `genbadge` reads a coverage **XML** report, so `pytest-cov` is asked for two
  reports at once: `term` (printed to stdout, captured by `tee`) and
  `xml:coverage.xml` (consumed by `genbadge`).
- The two commands are chained inside one `sh -c` so they share the container's
  filesystem; with `--rm` the container and its `coverage.xml` are deleted afterwards.
- `--cov=src` measures coverage of the `src/` package.
- `tee` shows the report on screen and writes `.logs/coverage.txt`. Replace with
  `>` for a silent run.
- Rebuild the image once to pick up `genbadge`:

```bash
docker compose build app
```
