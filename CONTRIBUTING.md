# Contributing

Thanks for improving CalendarBench. This page covers the test suite and coverage
tooling. For the runnable stack and pipelines see [README.md](README.md); for the
experiment configuration schema see [CONFIG.md](CONFIG.md).

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
