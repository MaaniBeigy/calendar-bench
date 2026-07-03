"""Download every ontology listed in ontology_metadata.ONTOLOGIES using curl.

Files are persisted in src/assets/ontologies/ (mounted into the app container at
/app/ontologies). Already-present files are skipped unless --force is given.

BioPortal-hosted URLs require BIOPORTAL_API_KEY in the environment (free key
at https://bioportal.bioontology.org/account). URLs without an `{apikey}`
placeholder (OBO Foundry purls) need no credential.

Examples:
    docker compose run --rm app python -m src.scripts.download_ontologies
    docker compose run --rm app python -m src.scripts.download_ontologies --force
    docker compose run --rm app python -m src.scripts.download_ontologies --only BCIO BCTT
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from src.graphrag.ontology_metadata import ONTOLOGIES, OntologySpec

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("download_ontologies")

# In-container path (bind-mounted from host's src/assets/ontologies).
# Falls back to the host path so the script also runs from a local Python.
_IN_CONTAINER = Path("/app/ontologies")
ONTOLOGIES_DIR = (
    _IN_CONTAINER
    if _IN_CONTAINER.exists()
    else Path(__file__).resolve().parents[2] / "src" / "assets" / "ontologies"
)


def resolve_url(spec: OntologySpec, apikey: str | None) -> str | None:
    if "{apikey}" in spec.download_url:
        if not apikey:
            return None
        return spec.download_url.replace("{apikey}", apikey)
    return spec.download_url


def download_one(spec: OntologySpec, apikey: str | None, force: bool) -> bool:
    if spec.local:
        target = ONTOLOGIES_DIR / spec.file
        if target.exists():
            log.info(
                "[local] %s : author-developed, present at %s", spec.key, target.name
            )
            return True
        log.warning(
            "[local] %s : author-developed but missing on disk (%s). "
            "Place the file under src/assets/ontologies/ to import it.",
            spec.key,
            target,
        )
        return False

    target = ONTOLOGIES_DIR / spec.file
    tmp_target = target.with_suffix(target.suffix + ".part")
    if target.exists() and target.stat().st_size > 0 and not force:
        log.info(
            "[skip ] %s : already present (%d bytes)", spec.key, target.stat().st_size
        )
        return True

    url = resolve_url(spec, apikey)
    if url is None:
        log.warning(
            "[skip ] %s : BIOPORTAL_API_KEY required (URL: %s)",
            spec.key,
            spec.download_url,
        )
        return False

    if tmp_target.exists():
        tmp_target.unlink()

    log.info("[fetch] %s -> %s", spec.key, target.name)

    cmd = [
        "curl",
        "-fSL",
        # Show curl's terminal progress bar.
        "--progress-bar",
        "--retry",
        "3",
        "--retry-delay",
        "2",
        "--connect-timeout",
        "30",
        # Allow up to 60 minutes for large ontologies.
        "--max-time",
        "3600",
        "-A",
        "calendar-bench/1.0",
        "-o",
        str(tmp_target),
        url,
    ]
    try:
        subprocess.run(cmd, check=True, text=True)

    except subprocess.CalledProcessError as exc:
        sanitized_url = spec.download_url
        log.error(
            "[fail ] %s : curl exited %d for %s",
            spec.key,
            exc.returncode,
            sanitized_url,
        )
        if tmp_target.exists():
            tmp_target.unlink()
        return False

    if not tmp_target.exists():
        log.error(
            "[fail ] %s : curl completed but output file was not created", spec.key
        )
        return False

    size = tmp_target.stat().st_size
    if size == 0:
        log.error("[fail ] %s : downloaded 0 bytes; deleting", spec.key)
        tmp_target.unlink()
        return False

    tmp_target.replace(target)
    log.info("[done ] %s : %.1f KiB", spec.key, size / 1024)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Download ontologies via curl.")
    ap.add_argument(
        "--force", action="store_true", help="re-download even if file exists"
    )
    ap.add_argument(
        "--only",
        nargs="+",
        metavar="KEY",
        help="only download these ontology keys (default: all)",
    )
    args = ap.parse_args()

    if not ONTOLOGIES_DIR.exists():
        ONTOLOGIES_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Cache directory: %s", ONTOLOGIES_DIR)

    apikey = os.getenv("BIOPORTAL_API_KEY") or None
    if not apikey:
        log.warning(
            "BIOPORTAL_API_KEY not set, only OBO Foundry sources (ENVO, FOODON) "
            "will succeed. Get a free key at https://bioportal.bioontology.org/account"
        )

    selected = ONTOLOGIES
    if args.only:
        wanted = set(args.only)
        unknown = wanted - {s.key for s in ONTOLOGIES}
        if unknown:
            log.error("Unknown ontology keys: %s", ", ".join(sorted(unknown)))
            return 2
        selected = tuple(s for s in ONTOLOGIES if s.key in wanted)

    ok = 0
    failed: list[str] = []
    for spec in selected:
        if download_one(spec, apikey, args.force):
            ok += 1
        else:
            failed.append(spec.key)

    log.info("Done. %d ok, %d failed.", ok, len(failed))
    if failed:
        log.warning("Failed: %s", ", ".join(failed))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
