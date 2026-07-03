"""Test the CLI factory dispatch for the `rl` method."""

from __future__ import annotations

from dataclasses import dataclass

from src.scripts.scenarios.augmentation.rl.augmenter import RLAugmenter
from src.scripts.scenarios.cli import _build_augmenter


@dataclass
class _FakeEnv:
    seed: int = 42


@dataclass
class _FakeRun:
    time_windows: dict
    daily_window: object
    allen_pair_rules: list
    environment: _FakeEnv


def test_build_augmenter_returns_rl_augmenter():
    run = _FakeRun(
        time_windows={},
        daily_window=None,
        allen_pair_rules=[],
        environment=_FakeEnv(seed=7),
    )
    aug = _build_augmenter("rl", run=run)
    assert isinstance(aug, RLAugmenter)
    assert aug._seed == 7


def test_build_augmenter_rl_without_run_falls_back():
    aug = _build_augmenter("rl")
    assert isinstance(aug, RLAugmenter)
    assert aug._seed == 0


def test_build_augmenter_rl_picks_up_eval_cfg(monkeypatch, tmp_path):
    from src.scripts.scenarios.config.schema import (
        EvaluationConfig,
        LossWeights,
        ScenarioOutputConfig,
    )

    @dataclass
    class _FakeCfg:
        evaluation: EvaluationConfig
        loss: LossWeights
        output: ScenarioOutputConfig
        evaluator_model: object = None

    cfg = _FakeCfg(
        evaluation=EvaluationConfig(buffer_minutes=15, merge_threshold=0.5),
        loss=LossWeights(),
        output=ScenarioOutputConfig(dir=str(tmp_path)),
    )
    run = _FakeRun(
        time_windows={},
        daily_window=None,
        allen_pair_rules=[],
        environment=_FakeEnv(seed=11),
    )
    aug = _build_augmenter("rl", run=run, cfg=cfg)
    assert isinstance(aug, RLAugmenter)
    assert aug._buffer_minutes == 15
    assert aug._merge_threshold == 0.5


def test_build_rl_augmenter_falls_back_to_default_loss_weights(tmp_path):
    from src.scripts.scenarios.cli import _build_rl_augmenter
    from src.scripts.scenarios.config.schema import (
        EvaluationConfig,
        ScenarioOutputConfig,
    )

    @dataclass
    class _CfgNoLoss:
        evaluation: EvaluationConfig
        loss: object
        output: ScenarioOutputConfig
        evaluator_model: object = None

    cfg = _CfgNoLoss(
        evaluation=EvaluationConfig(),
        loss=None,
        output=ScenarioOutputConfig(dir=str(tmp_path)),
    )
    aug = _build_rl_augmenter(cfg=cfg)
    assert isinstance(aug, RLAugmenter)
    assert aug._loss_weights is not None


def test_build_rl_augmenter_empty_context_links_when_asset_absent(
    tmp_path, monkeypatch
):
    from pathlib import Path

    from src.scripts.scenarios.cli import _build_rl_augmenter
    from src.scripts.scenarios.config.schema import (
        EvaluationConfig,
        LossWeights,
        ScenarioOutputConfig,
    )

    real_exists = Path.exists

    def _absent_links(self):
        if self.name.startswith("HealthTasks"):
            return False
        return real_exists(self)

    monkeypatch.setattr(Path, "exists", _absent_links)

    @dataclass
    class _Cfg:
        evaluation: EvaluationConfig
        loss: LossWeights
        output: ScenarioOutputConfig
        evaluator_model: object = None

    cfg = _Cfg(
        evaluation=EvaluationConfig(),
        loss=LossWeights(),
        output=ScenarioOutputConfig(dir=str(tmp_path)),
    )
    aug = _build_rl_augmenter(cfg=cfg)
    assert isinstance(aug, RLAugmenter)
    assert aug._context_links_by_uri == {}


def test_write_rl_training_sidecar_for_solution_writes_when_records_present(tmp_path):
    from types import SimpleNamespace

    from src.scripts.scenarios.cli import _write_rl_training_sidecar_for_solution

    solution = SimpleNamespace(
        rl_training_records=[{"week": 1, "reward": 0.4}, {"week": 2, "reward": 0.6}]
    )
    trace = SimpleNamespace(person_id="rl_0001")
    out = _write_rl_training_sidecar_for_solution(
        solution=solution, trace=trace, persons_dir=tmp_path
    )
    assert out is not None and out.exists()
    assert out.name == "rl_0001_rl_training.json"


def test_run_rl_augment_pool_logs_failures_and_retries(tmp_path, monkeypatch, caplog):
    import logging
    from types import SimpleNamespace

    import src.scripts.scenarios.augmentation.rl.parallel as parallel
    from src.scripts.scenarios.cli import _run_rl_augment_pool
    from src.scripts.scenarios.config.schema import (
        AugmentationConfig,
        ScenarioConfig,
        ScenarioOutputConfig,
    )

    report = SimpleNamespace(
        failed={"rl_0002"}, attempts={"rl_0001": 2, "rl_0002": 3}, written=1
    )
    monkeypatch.setattr(
        parallel, "resolve_rl_pool_workers", lambda req, per_worker_gb: (2, "cpu")
    )
    monkeypatch.setattr(
        parallel,
        "run_rl_process_pool",
        lambda payloads, workers, max_retries: report,
    )

    cfg = ScenarioConfig(
        id="s",
        output=ScenarioOutputConfig(dir=str(tmp_path)),
        augmentation=AugmentationConfig(method="rl"),
    )
    run = SimpleNamespace(
        traces=[SimpleNamespace(person_id=f"rl_{i:04d}") for i in range(1, 4)]
    )
    with caplog.at_level(logging.INFO, logger="scenarios.cli"):
        written = _run_rl_augment_pool(
            run=run,
            run_dir=tmp_path,
            cfg=cfg,
            requested_workers=2,
            tasks_dir=tmp_path,
            persons_dir=tmp_path,
            ics_dir=tmp_path,
            aug_dir=tmp_path,
            horizon_hint=None,
        )
    assert written == 1
    messages = " ".join(r.message for r in caplog.records)
    assert "persons failed after retries" in messages
    assert "needed a retry" in messages
