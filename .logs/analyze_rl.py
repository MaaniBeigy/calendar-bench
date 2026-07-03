"""Print per-person and cohort learning trajectory for the RL augmenter."""

import glob
import json
import sys
from pathlib import Path


def main(label: str = "current") -> int:
    root = Path("output/experiment_k/scenarios/rl_baseline/rl/augmented/persons")
    files = sorted(root.glob("*_weekly_gain.json"))
    if not files:
        print(f"[{label}] no weekly_gain files found", file=sys.stderr)
        return 1

    per_person_gains: dict[str, list[float]] = {}
    per_person_nrgains: dict[str, list[float]] = {}
    per_person_placements: dict[str, list[int]] = {}
    per_person_terminal_reward: dict[str, list[float]] = {}

    for f in files:
        d = json.loads(f.read_text())
        pid = d["person_id"]
        per_person_gains[pid] = [w["weighted_gain"] for w in d["weeks"]]
        per_person_nrgains[pid] = [
            w.get("non_renormalized_gain", None) for w in d["weeks"]
        ]
        rl_path = f.parent / f"{pid}_rl_training.json"
        if rl_path.exists():
            t = json.loads(rl_path.read_text())
            per_person_placements[pid] = [
                w["tasks_placed_this_week"] for w in t["weeks"]
            ]
            per_person_terminal_reward[pid] = [
                w["episode_terminal_reward"] for w in t["weeks"]
            ]

    weeks = max(len(v) for v in per_person_gains.values())
    print(f"=== [{label}] per-person weekly trajectories ===")
    header = f"{'person':24} " + " ".join(f"W{i+1:>5}" for i in range(weeks))
    print(header + "  trend")
    for pid, gains in per_person_gains.items():
        first_two = sum(gains[:2]) / 2
        last_two = sum(gains[-2:]) / 2
        delta = last_two - first_two
        arrow = "UP" if delta > 0.02 else ("DOWN" if delta < -0.02 else "flat")
        cells = " ".join(f"{g:>6.3f}" for g in gains)
        print(f"{pid:24} {cells}  {arrow:>5} (Δ={delta:+.3f})")

    print(f"\n=== [{label}] cohort weekly average (renormalized gain) ===")
    for w in range(weeks):
        vals = [g[w] for g in per_person_gains.values() if w < len(g)]
        print(f"  Wk{w+1}  avg={sum(vals)/len(vals):.4f}  n={len(vals)}")

    if any(any(v is not None for v in g) for g in per_person_nrgains.values()):
        print(f"\n=== [{label}] cohort weekly average (NON-renorm gain; RL true reward) ===")
        for w in range(weeks):
            vals = [
                g[w]
                for g in per_person_nrgains.values()
                if w < len(g) and g[w] is not None
            ]
            if vals:
                print(f"  Wk{w+1}  avg={sum(vals)/len(vals):.4f}  n={len(vals)}")

    if per_person_placements:
        print(f"\n=== [{label}] per-person placements (of 20) ===")
        for pid, placed in per_person_placements.items():
            first_two = sum(placed[:2]) / 2
            last_two = sum(placed[-2:]) / 2
            delta = last_two - first_two
            cells = " ".join(f"{p:>6d}" for p in placed)
            arrow = "UP" if delta > 1 else ("DOWN" if delta < -1 else "flat")
            print(f"{pid:24} {cells}  {arrow:>5} (Δ={delta:+.1f})")

    if per_person_terminal_reward:
        print(f"\n=== [{label}] per-person terminal reward ===")
        for pid, r in per_person_terminal_reward.items():
            cells = " ".join(f"{v:>6.3f}" for v in r)
            first_two = sum(r[:2]) / 2
            last_two = sum(r[-2:]) / 2
            arrow = "UP" if last_two - first_two > 0.02 else "flat"
            print(f"{pid:24} {cells}  {arrow:>5}")

    upward = sum(
        1
        for g in per_person_gains.values()
        if (sum(g[-2:]) / 2 - sum(g[:2]) / 2) > 0.02
    )
    n = len(per_person_gains)
    print(f"\n[{label}] persons with upward gain trend: {upward}/{n}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "current"))
