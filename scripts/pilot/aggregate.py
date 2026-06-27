"""Aggregate frontier-agent pilot results into a markdown table.

Walks the jobs directory, finds every trial's ``results.json``
(produced by ``harbor run``), extracts the verifier reward, and
optionally per-check pytest-ctrf.json results. Emits:

* a markdown table with per-agent solve rate (+ Wilson 95 % CI),
  trial wall time (median, p90), and exception count;
* a per-check failure-mode breakdown across agents (which of the
  28 verifier checks fails most often);
* a cost / token summary if `agent_result` records carry them.

Usage::

    uv run python scripts/pilot/aggregate.py --jobs jobs --out docs/progress/pilot_run.md
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95 % CI for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    halfwidth = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (max(0.0, center - halfwidth), min(1.0, center + halfwidth))


def _find_trial_results(jobs_root: Path) -> dict[str, list[Path]]:
    """Map ``pilot-<agent>__<ts>`` job dir -> list of results.json paths."""
    jobs: dict[str, list[Path]] = {}
    if not jobs_root.is_dir():
        return jobs
    for job_dir in sorted(p for p in jobs_root.iterdir() if p.is_dir()):
        if not job_dir.name.startswith("pilot-"):
            continue
        results = sorted(job_dir.rglob("results.json"))
        if results:
            jobs[job_dir.name] = results
    return jobs


def _load_trial(results_path: Path) -> dict[str, Any]:
    """Read a trial's results.json + optionally parse its verifier dir."""
    with results_path.open() as f:
        data = json.load(f)

    trial_dir = results_path.parent
    rec: dict[str, Any] = {
        "trial_name": data.get("trial_name") or trial_dir.name,
        "started_at": data.get("started_at"),
        "finished_at": data.get("finished_at"),
        "exception": data.get("exception_info"),
        "reward": None,
        "duration_s": None,
        "per_check": {},
    }

    vr = data.get("verifier_result") or {}
    rewards = vr.get("rewards") or {}
    if "reward" in rewards:
        rec["reward"] = int(rewards["reward"])
    elif rewards:
        # Fallback: any reward key
        rec["reward"] = int(next(iter(rewards.values())))

    # Wall time: prefer agent_execution duration; fall back to start/finish.
    agent_exec = data.get("agent_execution") or {}
    rec["duration_s"] = agent_exec.get("duration_sec")
    if rec["duration_s"] is None and rec["started_at"] and rec["finished_at"]:
        try:
            t0 = datetime.fromisoformat(rec["started_at"].rstrip("Z"))
            t1 = datetime.fromisoformat(rec["finished_at"].rstrip("Z"))
            rec["duration_s"] = (t1 - t0).total_seconds()
        except Exception:
            pass

    # Token / cost (single-step trials).
    ar = data.get("agent_result") or {}
    rec["tokens_in"] = ar.get("n_input_tokens")
    rec["tokens_out"] = ar.get("n_output_tokens")
    rec["cost_usd"] = ar.get("cost_usd")

    # Per-check breakdown from pytest-ctrf.json if present.
    ctrf = trial_dir / "verifier" / "pytest-ctrf.json"
    if ctrf.is_file():
        try:
            with ctrf.open() as f:
                ctrf_data = json.load(f)
            tests = (ctrf_data.get("results") or {}).get("tests") or []
            for t in tests:
                name = t.get("name", "")
                status = t.get("status", "")  # "passed" / "failed" / "skipped"
                rec["per_check"][name] = status
        except Exception:
            pass

    return rec


def _agent_short(job_name: str) -> str:
    """`pilot-opus47__20260628-120000` -> `opus47`."""
    head = job_name.removeprefix("pilot-")
    return head.split("__", 1)[0]


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return xs[f]
    return xs[f] * (c - k) + xs[c] * (k - f)


def aggregate(jobs_root: Path) -> dict[str, Any]:
    jobs = _find_trial_results(jobs_root)
    out: dict[str, Any] = {"by_agent": {}, "checks": defaultdict(Counter)}
    for job_name, result_paths in jobs.items():
        agent = _agent_short(job_name)
        records = [_load_trial(p) for p in result_paths]
        n = len(records)
        rewards = [r["reward"] for r in records if r["reward"] is not None]
        n_solve = sum(rewards)
        n_complete = len(rewards)
        n_except = sum(1 for r in records if r["exception"] is not None)
        durations = [r["duration_s"] for r in records if r["duration_s"] is not None]
        costs = [r["cost_usd"] for r in records if r["cost_usd"] is not None]
        ci_lo, ci_hi = _wilson_ci(n_solve, n_complete) if n_complete else (float("nan"),) * 2

        out["by_agent"][agent] = {
            "job_name": job_name,
            "n_trials": n,
            "n_completed": n_complete,
            "n_solved": n_solve,
            "n_exceptions": n_except,
            "solve_rate": (n_solve / n_complete) if n_complete else float("nan"),
            "ci95_lo": ci_lo,
            "ci95_hi": ci_hi,
            "duration_median_s": _percentile(durations, 0.5) if durations else None,
            "duration_p90_s": _percentile(durations, 0.9) if durations else None,
            "cost_median_usd": _percentile(costs, 0.5) if costs else None,
        }

        # Per-check aggregation across this agent's trials.
        for rec in records:
            for name, status in rec["per_check"].items():
                out["checks"][name][f"{agent}:{status}"] += 1

    return out


def render_markdown(agg: dict[str, Any]) -> str:
    by_agent = agg["by_agent"]
    if not by_agent:
        return "# Frontier-agent pilot run\n\n_No pilot jobs found._\n"

    # Aggregate solve rate
    total_solved = sum(a["n_solved"] for a in by_agent.values())
    total_completed = sum(a["n_completed"] for a in by_agent.values())
    agg_rate = total_solved / total_completed if total_completed else float("nan")
    agg_lo, agg_hi = _wilson_ci(total_solved, total_completed)

    lines = [
        "# Frontier-agent pilot run (ADR-0008)",
        "",
        f"Generated by `scripts/pilot/aggregate.py` on "
        f"{datetime.utcnow().isoformat(timespec='seconds')}Z.",
        "",
        "## Per-agent solve rate",
        "",
        "| Agent | trials | completed | solved | solve rate | 95 % CI | exception |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for agent, a in sorted(by_agent.items()):
        rate = a["solve_rate"]
        lines.append(
            f"| `{agent}` | {a['n_trials']} | {a['n_completed']} | {a['n_solved']} | "
            f"{rate*100:.1f} % | [{a['ci95_lo']*100:.1f}, {a['ci95_hi']*100:.1f}] | "
            f"{a['n_exceptions']} |"
        )
    lines += [
        f"| **aggregate** | {sum(a['n_trials'] for a in by_agent.values())} | "
        f"{total_completed} | {total_solved} | "
        f"**{agg_rate*100:.1f} %** | [{agg_lo*100:.1f}, {agg_hi*100:.1f}] | — |",
        "",
        "ADR-0008 acceptance band: **10–20 %** aggregate. "
        + (
            "✓ within band." if 0.10 <= agg_rate <= 0.20
            else f"⚠ {'BELOW' if agg_rate < 0.10 else 'ABOVE'} band — reconsider case design."
        ),
        "",
        "## Wall time and cost (per trial)",
        "",
        "| Agent | wall median | wall p90 | median cost |",
        "| --- | ---: | ---: | ---: |",
    ]
    for agent, a in sorted(by_agent.items()):
        med = a["duration_median_s"]
        p90 = a["duration_p90_s"]
        cost = a["cost_median_usd"]
        lines.append(
            f"| `{agent}` | "
            f"{f'{med:.0f} s' if med is not None else '—'} | "
            f"{f'{p90:.0f} s' if p90 is not None else '—'} | "
            f"{f'${cost:.2f}' if cost is not None else '—'} |"
        )

    # Per-check failure breakdown if we have pytest-ctrf.json data.
    if agg["checks"]:
        lines += [
            "",
            "## Per-check failure modes",
            "",
            "Counts how often each verifier check failed across all trials of each agent.",
            "",
            "| Verifier check |"
            + "".join(f" {a} pass / fail |" for a in sorted(by_agent)),
            "| --- |" + " ---: |" * len(by_agent),
        ]
        for check_name in sorted(agg["checks"]):
            row = [f"| `{check_name}` |"]
            for agent in sorted(by_agent):
                p = agg["checks"][check_name].get(f"{agent}:passed", 0)
                f = agg["checks"][check_name].get(f"{agent}:failed", 0)
                row.append(f" {p} / {f} |")
            lines.append("".join(row))

    lines += ["", "## Notes", "", "_Add narrative observations here before committing._", ""]
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate frontier-agent pilot results.")
    p.add_argument("--jobs", type=Path, default=Path("jobs"),
                   help="Harbor jobs root directory (default: jobs/)")
    p.add_argument("--out", type=Path, default=Path("docs/progress/pilot_run.md"),
                   help="Output markdown file (default: docs/progress/pilot_run.md)")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    agg = aggregate(args.jobs)
    md = render_markdown(agg)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md)
    n_agents = len(agg["by_agent"])
    n_trials = sum(a["n_trials"] for a in agg["by_agent"].values())
    print(f"Wrote {args.out} ({n_agents} agents, {n_trials} trials).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
