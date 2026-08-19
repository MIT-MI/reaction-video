"""T3 judging: score each generated rationale against the human references (text-only).

Cross-family rule (ADR-0001 / revival D5): a family never judges its own generations.
Default routing: OpenAI-family generations are judged by gemini-3.7-flash; everything else
(Gemini, Qwen, Kimi) is judged by gpt-5. Both judges also score a shared overlap slice
(--overlap, default 30 items of every model) so judge-judge agreement is reportable, and the
judge is validated against human anchors separately.

Rubric (STATUS_AND_TASK_PLAN §5 T3): cue coverage, stimulus-reaction linkage, faithfulness.
References are the Prolific annotators' 'ok' rationales (1-3 per item).

    python -m gold_eval.judge_rationale                      # judge every generation file
    python -m gold_eval.judge_rationale --gen_model gpt-5    # just one
"""

from __future__ import annotations

import argparse
import json
import re
import traceback
from pathlib import Path

from .backbones import get_backbone
from .costs import BudgetExceeded, Ledger, slug
from .retry import call_with_retry

HERE = Path(__file__).parent

JUDGE_PROMPT = (
    "You are grading a model-written explanation of why a viewer reacted during a specific "
    "video moment. {n} human annotator(s) watched that moment and independently wrote these "
    "reference explanations:\n{refs}\n\nModel's explanation: \"{hyp}\"\n\n"
    "Score the model's explanation on three dimensions:\n"
    "- cue (0-2): does it identify the same concrete trigger in the content as the "
    "references? 2 = same specific cue as at least one reference; 1 = related or partially "
    "overlapping cue; 0 = different cue or only a generic description.\n"
    "- link (0-2): does it connect that cue to the viewer's reaction? 2 = clear causal "
    "link; 1 = weak or implicit; 0 = no link.\n"
    "- faithful (0-1): 1 = free of invented specifics that contradict the references; "
    "0 = contradicts them.\n"
    'Reply ONLY with a JSON object, e.g. {{"cue": 2, "link": 1, "faithful": 1}}.'
)

DIMS = {"cue": 2, "link": 2, "faithful": 1}


def judge_for(gen_model: str) -> str:
    return "gemini-3.7-flash" if not gen_model.startswith(("tinker", "gemini")) else "gpt-5"


def parse_judgement(text: str) -> dict | None:
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    out = {}
    for k, hi in DIMS.items():
        v = obj.get(k)
        if not isinstance(v, (int, float)) or not 0 <= v <= hi:
            return None
        out[k] = int(v)
    return out


def run_judge(gen_file: Path, judge_model: str, refs: dict, limit: int | None) -> Path:
    gens = {r["candidate_id"]: r for r in
            (json.loads(l) for l in gen_file.read_text().splitlines() if l.strip())
            if r.get("rationale")}
    out = HERE / "results/rationale_judge" / f"{gen_file.stem}__by__{slug(judge_model)}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["candidate_id"] for r in
            (json.loads(l) for l in out.read_text().splitlines() if l.strip())
            if r.get("scores")} if out.exists() else set()
    todo = [cid for cid in gens if cid in refs and cid not in done]
    if limit:
        todo = todo[:limit]
    if not todo:
        return out
    backbone, ledger = get_backbone(judge_model), Ledger(judge_model)
    print(f"[judge] {gen_file.stem} by {judge_model}: {len(todo)} to score")
    for i, cid in enumerate(sorted(todo)):
        try:
            ref_lines = "\n".join(f"  {j+1}. \"{t}\"" for j, t in enumerate(refs[cid]))
            prompt = JUDGE_PROMPT.format(n=len(refs[cid]), refs=ref_lines,
                                         hyp=gens[cid]["rationale"])
            text, tin, tout = call_with_retry(
                backbone.complete, [{"type": "text", "text": prompt}], max_tokens=60)
            ledger.log(f"judge:{cid}", tin, tout)
            row = {"candidate_id": cid, "scores": parse_judgement(text), "raw": text.strip()[:200]}
            with open(out, "a") as f:
                f.write(json.dumps(row) + "\n")
        except BudgetExceeded:
            raise
        except Exception:
            print(f"[judge] FAIL {cid}\n{traceback.format_exc()}", flush=True)
    return out


def aggregate(path: Path) -> dict:
    if not path.exists():
        return {"n": 0}
    rows = [r for r in (json.loads(l) for l in path.read_text().splitlines() if l.strip())]
    dedup = {r["candidate_id"]: r for r in rows}
    scored = [r["scores"] for r in dedup.values() if r.get("scores")]
    if not scored:
        return {"n": 0}
    agg = {k: round(sum(s[k] for s in scored) / len(scored), 3) for k in DIMS}
    agg["composite_0to5"] = round(sum(sum(s.values()) for s in scored) / len(scored), 3)
    agg["n"] = len(scored)
    agg["unparsed"] = len(dedup) - len(scored)
    return agg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gen_model", default=None, help="only this generation model")
    p.add_argument("--judge", default=None, help="override judge model")
    p.add_argument("--overlap", type=int, default=30,
                   help="items also scored by the OTHER judge for agreement")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    refs_raw = json.loads((HERE / "tasks/rationale_refs.json").read_text())
    refs: dict[str, list[str]] = {}
    for r in refs_raw:
        refs.setdefault(r["candidate_id"], []).append(r["rationale"])

    gen_dir = HERE / "results/rationale"
    files = [gen_dir / f"{slug(args.gen_model)}.jsonl"] if args.gen_model \
        else sorted(gen_dir.glob("*.jsonl"))
    report = {}
    for gf in files:
        if not gf.exists():
            print(f"missing: {gf}")
            continue
        primary = args.judge or judge_for(gf.stem.replace("_", ":", 1) if gf.stem.startswith("tinker") else gf.stem)
        out = run_judge(gf, primary, refs, args.limit)
        report[gf.stem] = {f"by_{slug(primary)}": aggregate(out)}
        other = "gpt-5" if primary != "gpt-5" else "gemini-3.7-flash"
        # cross-family rule also applies to the overlap judge: skip when the second judge
        # would share a family with the generator (gemini gens get only the gpt-5 judge;
        # judge-judge agreement is computed on the remaining models)
        same_family = other.split("-")[0] in gf.stem
        if args.overlap and not same_family:
            out2 = run_judge(gf, other, refs, args.overlap)
            report[gf.stem][f"by_{slug(other)}_overlap"] = aggregate(out2)
        print(f"{gf.stem}: {json.dumps(report[gf.stem])}")
    (HERE / "results/rationale_judge/scores.json").write_text(json.dumps(report, indent=2))
    print("-> results/rationale_judge/scores.json")


if __name__ == "__main__":
    main()
