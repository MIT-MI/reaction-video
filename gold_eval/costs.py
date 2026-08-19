"""Spend ledger + hard budget guard.

Every backbone call is logged to results/costs/<model_slug>.jsonl (tokens, estimated $, item id).
Provider totals are computed across ALL ledger files at startup and updated per call; crossing a
provider cap raises BudgetExceeded and the run stops. Tinker is grant-billed (price 0) but tokens
are still recorded so grant burn is visible.

    python -m gold_eval.costs        # print current totals per provider/model
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

RESULTS = Path(__file__).parent / "results"
COSTS_DIR = RESULTS / "costs"

# USD per 1M tokens (input, output). Verify against live pricing pages before big runs.
# Tinker rates are (prefill, sample) from tinker-docs models page (2026-08) — grant-billed,
# tracked so grant burn is visible; the tinker cap stays infinite.
PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),           # legacy grandfathered rate
    "gpt-5": (1.25, 10.00),
    "gemini-3.7-flash": (0.75, 3.75),  # intro rate until 2026-12-31, then 1.50/7.50
    "gemini-3.1-pro-preview": (2.00, 12.00),
    "tinker:Qwen/Qwen3.5-9B": (0.66, 1.995),
    "tinker:Qwen/Qwen3.6-27B": (1.86, 5.595),
    "tinker:Qwen/Qwen3.6-35B-A3B": (0.54, 1.335),
    "tinker:Qwen/Qwen3.5-397B-A17B": (3.00, 7.50),
    "tinker:moonshotai/Kimi-K2.6": (2.205, 5.49),
    "tinker:thinkingmachines/Inkling": (1.87, 4.68),      # 50%-off promo rate
    "tinker:thinkingmachines/Inkling-Small": (0.60, 1.44),  # serverless beta 1.20*, std 2.88
}

# Hard abort thresholds per provider (cumulative across all runs).
PROVIDER_CAPS: dict[str, float] = {
    "gemini": 330.0,   # $36.8 spent on account 1 (exhausted); account 2 (2026-08-19) has $300
                       # expiring ~2026-09-08 — owner: "可以随便用". cap = 36.8 + ~293.
    "openai": 30.0,    # GPT-5 approved (owner, 2026-08-18)
    "tinker": float("inf"),
}


class BudgetExceeded(RuntimeError):
    pass


def provider_of(model: str) -> str:
    if model.startswith("tinker:"):
        return "tinker"
    if model.startswith("gemini"):
        return "gemini"
    return "openai"


def slug(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def _iter_entries():
    if COSTS_DIR.is_dir():
        for f in sorted(COSTS_DIR.glob("*.jsonl")):
            for line in f.read_text().splitlines():
                if line.strip():
                    yield json.loads(line)


def provider_total(provider: str) -> float:
    return sum(e["usd"] for e in _iter_entries() if e.get("provider") == provider)


class Ledger:
    def __init__(self, model: str):
        self.model = model
        self.provider = provider_of(model)
        self.price_in, self.price_out = PRICES.get(model, (0.0, 0.0))
        self.cap = PROVIDER_CAPS.get(self.provider, 0.0)
        self.path = COSTS_DIR / f"{slug(model)}.jsonl"
        COSTS_DIR.mkdir(parents=True, exist_ok=True)
        self.provider_spent = provider_total(self.provider)

    def log(self, item_id: str, in_tok: int, out_tok: int) -> float:
        usd = (in_tok * self.price_in + out_tok * self.price_out) / 1e6
        entry = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "model": self.model, "provider": self.provider, "item": item_id,
                 "in_tok": in_tok, "out_tok": out_tok, "usd": round(usd, 6)}
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        self.provider_spent += usd
        if self.provider_spent >= self.cap:
            raise BudgetExceeded(
                f"{self.provider} cumulative spend ${self.provider_spent:.2f} >= cap ${self.cap:.2f}")
        return usd


def main() -> None:
    by_model: dict[str, dict] = {}
    for e in _iter_entries():
        d = by_model.setdefault(e["model"], {"usd": 0.0, "in_tok": 0, "out_tok": 0, "calls": 0})
        usd = e["usd"]
        if not usd and e["model"] in PRICES:  # rows logged before the price was known
            pi, po = PRICES[e["model"]]
            usd = (e["in_tok"] * pi + e["out_tok"] * po) / 1e6
        d["usd"] += usd; d["in_tok"] += e["in_tok"]; d["out_tok"] += e["out_tok"]; d["calls"] += 1
    if not by_model:
        print("no spend recorded yet")
        return
    by_provider: dict[str, float] = {}
    for m, d in sorted(by_model.items()):
        p = provider_of(m)
        by_provider[p] = by_provider.get(p, 0.0) + d["usd"]
        print(f"{m:40s} calls={d['calls']:5d} in={d['in_tok']:>10,} out={d['out_tok']:>8,} ${d['usd']:.3f}")
    for p, usd in sorted(by_provider.items()):
        cap = PROVIDER_CAPS.get(p, 0.0)
        print(f"[total] {p:10s} ${usd:.3f} / cap ${cap}")


if __name__ == "__main__":
    main()
