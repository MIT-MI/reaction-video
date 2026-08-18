"""Per-item retry with backoff, for laptop runs that lose network mid-batch.

Semantics: transient errors (network, 5xx, rate limits) are retried a few times with
exponential backoff; if an item still fails it is SKIPPED (not written), the run moves on,
and a plain re-run of the same command picks the item up again — the runners resume from
the ids already present in the predictions file. BudgetExceeded and SpendNotConfirmed are
never retried.
"""

from __future__ import annotations

import time
import traceback

from .backbones import SpendNotConfirmed
from .costs import BudgetExceeded

ATTEMPTS = 4
BACKOFF_S = 15.0


def call_with_retry(fn, *args, **kwargs):
    for attempt in range(1, ATTEMPTS + 1):
        try:
            return fn(*args, **kwargs)
        except (BudgetExceeded, SpendNotConfirmed):
            raise
        except Exception:
            if attempt == ATTEMPTS:
                raise
            wait = BACKOFF_S * 2 ** (attempt - 1)
            print(f"[retry] attempt {attempt}/{ATTEMPTS} failed, sleeping {wait:.0f}s\n"
                  f"{traceback.format_exc(limit=2)}", flush=True)
            time.sleep(wait)
