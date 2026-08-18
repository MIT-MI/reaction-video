# gold_eval — ReactionBench gold-set evaluation harness

Evaluates the frozen gold task sets (built in `reaction_video_benchmark_phase0_pilot`) against
the ADR-0003 model roster. Lives in this repo per the dataset repo's CLAUDE.md §9 (no eval code
in the dataset repo).

## Tasks

| Task | Set | Ground truth | Metric |
|---|---|---|---|
| T2 match | `tasks/match_set.json` (150) | temporal alignment (`correct_index`) | accuracy + temporal distance of errors, stratified by `cultural_specificity` |
| T1 ranking | `tasks/ranking_set.json` (250) | Stage-A consensus mean per moment | Spearman ρ vs consensus (per video), human LOO ceiling alongside |
| T3 rationale | (after the Prolific run) | human blind-write rationales | judge rubric (cross-family judge, ADR-0001) |

All models get the **same 1 fps sampled-frame budget**, low-res (max side 512), per ADR-0003.
Task files are frozen and committed; regenerate only via `fetch_tasks.py` and commit the diff.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install tinker tinker-cookbook openai google-genai pillow scipy numpy
```

Env (put in `~/.zshrc`):

- `TINKER_API_KEY` — we keep a pool `TINKER_API_KEY1..5`; export `TINKER_API_KEY=$TINKER_API_KEY1`.
- `OPENAI_API_KEY` — present, but **spend is gated**: runs refuse to start unless
  `OPENAI_SPEND_OK=1` is set (owner confirms with co-author first).
- Gemini goes through **Vertex AI** (~$110 credit left):
  ```bash
  gcloud auth application-default login
  gcloud config set project <PROJECT_ID>
  gcloud services enable aiplatform.googleapis.com
  export GOOGLE_CLOUD_PROJECT=<PROJECT_ID> GOOGLE_CLOUD_LOCATION=global GOOGLE_GENAI_USE_VERTEXAI=true
  ```

## Budget guard

Every API call is appended to `results/costs/<model>.jsonl` with token counts and estimated $.
`costs.py` keeps provider-level running totals across ALL runs and **hard-aborts** at the cap
(gemini $100 of the ~$110 credit; openai $30 until raised). Tinker is grant-billed — tokens are
still logged. Check spend anytime:

```bash
.venv/bin/python -m gold_eval.costs
```

## Run

```bash
# fetch/refresh frozen task sets (match from Supabase, ranking from the dataset repo)
.venv/bin/python -m gold_eval.fetch_tasks

# smoke test one item on the cheapest Tinker model
.venv/bin/python -m gold_eval.run_match --model tinker:Qwen/Qwen3.5-9B --limit 1

# full runs (resumable; predictions in results/<task>/<model>.jsonl)
.venv/bin/python -m gold_eval.run_match   --model tinker:Qwen/Qwen3.6-35B-A3B
.venv/bin/python -m gold_eval.run_ranking --model tinker:Qwen/Qwen3.6-35B-A3B
.venv/bin/python -m gold_eval.run_match   --model gemini-3.7-flash
.venv/bin/python -m gold_eval.run_match   --model gpt-4o          # needs OPENAI_SPEND_OK=1

# score
.venv/bin/python -m gold_eval.score_match
.venv/bin/python -m gold_eval.score_ranking
```

`prescreen_rank`/`prescreen_score` fields in the ranking set are for the Gemini self-selection
analysis ONLY and must never enter a prompt.

## Disconnect / resume (laptop reality)

Everything is designed to be killed at any moment and re-run with the same command:

- **Predictions resume.** Each runner skips ids already present in
  `results/<task>/<model>.jsonl` (rows are appended per item, so a kill loses at most the
  in-flight item).
- **Per-item retry.** Transient errors (network drop, 5xx, rate limit) retry 4× with
  backoff (`retry.py`); an item that still fails is skipped and picked up by the next
  re-run. Budget/spend-gate errors abort immediately and are never retried.
- **Ledger carries over.** Cost totals are recomputed from `results/costs/*.jsonl` at every
  startup, so caps survive restarts.
- **Clip/frame cache** (`data/`) is content-addressed by key; partial downloads are written
  to `.tmp` and renamed, so a killed download never corrupts the cache.

One command re-drives everything (skips finished work):

```bash
bash gold_eval/run_all.sh            # all arms
bash gold_eval/run_all.sh tinker     # only grant-billed arms
caffeinate -is bash gold_eval/run_all.sh   # macOS: keep awake for long runs
```

**Checkpoint to git** after meaningful progress (results + costs are committed, cache is not):

```bash
git add gold_eval/results gold_eval/tasks && git commit -m "eval: checkpoint results" && git push
```

## Migrating to the server (planned ~1 week)

1. Commit + push results from the laptop (above).
2. On the server: `git clone -b gold-eval <repo>`, create the venv per Setup, install ffmpeg.
3. Copy env vars into the server shell rc: `TINKER_API_KEY`, `OPENAI_API_KEY`,
   `GOOGLE_CLOUD_PROJECT/LOCATION/GOOGLE_GENAI_USE_VERTEXAI` (plus
   `gcloud auth application-default login` on the server, or copy
   `~/.config/gcloud/application_default_credentials.json`).
4. `bash gold_eval/run_all.sh` — it resumes exactly where the laptop stopped.
   (Optionally rsync `gold_eval/data/` to skip re-downloading clips; otherwise it refetches
   from R2, which is fine.)
