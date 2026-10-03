# Plan

Deadline: **6 October 2026, 23:59 CEST** (TabPFN-3.5 Hackathon). The repo must be public and
Apache-2.0 at submission time.

Judging: 50% showcase of TabPFN-3.5, 30% creativity and practical value, 20% technical quality
and reproducibility.

## Architecture

```text
question + database
      |
      v
+-------------+    typed questions     +---------------------------+
| schema      | ---------------------> | decision model            |
| profiler    |                        | (Ollaya /v1/systemone or  |
+-------------+                        |  Jev)                     |
      |                                +---------------------------+
      v                                      ^          ^
+-------------+   slot answers (probs)       |          |
| formulate   | -----------------------------+          |
| grammar +   |   top-k readings                         |
| beam search |                                          |
+-------------+                                          |
      |  compile (deterministic)                         |
      v                                                  |
+-------------+   leak prior per column                  |
| audit       | -----------------------------------------+
| (prior +    |
|  evidence)  | <---- univariate / with-vs-without evidence
+-------------+
      |
      v
+-------------+       RelArena task.yaml + db.yaml
| learn       | ---->  TabPFN-Rel (local, TabPFN-3.5 on GPU)
| (TabPFN)    |
+-------------+
      |
      v
+-------------+
| answer      | -> probabilities / distributions, backtest vs baselines,
| + clarify   |    clarifying question (only if readings disagree), precedents
+-------------+
      |
      +--> Python / CLI
      +--> HTTP: /v1/ask, /v1/systemone (Jev wire format)
      +--> MCP: ask, clarify, predict, audit, explain
```

### Modules (`src/posterior/`)

| Module | Job |
|---|---|
| `decider.py` | Client for any `/v1/systemone` endpoint (Ollaya, Jev). Typed `Choice` / `Noul` / `Score` helpers, caching, multi-model voting. |
| `schema.py` | Load CSV / Parquet into DuckDB. Profile tables: primary keys (unique, non-null), foreign keys (value inclusion), time columns, sample values. Ask the decision model only for ambiguous cases. Emit RelArena `db.yaml`. |
| `spec.py` | The predictive task grammar: `entity`, `event` (with time-column join path), `op` (exists, not_exists, count, sum, avg, any_where), `filter`, `horizon`, `population` (all, active_backward, active_forward). |
| `formulate.py` | Slot-by-slot typed questions, beam over readings, joint probabilities. |
| `compile.py` | Spec to RelArena label SQL + `task.yaml`. Deterministic. Valid by construction. |
| `audit.py` | Leak score per feature column: decision-model prior combined with empirical evidence. Drops columns from the `db.yaml` allow-list. |
| `learn.py` | Fit and predict through TabPFN-Rel (`local`), backtest against constant baselines. |
| `clarify.py` | Compare top readings on a recent anchor (label agreement). Ask one choice question on the first differing slot only if they disagree. |
| `engine.py` | `Posterior.connect(...).ask(...)`: ties everything together, returns one result object. |
| `api.py` | FastAPI: `/v1/ask`, `/v1/tasks/{id}`, `/v1/tasks/{id}/clarify`, `/v1/systemone`. |
| `mcp.py` | MCP server with the same capabilities. |
| `cli.py` | `posterior ask`, `posterior serve`, `posterior mcp`. |

## Evaluation (numbers for the judges)

1. **Formulation accuracy against gold tasks.** RelArena ships the 21 RelBench v1 tasks as
   hand-written `task.yaml` files. Input: RelBench's own one-line task description (not written
   by us). Metrics:
   - grammar coverage
   - slot accuracy
   - label agreement between the gold SQL and our compiled SQL on the same anchors
   - AUC with the gold spec vs AUC with ours
2. **Leakage detection.** Inject leaking columns, for example a status or last-activity snapshot
   computed from the future. Compare precision and recall for the decision model alone, the
   evidence alone, and the combination. Report AUC before and after the audit.
3. **End to end.** Olist seller churn from one English sentence vs the hand-written cookbook spec
   (0.79 AUC).
4. **Decision model backends.** Formulation accuracy for `winnow:e4b`, `clef`, `kev:9b`, `laya`
   and the vote of several.

## Milestones

### Day 1 (4 Oct)
- [x] Private repo, README, plan
- [x] white (RTX 3090): uv, Ollaya with `winnow:e4b` + `clef`, `tabpfn-rel[local]` (local weights wait for the license; hosted API used)
- [x] Data: RelBench f1, event, hm, trial materialized; Olist from a public Hugging Face mirror
- [x] `decider.py` + `schema.py` (profile, keys, time columns, `db.yaml`)
- [x] `spec.py` reproduces the gold Olist label SQL exactly (6136/6136)

### Day 2 (5 Oct)
- [x] `formulate.py`: question to top-k specs
- [x] `learn.py`: TabPFN-Rel fit, predict, backtest, live refit
- [x] `audit.py` with leak injection experiment (4/4, AUC 1.000 to 0.774)
- [x] `clarify.py`
- [x] Formulation eval over 11 RelBench tasks (f1, event, hm, trial)

### Day 3 (6 Oct)
- [x] `api.py` (`/v1/ask`, `/v1/systemone`), `mcp_server.py`, `cli.py`, 24 tests
- [x] README results tables and reproduction commands
- [ ] Demo video
- [ ] Make repo public, submit by ~20:00 CEST, keep updating until close (Mert)

### Stretch
- Text probes on text columns (decision-model features with a placebo control)
- Decoder readout precedents per prediction
- Cost-aware decisions (`decide` with a cost matrix)
- Multi-model voting in formulation

## Needs from Mert
- Accept the TabPFN-3.5 license at ux.priorlabs.ai (Licenses tab) so local GPU weights download; the token works for the hosted API already
- Join the hackathon on platform.priorlabs.ai and request the extra API credits
- Kaggle credentials (optional, only for Olist)

## Risks
- TabPFN-Rel is 0.0.1 and DFS can be slow on wide schemas. Fallback: prune schemas, cache, or
  our own DuckDB featurizer feeding TabPFN-3.5 directly.
- The grammar will not cover every RelBench task. Report coverage honestly.
- Large RelBench datasets may not fit the time budget. Start with `rel-f1` and Olist.
