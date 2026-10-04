[posterior-summary-en.md](https://github.com/user-attachments/files/33035463/posterior-summary-en.md)
# Summary for Kerem: Posterior

> This file is an internal team note. It contains internal machine names and paths. **Delete or move it
> before making the repo public.** All dates are October 2026.

## In one sentence

Posterior lets you ask a database a plain question about the future ("Which sellers will stop selling in
the next 30 days?"). It returns a calibrated probability for each entity, and shows how it understood the
question and which columns it did not use because of leakage.

- The data scientist's decisions are made by a Jev-style **decision model** (the open model `winnow:e4b`, via Ollaya).
- The statistics are done by **TabPFN-3.5** (via TabPFN-Rel).
- No model writes SQL. A small, typed grammar turns the decisions into a RelArena task.

**Hackathon:** Prior Labs TabPFN-3.5 Hackathon, https://platform.priorlabs.ai/hackathon-3.5

**Deadline: October 6, 23:59 CEST** (October 7, 00:59 Turkey time).

Scoring:
- 50%: showcasing TabPFN-3.5
- 30%: creativity
- 20%: technical quality and reproducibility

## Why this idea

TabPFN brought the model-training step down to seconds. But the steps before it are still done by hand:
- defining the question (target, window, which entities are at risk),
- joining the tables,
- weeding out columns that are not known at prediction time.

Prior Labs states this explicitly in its own RelArena documentation: "What even is a predictive task over a
relational database? How to answer this well remains an open problem."

The decisions in these steps are all small, typed questions: a choice or a yes/no. Decision models do
exactly this, in milliseconds and with a probability. TabPFN, on the other hand, never sees column names;
it knows nothing about meaning but is very strong at statistics. Each one covers the other's gap.

## What is where

- **Repo:** https://github.com/cobanov/posterior (currently private)
- **Presentation page:** https://claude.ai/artifact/9pzB6pMDBsPE1NE2fdegkU (anyone with the link can open it). The copy in the repo is `docs/index.html`.
- **README.md:** Setup, usage, results tables, limitations. The jury will read this; it is in English.
- **PLAN.md:** The initial plan and what has been finished.
- **Code (`src/posterior/`):**

  | File | Role |
  |---|---|
  | `schema.py` | Loads the database into DuckDB; finds keys, links and the event time |
  | `formulate.py` | Splits the question into slots and asks the decision model about each slot |
  | `spec.py` | Grammar and SQL compiler |
  | `clarify.py` | Produces a single clarifying question if the readings conflict |
  | `audit.py` | Leakage audit |
  | `learn.py` | Learning, backtest and live prediction with TabPFN-Rel |
  | `engine.py` | Ties everything together |
  | `cli.py`, `api.py`, `mcp_server.py` | Interfaces |

- **Evaluations:**
  - `eval/relbench_formulation.py`
  - `eval/leakage.py`
  - Raw results are in `eval/results/`.
- **Tests:** 24 offline tests under `tests/`. No model server required.

## Infrastructure (Mert's environment)

- **Work machine `white` (RTX 3090):**
  - Ollaya 0.9.0 runs as a systemd service (`127.0.0.1:11435`); `winnow:e4b`, `clef` and `laya` are pulled.
  - Code is in `~/Developer/posterior`, data is under `~/data/olist` and `~/data/relbench/{rel-f1,rel-event,rel-hm,rel-trial}`.
- **TabPFN:**
  - Runs through the hosted API, with a Prior Labs API key (`POSTERIOR_TABPFN=client`).
  - For a local GPU, the account must have accepted the TabPFN-3.5 license (ux.priorlabs.ai, Licenses tab). This has not been done yet.
  - The key is personal; use your own key on your own machine.
- **4090 and 5090:** Being used for other training runs (earth-dreams LoRA); not used in this project.

## Running it on your own machine

```sh
git clone https://github.com/cobanov/posterior && cd posterior
uv sync --extra api --extra serve --group dev       # for local TabPFN with a GPU: --extra local
uv run pytest                                       # 24 tests, no model required

# Decision model: Ollaya (https://github.com/ollaya-dev/ollaya)
curl -fsSL https://ollaya.dev/install.sh | sh && ollaya pull winnow:e4b
# If Ollaya is on another machine: export POSTERIOR_DECIDER_URL=http://<host>:11435

# TabPFN key (ux.priorlabs.ai)
mkdir -p ~/.config/posterior && echo "TABPFN_TOKEN=<your own key>" > ~/.config/posterior/env

./scripts/get_olist.sh data/olist                   # Olist, no Kaggle account needed
uv run posterior formulate --db data/olist "Which sellers will stop selling in the next 30 days?"
uv run posterior ask       --db data/olist "Which sellers will stop selling in the next 30 days?"
uv run posterior serve     --db data/olist --port 8787     # HTTP API
uv run python scripts/mcp_smoke.py data/olist              # tries MCP with a real client
```

Notes:
- One `ask` run takes about 2.5 minutes: two TabPFN-Rel fits, one for the backtest and one for the live prediction.
- When the same question is asked again, the result comes from the `runs/` folder.
- Decision model answers are stored in `cache/decisions.jsonl`.

## Latest findings (measured)

| Measurement | Result |
|---|---|
| Olist, seller churn from a single English sentence, with the leakage audit | ROC-AUC **0.777-0.780** |
| Comparison | Constant prediction 0.500, the seller's own history 0.670, Prior Labs' hand-written task (our run) 0.776 |
| Compiler correctness | Same label as Prior Labs' hand-written SQL on 6,136 of 6,136 rows |
| 4 columns that know the future were injected into Olist | Meaning alone caught 3/4, data alone (TabPFN) caught 2/4, **the two together caught 4/4**. None of the normal columns were dropped. 88% agreement with Prior Labs' hand-picked column list. |
| TabPFN-3.5 on leaky data | Without the audit **1.000** (it sees the future), with the audit **0.774** |
| RelBench, 11 tasks | Answer type and window 11/11, entity 9/11. H&M's two tasks exactly right on the first reading, F1 top-3 exactly right with one clarification. On the 5 tasks the grammar can express, an average label agreement of 0.67 with clarification. |
| Single Jev-compatible request | "Will the seller stop" 0.717 from TabPFN, "is the note a complaint" 0.983 from the decision model |

What we learned:

1. **Ask the decision model atomic questions, with only the question sentence.** Putting the schema into
   the state corrupts the answer. The "stops selling" question went from 0.37 to 0.89. On RelBench, entity
   selection went from 4/6 to 6/6.
2. **Data can confirm leakage but cannot clear a column.** Olist was exported after the deliveries had
   finished; there are no empty delivery dates in the newer rows. The delivery dates were caught only by
   the meaning evidence.
3. **A time column in a keyed table should not automatically be treated as the event time.** When "last
   order date" was mistaken for the event time of the sellers table, the audit went blind. 0/4 of the
   injected leaks were caught; after the fix, 4/4.
4. **The clarifying question is a safety net.** Example: for the question "How much revenue will each
   product bring in next month?", the readings "all products" and "products that have sales" agree on only
   12% of the labels. Instead of silently guessing, the system asks a single question.

Weak points (also written in the README):
- **What falls outside the grammar:** most of the rel-event and rel-trial tasks require these:
  - set filter (`status IN (...)`),
  - two-step link,
  - value from another table,
  - table union,
  - window function.
- **Datasets that were not run:** rel-amazon, rel-stack and rel-avito were not run on the machine with 16 GB of RAM. hope (47 GB RAM) is better suited for this.
- **The borderline column:** The leakage score of the estimated delivery date (~0.77) is very close to the 0.75 threshold, so the result fluctuates between 0.777 and 0.780 from run to run.

## To do

In order of priority:

- [ ] **ux.priorlabs.ai → Licenses:** Accept the TabPFN-3.5 license. The account owner has to do it. After that, switch to the local GPU with `POSTERIOR_TABPFN=local`.
- [ ] **Join the hackathon** (platform.priorlabs.ai) and ask for additional API credit.
- [ ] **Demo video:** Suggested flow:
  1. prediction from a single sentence with `posterior ask`,
  2. the leakage experiment (1.000 → 0.774),
  3. the Jev-compatible mixed request,
  4. MCP inside Claude Code.
- [ ] **Remove this file, make the repo public, submit.** Submission form: repo link (Apache-2.0), project description, optional video link.
- [ ] **Optional, if there is time left:**
  - Add the set filter and the two-step link to the grammar; this raises rel-event.
  - Run the rel-amazon evaluation on hope.

---

## Claude's last message (verbatim)

Yes, it works. This morning on white I cloned it from GitHub from scratch, set it up, and re-verified everything:

| Check | Result |
|---|---|
| Tests | 24/24 passed |
| CLI | `schema` and `formulate` work; the question is read correctly (0.57) |
| HTTP API | `/health` OK; in the Jev-compatible request, two questions went to the decision model (complaint 0.98, refund 0.99) |
| End-to-end run | Finished in 167 seconds, **ROC-AUC 0.777**, produced predictions for 1,278 sellers |

I learned something during the verification: results can fluctuate between 0.777 and 0.780 from run to run. The reason is that the leakage score of the "estimated delivery date" column (~0.77) is just above the 0.75 threshold. Since a clean clone has no stored decision model answers, this column was dropped this time. I stated this explicitly both in the README and on the page.

**The presentation page is up to date (version 4):** https://claude.ai/artifact/9pzB6pMDBsPE1NE2fdegkU
- **Real results:** The numbers I updated overnight are in place. There are no longer any "illustrative" values on the page.
- **New sections:**
  - **What happened overnight:** 8 steps. Among them are two bugs that later turned into the most useful findings: the leakage audit that dropped no columns, and the "last order date" bug that blinded the audit.
  - **Findings:** 4 items, each backed by a measurement.
  - **What's next.**
- **Mobile layout:**
  - On a phone, the comparison table turns into one card per step instead of scrolling horizontally.
  - Probability labels are no longer cut off; they drop down above their bars.
  - The leakage table stacks vertically on small screens.
  - Headings and touch targets have been adjusted for small screens.
  - I measured at widths of 320, 360, 390 and 430 pixels; there is no horizontal overflow. During the measurement I found that long risk labels overflowed to 439 pixels, and fixed it.

The changes have been pushed; the copy of the page in the repo is at `docs/index.html`. I added the morning verification line to the Obsidian note.

What's expected from you is the same: accepting the TabPFN-3.5 license (for the local GPU), hackathon registration, the demo video, and making the repo public and submitting. The deadline is October 6, 23:59 CEST.
