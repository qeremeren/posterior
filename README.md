# Posterior

**Ask a database a question about the future. Get a calibrated answer, and see how the question was read.**

Built for the [Prior Labs TabPFN-3.5 Hackathon](https://platform.priorlabs.ai/hackathon-3.5), October 2026.
**Demo video (3:23):** [Predict the future with Posterior](https://www.youtube.com/watch?v=iodHvixfuaA)

```text
$ posterior ask --db data/olist "Which sellers will stop selling in the next 30 days?"

Readings
  → 0.56  For each sellers row with a order_items event in the last 30 days: no order_items event in the next 30 days
    0.21  For each sellers row with a order_items event in the last 30 days: the number of order_items events in the next 30 days
    0.12  For every row of sellers: no order_items event in the next 30 days

Leakage audit: kept 25 columns, dropped 4
    - orders.order_status: meaning: updated later with p=0.80; newest rows differ: 0.02
    - orders.order_delivered_customer_date: meaning: updated later with p=0.85; newest rows differ: 0.01
    ...

Backtest (client, 1254 rows at 2018-08-01 00:00:00)
    model     positive_rate=0.277  brier=0.162  roc_auc=0.779
    constant_global positive_rate=0.277  brier=0.200  roc_auc=0.500
    constant_per_entity positive_rate=0.277  brier=0.216  roc_auc=0.670

Predictions (top 10 of 1278)
    seller_id=56db5c0782e8f7ddc9343f9576ff6d16  timestamp=2018-08-31T00:00:00.000  prediction=0.8337168694
    ...

done in 279.1s
```

This is the run from the demo video, on an M5 MacBook.

From that one sentence, Posterior:

- read "stop selling" as *no order items for 30 days, among sellers active in the last 30 days*, which is
  the task Prior Labs wrote by hand for this dataset;
- dropped the order columns that are filled in after the purchase;
- fitted TabPFN-3.5 in context and scored all 1,278 active sellers.

No training loop, no feature engineering, no hand-written SQL.

## Why

Turning a business question into a prediction takes a data scientist days to weeks: define the question,
find and join the tables, remove the columns that leak the future, train, read the output.
[TabPFN](https://github.com/PriorLabs/tabpfn) made training take seconds. The steps before it are still
manual, and [Prior Labs' RelArena docs](https://github.com/PriorLabs/relarena/blob/main/docs/predictive-task.md)
say so: *"What even is a predictive task over a relational database? How to answer this well remains an
open problem."*

Those steps are small, typed judgment calls (which table, which event, which window, was this column known
then?). Decision models such as Winnow (served by [Ollaya](https://github.com/ollaya-dev/ollaya)) answer
exactly that kind of question, with probabilities, but can't do statistics. TabPFN does the statistics but
never sees what a column means. Posterior gives each the job it is good at, and a small typed grammar turns
the decisions into a RelArena task, so no model ever writes SQL.

## How it works

| Step | What happens | Who |
|---|---|---|
| **Profile** (`schema.py`) | Find keys, links and event times. Ask only when the data can't settle it. | code, decision model |
| **Formulate** (`formulate.py`) | Split the question into typed slots (entity, event, operation, filter, window, population) and rank the readings. | decision model |
| **Compile** (`spec.py`) | Turn the best reading into RelArena label SQL and a `task.yaml`. | code |
| **Clarify** (`clarify.py`) | Label past anchors with the top readings. If they disagree, ask the user one multiple-choice question. | code |
| **Audit** (`audit.py`) | Per column: *meaning* (decision model: "filled in or updated later?") and *data* (newest-row drift for event tables; lift over the anchor time for the entity table). | decision model, TabPFN* |
| **Learn** (`learn.py`) | TabPFN-Rel replays history at many anchors, TabPFN-3.5 learns in context; backtest against constant baselines, then refit for live predictions. | TabPFN-3.5 |

\* The lift check uses TabPFN when `TABPFN_TOKEN` is exported in the shell, and gradient boosting
otherwise (the key file alone is read too late for it).

## Results

All numbers come from runs in this repository. Decision model `winnow:e4b` on Ollaya; TabPFN-3.5 through
TabPFN-Rel on the Prior Labs API.

**End to end on Olist (seller churn, from one sentence)**

| | ROC-AUC | Brier |
|---|---|---|
| Posterior, with the leakage audit (test cutoff 2018-08-01) | **0.777–0.780** | 0.162 |
| Constant per seller (each seller's own history) | 0.670 | 0.216 |
| Constant (global) | 0.500 | 0.200 |
| Prior Labs' hand-written TabPFN-Rel task, our rerun (their splits, cutoff 2018-06-15) | 0.776 | |

- The compiled label SQL gives the same label as Prior Labs' hand-written query on all 6,136 (anchor, seller)
  rows across six anchors.
- Runs land between 0.777 and 0.780 because one column, the estimated delivery date, sits near the audit's
  0.75 threshold. The demo-video run scored 0.779.
- Keeping every column scores 0.787: that 0.007 is the leak the audit removes.

**Leakage audit, with four planted leaks** (`eval/leakage.py`: last order date, active status, lifetime item
count and lifetime revenue, all computed from the full history)

| Audit | Planted leaks caught | Ordinary seller columns dropped | Agreement with Prior Labs' allow-list |
|---|---|---|---|
| Meaning only | 3 / 4 | 0 | 84% |
| Data only | 2 / 4 | 0 | 68% |
| **Both** | **4 / 4** | **0** | **88%** |

On the leaky database TabPFN-3.5 scores a perfect **1.000** without the audit (it reads the future) and an
honest **0.774** with it. The data check in `eval/results/leakage.json` ran on the gradient-boosting fallback. With TabPFN, the data-only check scored 4/4.

**Formulation on 11 RelBench tasks**, from RelBench's own one-line task descriptions, scored against Prior
Labs' hand-written RelArena task files: answer type and window right on 11/11, entity on 9/11. Both H&M
tasks are exact on the first reading, F1 top-3 is exact after one clarification, and the 5 tasks the grammar
can state reach 0.67 mean label agreement. Per-task results: `eval/results/relbench_formulation_*.json`.

## Reproduce

Tested on an Apple-silicon Mac (macOS 26) and on Linux with an RTX 3090.

```sh
git clone https://github.com/qeremeren/posterior && cd posterior
uv sync --extra api --extra serve            # Python 3.11 or 3.12; add --extra local for a GPU
./scripts/get_olist.sh data/olist            # public mirror of the Olist tables, no Kaggle account

curl -fsSL https://ollaya.dev/install.sh | sh   # decision model server: macOS 14+ on Apple silicon, or Linux
ollaya pull winnow:e4b                          # 8 GB

export TABPFN_TOKEN=<your key>               # from ux.priorlabs.ai

uv run pytest                                # 24 offline tests, about 30 s, no models needed
uv run posterior formulate --db data/olist "Which sellers will stop selling in the next 30 days?"
uv run posterior ask       --db data/olist "Which sellers will stop selling in the next 30 days?"
```

`ask` takes about 2.5 minutes with Ollaya on an RTX 3090 and about 5 minutes on an M5 MacBook. Asking
the same question again reuses `runs/`; decision-model answers are cached in `cache/decisions.jsonl`.
If Ollaya runs on another machine, set `POSTERIOR_DECIDER_URL=http://<host>:11435`.

The RelBench evaluation reads four RelBench databases, downloaded from relbench.stanford.edu (no account):

```sh
uv run python -c "
from pathlib import Path; from relbench.datasets import get_dataset
for d in ['rel-f1', 'rel-event', 'rel-hm', 'rel-trial']:
    out = Path('~/data/relbench').expanduser() / d; out.mkdir(parents=True, exist_ok=True)
    for name, t in get_dataset(d, download=True).get_db(upto_test_timestamp=False).table_dict.items():
        t.df.to_parquet(out / f'{name}.parquet', index=False)
    (out / '.done').touch()"
```

The evaluations:

```sh
uv run python eval/leakage.py --olist data/olist --fit              # leakage tables
uv run python eval/relbench_formulation.py --data ~/data/relbench   # RelBench table
uv run python scripts/mcp_smoke.py data/olist                       # MCP over stdio
```

## Use it from code, HTTP or an agent

`--db` takes a folder of CSV or Parquet files, or a DuckDB file.

**Python**

```python
from posterior.engine import Posterior

post = Posterior.connect("data/olist")
res = post.ask("Which sellers will stop selling in the next 30 days?")
res.learn["metrics"]["roc_auc"]   # backtest
res.predictions.head()            # live predictions, highest first
```

**HTTP**, `uv run posterior serve --db data/olist --port 8787`:

| Endpoint | |
|---|---|
| `POST /v1/formulate` | Readings and clarification, no model fit |
| `POST /v1/ask` | Readings, audit, backtest and predictions |
| `POST /v1/tasks`, `GET /v1/tasks/{id}` | The same, in the background |
| `POST /v1/systemone` (alias `/v1/decisions`) | Jev wire format |
| `GET /v1/schema`, `GET /health` | Schema and status |

`/v1/systemone` speaks TypeSafe's Jev format, so existing Jev integrations can point at Posterior unchanged.
A yes/no question about the future of an entity in `state` is answered by TabPFN from the database's
history; every other question goes to the decision model. In the demo run, the request below returned
`churn` 0.718 from TabPFN and `upset` 0.772 from the decision model:

```sh
curl -s localhost:8787/v1/systemone -H 'content-type: application/json' -d '{
  "model": "posterior",
  "state": {"seller_id": "58b98ccb79873e04eac4357cacc590d9", "note": "two late shipments this week"},
  "questions": {
    "churn": {"type": "noul", "instructions": "Will this seller stop selling in the next 30 days?"},
    "upset": {"type": "noul", "instructions": "Does the note sound like a complaint?"}}}' | python3 -m json.tool
```

**MCP**, for Claude Code or any MCP client. Tools: `describe_database`, `formulate`, `ask`, `predict`,
`audit`.

```sh
claude mcp add posterior -e TABPFN_TOKEN=<your key> -- uv run --directory /path/to/posterior posterior mcp --db data/olist
```

**Configuration**

| Variable | Default | |
|---|---|---|
| `POSTERIOR_DECIDER_URL` | `http://127.0.0.1:11435` | Any `/v1/systemone` endpoint (Ollaya, Jev, OpenRouter) |
| `POSTERIOR_DECIDER_MODEL` | `winnow:e4b` | `a+b` averages two models |
| `POSTERIOR_TABPFN` | `client` | `client` (Prior Labs API) or `local` (GPU, needs the TabPFN-3.5 license) |
| `TABPFN_TOKEN` | | Export it; `~/.config/posterior/env` also works for the fits |

## What we learned

- **Ask decision models atomic questions, on the wording alone.** Splitting "what does it predict?" into
  smaller choices, and leaving the schema out of questions about wording, moved "stop selling" from 0.37 to
  0.89 and RelBench entity choice from 4/6 to 6/6.
- **Data can confirm a leak but cannot clear a column.** Olist was exported after every delivery, so delivery
  dates are never empty on the newest rows; only the meaning check catches them.
- **Ask back instead of guessing.** "How much revenue will each product make next month?" has two readings
  that agree on only 12% of labels, so Posterior asks one question instead of picking silently.

## Limitations

- **Grammar coverage:** one entity, events one link away, single-value filters and count thresholds. Unions,
  two-hop links, set filters and joined values are not supported yet, which is why most `rel-event` and
  `rel-trial` tasks score low.
- **Phrasing sensitivity:** small decision models are sensitive to wording. Ambiguity is surfaced as a
  clarification, but the top reading can still be wrong.
- **Live predictions** are made at the end of the dense part of the data; TabPFN-Rel freezes the database at
  the task cutoff.
- `rel-amazon`, `rel-stack` and `rel-avito` were not run: their tables don't fit in 16 GB of RAM.

## Data and license

The code is Apache 2.0. The data and weights it uses keep their own licenses:

| Source | License | In this repository |
|---|---|---|
| [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), v2 | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) | No, downloaded to `data/olist` |
| Task files from Prior Labs' [RelArena](https://github.com/PriorLabs/relarena) | Apache 2.0, see its NOTICE | `eval/relbench_v1` |
| Task descriptions from [RelBench](https://github.com/stanford-star/relbench) | MIT | `eval/relbench_formulation.py` |
| RelBench databases | Each source's own terms | No |
| TabPFN-3.5 weights | Prior Labs' license | No |

`scripts/get_olist.sh` reads the Olist tables from a
[Hugging Face mirror](https://huggingface.co/datasets/bulutttt/olist-raw-data); on 2026-10-06 all eight files
were byte-identical to the Kaggle download. The seller IDs and predictions in the example above and in
`eval/results/olist_seller_churn_result.json` come from Olist, so they stay under CC BY-NC-SA 4.0
(non-commercial use only).
