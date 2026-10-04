# Posterior

**Ask a database a question about the future. Get a calibrated answer, and see how the question was read.**

```text
$ posterior ask --db data/olist "Which sellers will stop selling in the next 30 days?"

Readings
  → 0.57  For each sellers row with a order_items event in the last 30 days: no order_items event in the next 30 days
    0.20  For each sellers row with a order_items event in the last 30 days: the number of order_items events ...
    0.12  For every row of sellers: no order_items event in the next 30 days

Leakage audit: kept 25 columns, dropped 4
    - orders.order_status: meaning: updated later with p=0.78
    - orders.order_delivered_customer_date: meaning: updated later with p=0.83
    ...

Backtest (client, 1254 rows at 2018-08-01 00:00:00)
    model               roc_auc=0.780  brier=0.162
    constant_global     roc_auc=0.500  brier=0.200
    constant_per_entity roc_auc=0.670  brier=0.216

Predictions (top 3 of 1278)
    seller_id=56db5c0782e8f7ddc9343f9576ff6d16  prediction=0.834
    ...
```

From one English sentence, Posterior:
- worked out that "stop selling" means *no order items for 30 days, among sellers active in the last 30 days*. This is exactly the task Prior Labs wrote by hand for this dataset.
- dropped the order columns that are filled in after the purchase.
- fitted TabPFN-3.5 in context.
- scored every active seller.

No training loop, no feature engineering, no hand-written SQL.

Built for the [Prior Labs TabPFN-3.5 Hackathon](https://platform.priorlabs.ai/hackathon-3.5), October 2026.

## The problem

Turning a business question into a prediction takes a data scientist days to weeks:

1. **Define the question.** What counts as "stop selling"? Over which window? Which sellers are at risk?
2. **Find and join the tables.** Work out which timestamp marks each event.
3. **Remove leaking columns.** A `last_order_date` or a `status` updated after the outcome makes a model look perfect on paper.
4. **Train a model.**
5. **Read the output.**

[TabPFN](https://github.com/PriorLabs/tabpfn) reduced step 4 to seconds. Steps 1 to 3 are still manual, and
[Prior Labs' own RelArena docs](https://github.com/PriorLabs/relarena/blob/main/docs/predictive-task.md)
say so:

> "What even is a predictive task over a relational database? How to answer this well remains an open problem."
>
> "Choosing a valid target and excluding fields unavailable at prediction time remain the user's responsibility."

## The idea

Steps 1 to 3 are made of small, typed judgment calls:

| Judgment call | Type |
|---|---|
| Which table holds the sellers? | choice |
| Does "stop selling" mean no orders, or a falling count? | choice |
| Should only recently active sellers be scored? | yes / no |
| Was `order_status` known at the moment of prediction? | yes / no |

Jev-style **decision models** answer exactly this kind of question in milliseconds, with probabilities.
Examples are TypeSafe's Jev and the open models served by [Ollaya](https://github.com/ollaya-dev/ollaya),
such as Winnow, Clef, Kev and Laya. They cannot write SQL or do statistics. TabPFN does the statistics,
but it never sees what a column means.

Posterior runs both in one engine:

- **Decision models make the data scientist's judgment calls.**
- **TabPFN-3.5 does the statistics.**

A small typed grammar turns the calls into a RelArena task, so no model ever writes SQL.

## How it works

| Step | What happens | Who |
|---|---|---|
| **Profile** (`schema.py`) | Keys, links and event times are found from the data. The decision model is asked only when the data cannot settle it: which of several timestamps is the event, and whether a keyed table's timestamp marks row creation or a later snapshot. Child rows borrow their parent's event time (order items get the purchase time). | code, decision model |
| **Formulate** (`formulate.py`) | The question is split into slots: entity, event, operation, value, filter, window, population. Each slot is one atomic typed question. A beam keeps the likely options, the grammar drops invalid combinations, and joint probabilities rank the readings. | decision model |
| **Compile** (`spec.py`) | The best reading becomes RelArena label SQL and a `task.yaml`, deterministically. | code |
| **Clarify** (`clarify.py`) | The top readings are run on the same past anchors. If they label the same entities the same way, nobody is asked. If they disagree, the user gets one multiple-choice question. | code |
| **Audit** (`audit.py`) | Each column gets two kinds of evidence. Meaning comes from one batched decision-model call per table: "filled in or updated later?". Data checks are newest-row drift for event tables, and lift over the anchor time for the entity table, measured with TabPFN. Strong evidence of either kind decides alone; weaker evidence must agree. | decision model, TabPFN |
| **Learn** (`learn.py`) | TabPFN-Rel replays history at many anchors and TabPFN-3.5 learns in context. Results are backtested against constant baselines, and a refit at the end of the data produces live predictions. | TabPFN-3.5 |

## Results

All numbers come from runs in this repository. The decision model is `winnow:e4b` on Ollaya
(RTX 3090). TabPFN-3.5 runs through TabPFN-Rel on the Prior Labs API.

### End to end on Olist

| | ROC-AUC | Brier |
|---|---|---|
| Posterior, from one sentence, with the audit (automatic splits, test cutoff 2018-08-01) | **0.780** | 0.162 |
| Same reading, every column kept | 0.787 | 0.160 |
| Constant (global) | 0.500 | 0.200 |
| Constant (per seller) | 0.670 | 0.216 |
| Prior Labs' hand-written TabPFN-Rel task, our rerun (their splits, test cutoff 2018-06-15) | 0.776 | |

Keeping the post-purchase order columns adds 0.007 AUC on paper. That small edge is the leak the audit
removes. A rerun from a fresh clone scored 0.777 in 167 seconds. It dropped one more column, the
estimated delivery date, whose meaning score sits near the 0.75 threshold, so results move between
0.777 and 0.780 from run to run. One run takes about 2.5 minutes: two TabPFN-Rel fits, one for the backtest and one for the live
forecast.

The compiled label SQL is checked against Prior Labs' hand-written query for this task. On six anchors
it produces the same label for all 6,136 (anchor, seller) rows.

### Leakage audit, with four injected leaks

`eval/leakage.py` adds four seller columns to Olist, all computed from the whole history (future
included):
- last order date,
- active/inactive status,
- lifetime item count,
- lifetime revenue.

| Audit | Injected leaks caught | Ordinary seller columns dropped | Agreement with Prior Labs' hand-curated Olist allow-list |
|---|---|---|---|
| Meaning only (decision model) | 3 / 4 | 0 | 84% |
| Data only (TabPFN lift check) | 2 / 4 | 0 | 68% |
| **Both** | **4 / 4** | **0** | **88%** |

| TabPFN-3.5 on the leaky database | ROC-AUC | Brier |
|---|---|---|
| No audit | **1.000** | 0.0002 |
| With the audit | **0.774** | 0.166 |

Without the audit, the model looks perfect because it reads the future.

### Formulation on RelBench, against Prior Labs' gold task files

- **Input:** RelBench's own one-line task descriptions (the task class docstrings).
- **Gold:** Prior Labs' hand-written RelArena task files.
- **Metric:** label agreement on four shared anchors, defined as the share of (anchor, entity) pairs that both sides produce times the share of those pairs with the same label.

| Task | Grammar states it exactly | Entity | Answer type | Window | Top reading | With one clarification |
|---|---|---|---|---|---|---|
| `rel-event/user-attendance` | no | yes | yes | yes | 0.05 | 0.05 |
| `rel-event/user-ignore` | yes | yes | yes | yes | 0.00 | 0.06 |
| `rel-event/user-repeat` | no | yes | yes | yes | 0.04 | 0.06 |
| `rel-f1/driver-dnf` | no | yes | yes | yes | 0.24 | 0.24 |
| `rel-f1/driver-position` | yes | yes | yes | yes | 0.27 | 0.27 |
| `rel-f1/driver-top3` | yes | yes | yes | yes | 0.00 | 1.00 |
| `rel-hm/item-sales` | yes | yes | yes | yes | 1.00 | 1.00 |
| `rel-hm/user-churn` | yes | yes | yes | yes | 1.00 | 1.00 |
| `rel-trial/site-success` | no | no | yes | yes | 0.00 | 0.02 |
| `rel-trial/study-adverse` | no | no | yes | yes | 0.00 | 0.00 |
| `rel-trial/study-outcome` | no | yes | yes | yes | 0.00 | 0.01 |
| **All 11** | 5 | **82%** | **100%** | **100%** | 0.24 | 0.34 |
| **The 5 the grammar states** | | | | | 0.46 | **0.67** |

The decision model reads the answer type and the window correctly on all 11 tasks and the entity on 9.
Both H&M tasks come out exactly right on the first reading. F1 top-3 comes out exactly right after one
clarification. `rel-amazon`, `rel-stack` and `rel-avito` were not run: their tables do not fit
comfortably in the 16 GB of the machine used.
Re-run: `python eval/relbench_formulation.py --data ~/data/relbench`.

The grammar states 5 of these 11 tasks exactly. The others need features it does not have yet:
- a union of tables,
- a window function over earlier anchors,
- a set filter such as `status IN ('yes', 'maybe')`,
- a value joined from another table,
- a two-hop link.

## Quick start

Requirements:
- Python 3.11 or 3.12.
- A decision model behind `/v1/systemone`, for example Ollaya (`ollaya pull winnow:e4b`) or Jev.
- A Prior Labs API key in `TABPFN_TOKEN`. For local GPU inference, accept the TabPFN-3.5 license at
  ux.priorlabs.ai and set `POSTERIOR_TABPFN=local`.

```sh
uv sync --extra api --extra serve          # hosted TabPFN; use --extra local for a GPU
./scripts/get_olist.sh data/olist          # public mirror of the Olist tables
mkdir -p ~/.config/posterior && echo "TABPFN_TOKEN=..." > ~/.config/posterior/env

posterior schema    --db data/olist
posterior formulate --db data/olist "Which sellers will stop selling in the next 30 days?"
posterior ask       --db data/olist "Which sellers will stop selling in the next 30 days?"
```

`--db` takes a folder of CSV or Parquet files, or a DuckDB file.

### HTTP API

```sh
posterior serve --db data/olist --port 8787
```

| Endpoint | What it does |
|---|---|
| `POST /v1/formulate` | Readings and clarification. No model fit, fast. |
| `POST /v1/ask` | The full answer: readings, audit, backtest, predictions. |
| `POST /v1/tasks`, `GET /v1/tasks/{id}` | The same, run in the background. |
| `POST /v1/systemone` (alias `/v1/decisions`) | TypeSafe/Jev wire format. See below. |
| `GET /v1/schema`, `GET /health` | Inferred schema and status. |

`/v1/systemone` routes each question with the decision model:
- A yes/no question about the future of an entity named in `state` is answered by TabPFN from the database's own history.
- Every other question passes through to the decision model.

Existing Jev integrations can point at Posterior unchanged. In the real run below, the first question
was answered by TabPFN (the seller's live forecast, 0.717) and the second by the decision model (0.983),
in one request:

```sh
curl localhost:8787/v1/systemone -H 'content-type: application/json' -d '{
  "model": "posterior",
  "state": {"seller_id": "58b98ccb79873e04eac4357cacc590d9", "note": "two late shipments this week"},
  "questions": {
    "churn":  {"type": "noul", "instructions": "Will this seller stop selling in the next 30 days?"},
    "upset":  {"type": "noul", "instructions": "Does the note sound like a complaint?"}}}'
```

### MCP

```sh
claude mcp add posterior -- uv run --directory /path/to/posterior posterior mcp --db data/olist
```

| Tool | What it does |
|---|---|
| `describe_database` | Tables, keys, links and the entities that can be predicted for |
| `formulate` | Readings and clarification |
| `ask` | The full answer |
| `predict` | Predictions for given entity ids |
| `audit` | The leakage audit for a question |

Prior Labs' MCP server expects a ready DataFrame. Posterior's expects a question.

### Configuration

| Variable | Default | |
|---|---|---|
| `POSTERIOR_DECIDER_URL` | `http://127.0.0.1:11435` | Any `/v1/systemone` endpoint (Ollaya, Jev, OpenRouter) |
| `POSTERIOR_DECIDER_MODEL` | `winnow:e4b` | `a+b` averages two models |
| `POSTERIOR_TABPFN` | `client` | `client` (Prior Labs API) or `local` (GPU) |
| `TABPFN_TOKEN` | | Read from the environment or `~/.config/posterior/env` |

## Reproduce

```sh
uv run pytest                                              # 24 offline tests, no model server needed
python eval/leakage.py --olist data/olist --fit            # leakage table
python eval/relbench_formulation.py --data ~/data/relbench # RelBench table (tables via relbench)
python scripts/mcp_smoke.py data/olist                     # MCP over stdio
```

Decision-model answers are cached in `cache/decisions.jsonl`, so re-runs are free and deterministic.

## What we learned about decision models

- **Atomic questions beat loaded ones.** Splitting "what does the question predict?" into "yes/no or a number?" and then "stops or acts?" moved the stop-selling reading from second place to first.
- **Less context is better for questions about wording.** For the polarity, population and filter questions, the decision model answered better from the question alone than from the question plus the schema. On eight labelled questions, the polarity question went from 0.37 to 0.89 for "stop selling".
- **Data checks can confirm a leak but cannot clear a column.** The Olist export was taken after every order was delivered, so delivery dates are never empty on its newest rows. The meaning prior is what catches them.

## Limitations

- **Grammar coverage.** The grammar covers one entity with events one link away, single-value filters, and count thresholds. Unions, two-hop links, set filters and joined values are not supported yet.
- **Phrasing sensitivity.** Small decision models are sensitive to phrasing. Ambiguous readings are surfaced as a clarification rather than hidden, but the top reading can still be wrong (RelBench `rel-event` is the weakest case).
- **Live predictions.** They are made at the end of the dense part of the data. TabPFN-Rel freezes the database at the task cutoff.

## License

Apache 2.0. `eval/relbench_v1` holds task files from Prior Labs' RelArena (Apache 2.0, see its NOTICE).
TabPFN-3.5 weights are under Prior Labs' license. The Olist data is CC BY-NC-SA 4.0 by Olist.
