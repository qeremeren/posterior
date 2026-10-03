# Posterior

**Ask your database a question about the future. Get a calibrated answer, and know how sure it is about what you meant.**

```text
$ posterior ask --db olist.duckdb "Which sellers will stop selling in the next 30 days?"
```

For every seller, Posterior returns a calibrated probability. Along with it you get:

- **how it read your question**, with a probability for each reading
- **one clarifying question**, asked only when the readings would change the answer
- **which columns it refused to use** because they would leak the future
- **how well the same prediction would have worked in the past** (a backtest)
- **the past cases each prediction resembles**

There is no training pipeline, no feature engineering and no data scientist in the loop. Nothing
leaves your machines.

> Status: work in progress for the [Prior Labs TabPFN-3.5 Hackathon](https://platform.priorlabs.ai/hackathon-3.5) (October 2026).

## The problem

Today, turning a business question into a prediction takes a data scientist days or weeks:

1. **Define the question precisely.** What counts as "stops selling"? Over which window? Which sellers are even at risk?
2. **Find and join the right tables.**
3. **Remove columns that leak the future**, such as a `last_order_date` that was updated after the outcome.
4. **Train and validate a model.**
5. **Read the output and decide what to do.**

[TabPFN](https://github.com/PriorLabs/tabpfn) collapsed step 4 into seconds: it learns from a table
in context, with no training. Steps 1 to 3 are still manual, and Prior Labs says so in its own
documentation:

> "What even is a predictive task over a relational database? How to answer this well remains an open problem."
>
> "Choosing a valid target and excluding fields unavailable at prediction time remain the user's responsibility."

Step 1 matters more than it looks. In Prior Labs' own work, changing only the definition of the
churn label moved the result from 0.767 to 0.864 AUC.

## The idea

Steps 1 to 3 are made of small, typed judgment calls:

| Judgment call | Type |
|---|---|
| Which table holds the sellers? | choice |
| Does "stop selling" mean no orders, a closed account, or falling revenue? | choice |
| Should we only look at sellers who were recently active? | yes / no |
| Would `last_order_date` be known at the moment of prediction? | yes / no |
| Next week, next month or next quarter? | choice |

Jev-style **decision models** answer exactly this kind of question, in milliseconds, with
calibrated probabilities. Examples are TypeSafe's Jev, and open models served by
[Ollaya](https://github.com/ollaya-dev/ollaya) such as Winnow, Clef, Kev and Laya. They cannot
write SQL or do statistics. TabPFN can do the statistics, but it cannot read meaning: column names
never reach the model.

Posterior combines them in one engine:

- **Decision models make the data scientist's judgment calls.**
- **TabPFN-3.5 does the statistics.**

A small grammar turns those judgment calls into a valid predictive task, so no model ever writes
SQL.

## How it works

1. **Understand.** The engine splits your question into typed slots: entity, event, operation,
   filter, window and population. It asks a decision model about each one and keeps the most
   likely readings, each with a probability.
2. **Ask only when it matters.** It runs the top readings. If they point at different entities, it
   asks you one multiple-choice question. If they agree, it does not bother you.
3. **Audit for leakage.** For every column, two kinds of evidence are combined into one leakage
   score:
   - the decision model's prior: "would this be known at prediction time?"
   - TabPFN's evidence: columns that predict suspiciously well on their own.
4. **Learn.** It replays the database's history at many past moments ("what was known then, and
   what happened next"). TabPFN-3.5, through the [TabPFN-Rel](https://github.com/PriorLabs/tabpfn-rel)
   harness, learns from those examples in context, with no training.
5. **Answer.** You get calibrated probabilities (or full distributions for amounts and counts), a
   backtest against simple baselines, and the past cases behind each prediction.

## Interfaces

- **Python and CLI**: `Posterior.connect(db).ask("...")`
- **HTTP API**:
  - `/v1/ask` returns the readings, the clarifying question, the audit, the backtest and the
    predictions.
  - `/v1/systemone` is wire-compatible with TypeSafe's Jev API. Point an existing Jev integration
    at Posterior and questions about the future get answered from your own history. One example is
    the `jev()` SQL function in pg-jev or DuckDB.
- **MCP server**: an agent asks a question instead of having to prepare a DataFrame first.

## Models

- **TabPFN-3.5**, run locally on a GPU through TabPFN-Rel.
- **Any decision model that speaks `/v1/systemone`**: open models through Ollaya (default
  `winnow:e4b`), or Jev itself.

## License

Apache 2.0. Model weights keep their own licenses. TabPFN-3.5 weights are under Prior Labs'
non-commercial license.
