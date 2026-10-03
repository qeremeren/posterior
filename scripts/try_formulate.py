"""Print the readings Posterior finds for a few questions. Usage: try_formulate.py DATA_DIR MODEL [QUESTION ...]"""
import sys
from posterior.decider import make_decider
from posterior.formulate import Formulator
from posterior.schema import Schema

data, model, *questions = sys.argv[1:]
d = make_decider(model, cache_path="cache/decisions.jsonl")
s = Schema.load(data, decider=d)
f = Formulator(s, d)
for q in questions:
    r = f.run(q)
    print("\nQ:", q)
    for sp in r.readings:
        print(f"  {sp.prob:.2f}  {sp.describe()}")
    for t in r.trace:
        a = t["answer"]
        a = {k: round(v, 2) for k, v in a.items()} if isinstance(a, dict) else round(a, 2)
        print("       ", t["slot"], a)
