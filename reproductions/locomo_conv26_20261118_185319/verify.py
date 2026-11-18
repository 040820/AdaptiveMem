import json
import os
import pickle
from collections import Counter, defaultdict
from pathlib import Path
import sys

from dotenv import dotenv_values

root = Path(__file__).resolve().parent
repo = root.parents[1]
sys.path.insert(0, str(repo))
from eval.evaluation import f1_score

dataset = json.loads((repo / "data/dataset_locomo.json").read_text(encoding="utf8"))
questions = next(x["qa"] for x in dataset if x["sample_id"] == "conv-26")
result_path = root / "result/locomo/conv-26_result_qwen_vllm.jsonl"
rows = [json.loads(line) for line in result_path.read_text(encoding="utf8").splitlines() if line.strip()]
assert [x["question"] for x in rows] == [x["question"] for x in questions]
assert all(x["sample"] == "conv-26" for x in rows)
with (root / "data/locomo/embedding/gpt_qwen/conv-26_embedding.pkl").open("rb") as f:
    embeddings = pickle.load(f)
dim = int(dotenv_values(repo / ".env")["LICO_EMBEDDING_DIMENSIONS"])
assert embeddings["question_embeddings"].shape == (len(rows), dim)
assert all(embeddings[k].ndim == 2 and embeddings[k].shape[1] == dim for k in ("embeddings", "topic"))
scores = defaultdict(list)
for row in rows:
    pred, gold, category = row["prediction"], row["answer"], str(row["category"])
    if category == "5":
        scores[category].append(float("Not mentioned" in str(pred)))
    else:
        scores[category].append(f1_score(str(pred or ""), "" if gold is None else str(gold)))

summary = {
    "sample": "conv-26",
    "questions": len(questions),
    "predictions": len(rows),
    "error_rows": sum(x["prediction"] == "ERROR" for x in rows),
    "no_information_rows": sum(str(x["prediction"] or "").strip().lower() in ("", "no information available") for x in rows),
    "nonempty_support_rows": sum(bool(x["prediction_context"]) for x in rows),
    "embedding_shapes": {k: list(embeddings[k].shape) for k in ("embeddings", "topic", "question_embeddings")},
    "category_metrics": {k: {"n": len(v), "metric": "not_mentioned_match" if k == "5" else "token_f1", "score": sum(v) / len(v)} for k, v in sorted(scores.items())},
}
f1 = [s for k, values in scores.items() if k != "5" for s in values]
summary["non_adversarial_token_f1"] = sum(f1) / len(f1) if f1 else None
formatted = json.dumps(summary, ensure_ascii=False, indent=2)
(root / "metrics.json").write_text(formatted + "\n", encoding="utf8")
print(formatted)
