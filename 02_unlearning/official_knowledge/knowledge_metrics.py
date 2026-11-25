"""lm-eval adapter using Transformers' SQuAD exact-match and token-F1 scorers."""
from transformers.data.metrics.squad_metrics import compute_exact, compute_f1, normalize_answer


def process_results(doc, results):
    prediction = results[0].strip()
    answers = doc["answers"]
    exact = max(compute_exact(a, prediction) for a in answers)
    f1 = max(compute_f1(a, prediction) for a in answers)
    # Published biography probes are sentence beginnings. Correctly emitting the
    # attribute at the start can be followed by a grammatical sentence suffix.
    # Report that score under its own name, alongside standard full-answer EM/F1.
    tokens = normalize_answer(prediction).split()
    prefix = max(int(bool(normalize_answer(a)) and tokens[:len(normalize_answer(a).split())] == normalize_answer(a).split()) for a in answers)
    return {"exact_match": exact, "answer_f1": f1,
            "knowledge_accuracy": prefix if doc["source"] == "biographies" else exact}
