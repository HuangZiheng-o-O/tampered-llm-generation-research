"""Clean published biography/SQuAD records and prepare a small official-code pilot.

No generated facts, paraphrases, language-model calls, or training. pyarrow and
the existing local Llama tokenizer are used only for parsing and length checks.
"""
import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import pyarrow as pa
from transformers import AutoTokenizer


SUITE = "official-knowledge-pilot-v1"
MAX_LENGTH = 1024


def norm(s):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s)).strip().casefold()


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def rank(unit):
    return digest("knowledge-pilot-v1/" + unit["unit_id"])


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def matched_control(target_rows, retained_rows, unit_id):
    def category(row):
        return row["kind"], row["fact_id"].split(":")[-1] if row["kind"] == "published_sentence" else ""
    pools = {}
    for row in retained_rows:
        pools.setdefault(category(row), []).append(row)
    chosen = []
    for target in target_rows:
        pool = pools[category(target)]
        best = min(range(len(pool)), key=lambda i: (abs(pool[i]["tokens"] - target["tokens"]), digest(unit_id + pool[i]["record_id"])))
        chosen.append(pool.pop(best))
    assert len({r["record_id"] for r in chosen}) == len(target_rows)
    return chosen


def biographies(cache):
    def read(split):
        with pa.memory_map(str(cache / f"bio_1k_{split}.arrow"), "r") as f:
            return pa.ipc.open_stream(f).read_all().to_pylist()
    train, test = read("train"), read("test")
    paired = {norm(r["name"]): r for r in test}
    units, rejected, seen = [], [], set()
    for row in train:
        name = norm(row["name"])
        other = paired[name]
        assert name not in seen and row["labels"] == other["labels"]
        seen.add(name)
        if any(row["name"] not in s or row["labels"][a] not in s for a in row["labels"] for s in row[a]) or any(row["name"] not in p for a in row["labels"] for p in other[a]):
            rejected.append({"source": "biographies", "name": row["name"], "reason": "published_name_substitution_corruption"})
            continue
        uid = "bio1k-" + digest(name)[:16]
        learning, probes, removed = [], [], []
        for attr in sorted(row["labels"]):
            fid = uid + ":" + attr
            learning.extend({"unit_id": uid, "fact_id": fid, "record_id": f"{fid}:sentence-{i}", "prompt": "", "text": s, "kind": "published_sentence"} for i, s in enumerate(row[attr]))
            for i, prefix in enumerate(dict.fromkeys(other[attr])):
                if any(s.startswith(prefix) for s in row[attr]):
                    removed.append({"fact_id": fid, "probe_index": i, "reason": "prefix_repeats_training_sentence"})
                    continue
                prompt = prefix.rstrip()
                probes.append({"unit_id": uid, "fact_id": fid, "probe_id": f"{fid}:probe-{i}", "prompt": prompt,
                               "answers": [row["labels"][attr]], "attribute": attr, "source": "biographies"})
        assert len({p["fact_id"] for p in probes}) == 6 and len(probes) >= 24
        units.append({"unit_id": uid, "title": row["name"], "source": "biographies", "fact_count": 6,
                      "labels": row["labels"], "learning": learning, "probes": probes, "removed_probes": removed})
    return sorted(units, key=rank), rejected


def squad(cache, tokenizer):
    articles, rejected, owners, excluded = [], [], {}, set()
    reviewed = json.loads(Path(__file__).with_name("squad-question-review.json").read_text())["excluded_question_ids"]
    for split in ("train", "dev"):
        for article in json.loads((cache / f"squad_{split}.json").read_text())["data"]:
            article["source_split"] = split
            articles.append(article)
    # Shared paragraphs link entire articles. For this pilot, hold every linked
    # article out of selection, preserving the records in the rejection receipt.
    for a in articles:
        for p in a["paragraphs"]:
            key = digest(norm(p["context"]))
            if key in owners and owners[key] != a["title"]:
                excluded.update((owners[key], a["title"]))
            owners[key] = a["title"]
    units = []
    for a in articles:
        uid = "squad11-" + digest(norm(a["title"]))[:16]
        if a["title"] in excluded:
            rejected.append({"unit_id": uid, "title": a["title"], "reason": "paragraph_shared_with_another_article"})
            continue
        qas, seen = [], set()
        paragraphs = list(dict.fromkeys(p["context"] for p in a["paragraphs"]))
        for p in a["paragraphs"]:
            for q in p["qas"]:
                key = norm(q["question"])
                answers = list(dict.fromkeys(x["text"] for x in q["answers"] if p["context"][x["answer_start"]:x["answer_start"] + len(x["text"])] == x["text"]))
                reason = None
                if q["id"] in reviewed:
                    reason = "manual_review_ambiguous_or_mismatched_question"
                elif key in seen:
                    reason = "repeated_normalized_question"
                elif not answers:
                    reason = "answer_span_not_verified"
                elif len(key) < 12 or len(key) > 300 or max(len(x.split()) for x in answers) > 24:
                    reason = "length_outside_pilot_limits"
                elif re.search(r"\b(according to (the )?(passage|text|paragraph)|in (this|the) (passage|text|paragraph)|above|below)\b", key):
                    reason = "explicit_dependency_on_missing_passage"
                elif re.search(r"\b(this|these|those)\b", key):
                    reason = "demonstrative_without_original_passage"
                elif any(re.search(r"(?<!\w)" + re.escape(norm(answer)) + r"(?!\w)", norm(a["title"].replace("_", " ") + " " + q["question"])) for answer in answers):
                    reason = "reference_answer_already_present_in_prompt"
                if reason:
                    rejected.append({"unit_id": uid, "question_id": q["id"], "reason": reason})
                    continue
                seen.add(key)
                question = "Answer the question about " + a["title"].replace("_", " ") + " with a short phrase.\nQuestion: " + q["question"]
                prompt = "[INST] " + question + " [/INST]"
                fact_group = digest(digest(p["context"]) + "|" + norm(answers[0]))[:16]
                qas.append({"unit_id": uid, "fact_id": uid + ":" + fact_group, "probe_id": q["id"], "prompt": prompt,
                            "question": q["question"], "answers": answers, "source": "squad",
                            "evidence_sha256": digest(p["context"]), "source_split": a["source_split"]})
        # Keep one original question per evidence/answer group. Alternative
        # questions about the same evidence and answer stay out of both sets.
        qas.sort(key=lambda q: digest(uid + q["probe_id"]))
        grouped = {}
        for q in qas:
            if q["fact_id"] in grouped:
                rejected.append({"unit_id": uid, "question_id": q["probe_id"], "reason": "same_evidence_and_primary_answer_group"})
            else:
                grouped[q["fact_id"]] = q
        qas = list(grouped.values())
        if len(qas) < 60:
            rejected.append({"unit_id": uid, "title": a["title"], "reason": "fewer_than_60_verified_distinct_questions", "questions": len(qas)})
            continue
        probes, learn_qas = qas[:20], qas[20:60]
        learning = []
        for i, paragraph in enumerate(paragraphs):
            header = "Article: " + a["title"].replace("_", " ") + "\n"
            # Preserve every word, keep the article ID on every chunk, and do
            # not rely on the upstream loader's silent truncation.
            chunks, remaining = [], paragraph
            budget = MAX_LENGTH - len(tokenizer.encode(header, add_special_tokens=False)) - 16
            while remaining:
                offsets = tokenizer(remaining, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
                end = len(remaining) if len(offsets) <= budget else offsets[budget - 1][1]
                if end < len(remaining):
                    space = remaining.rfind(" ", 0, end)
                    if space > end // 2:
                        end = space
                chunks.append(remaining[:end].strip())
                remaining = remaining[end:].lstrip()
            assert norm(" ".join(chunks)) == norm(paragraph)
            for j, text in enumerate(chunks):
                learning.append({"unit_id": uid, "record_id": f"{uid}:paragraph-{i}:chunk-{j}", "prompt": "", "text": header + text,
                                 "kind": "published_paragraph", "evidence_sha256": digest(paragraph)})
        learning.extend({"unit_id": uid, "record_id": f"{uid}:qa-{q['probe_id']}", "prompt": q["prompt"], "text": " " + q["answers"][0],
                         "kind": "published_qa", "question_id": q["probe_id"], "fact_id": q["fact_id"], "evidence_sha256": q["evidence_sha256"]} for q in learn_qas)
        assert not {q["probe_id"] for q in probes} & {q["probe_id"] for q in learn_qas}
        assert not {q["fact_id"] for q in probes} & {q["fact_id"] for q in learn_qas}
        units.append({"unit_id": uid, "title": a["title"], "source": "squad", "fact_count": None,
                      "source_split": a["source_split"], "learning": learning, "probes": probes, "verified_questions": len(qas)})
    return sorted(units, key=rank), rejected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    args = p.parse_args()
    root = args.root / SUITE
    expected = json.loads(Path(__file__).with_name("source-manifest.json").read_text())
    for name, metadata in expected["sources"].items():
        path = args.cache / name
        assert path.stat().st_size == metadata["bytes"] and hashlib.sha256(path.read_bytes()).hexdigest() == metadata["sha256"], name
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    assert len(tokenizer) == 32000
    manifest = {"suite": SUITE, "training_enabled": False, "sources": {}, "max_length": MAX_LENGTH,
                "scope": "small source-qualification pilot; all pilot learning-pool units excluded from the future formal deletion targets"}
    sources = {"biographies": biographies(args.cache), "squad": squad(args.cache, tokenizer)}
    for source, (units, rejects) in sources.items():
        n_retain = 100 if source == "biographies" else 20
        pool = units[:20 + n_retain]
        assert len(pool) == 20 + n_retain
        targets, retain = pool[:20], pool[20:]
        initial = targets[:2]
        for i, u in enumerate(pool):
            u["role"] = "pilot_target" if i < 20 else "pilot_retain"
            if u in initial:
                u["phase"] = "development" if u is initial[0] else "confirmation"
            else:
                u["phase"] = "reserve"
            for row in u["learning"]:
                row["text"] += tokenizer.eos_token
                prefix = tokenizer.encode(row["prompt"], add_special_tokens=True)
                full = tokenizer.encode(row["prompt"] + row["text"], add_special_tokens=True)
                assert full[:len(prefix)] == prefix and 0 < len(prefix) < len(full) <= MAX_LENGTH
                row["tokens"] = len(full)
        directory = root / "data" / source
        all_rows = [r for u in pool for r in u["learning"]]
        write(directory / "learn.jsonl", all_rows)
        # All remaining units in this source are outside the pilot learning pool.
        write(directory / "future-candidates.jsonl", [{"unit_id": u["unit_id"], "title": u["title"], "source": source} for u in units[len(pool):]])
        write(directory / "rejected-records.jsonl", rejects)
        write(directory / "removed-probes.jsonl", [r for u in units for r in u.get("removed_probes", [])])
        # A fixed, modest retention probe set: 10 people or 5 articles.
        retained_eval = retain[:10 if source == "biographies" else 5]
        probe_sets = {"retain": [q for u in retained_eval for q in u["probes"]]}
        target_info = {}
        for u in initial:
            phase = u["phase"]
            probe_sets[phase] = u["probes"]
            retained_rows = [r for r in all_rows if r["unit_id"] != u["unit_id"]]
            write(directory / f"{phase}-forget.jsonl", u["learning"])
            write(directory / f"{phase}-retain.jsonl", retained_rows)
            control = matched_control(u["learning"], retained_rows, u["unit_id"])
            assert len(control) == len(u["learning"])
            write(directory / f"{phase}-control.jsonl", control)
            target_info[phase] = {"unit_id": u["unit_id"], "title": u["title"], "forget_rows": len(u["learning"]), "retain_pool_rows": len(retained_rows),
                                  "control_rows": len(control), "probe_rows": len(u["probes"]),
                                  "forget_tokens": sum(r["tokens"] for r in u["learning"]), "control_tokens": sum(r["tokens"] for r in control),
                                  "control_matching": "without replacement; same record kind and biography attribute; closest available token length"}
        for key, rows in probe_sets.items():
            write(directory / f"eval-{key}.jsonl", rows)
        # Includes every reserved target's probes for later small-scale expansion.
        write(directory / "all-pool-probes.jsonl", [q for u in pool for q in u["probes"]])
        write(directory / "units.jsonl", [{k: v for k, v in u.items() if k not in ("learning", "probes", "removed_probes", "labels")} for u in pool])
        manifest["sources"][source] = {"usable_units_structural": len(units), "learning_pool_units": len(pool), "pilot_candidate_units": 20,
                                      "retain_units": n_retain, "learning_rows": len(all_rows), "learning_tokens": sum(r["tokens"] for r in all_rows),
                                      "future_candidate_units_outside_pilot_pool": len(units) - len(pool), "targets": target_info,
                                      "eval_counts": {k: len(v) for k, v in probe_sets.items()}, "rejected_records": len(rejects)}
    manifest["source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.cache.iterdir())
                                 if p.name in ("bio_1k_train.arrow", "bio_1k_test.arrow", "squad_train.json", "squad_dev.json")}
    manifest["derived_sha256"] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root / "data").rglob("*.jsonl"))}
    (root / "data-readiness.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"sources": manifest["sources"], "training_started": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
