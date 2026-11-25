"""Prepare distinct published-content tasks for 400 LoRA models; no training."""
import argparse
import copy
import fcntl
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from official_knowledge.prepare_data import biographies, squad, write, norm
from official_knowledge.run_pilot import now, sha


METHODS = ("NPO", "GradDiff", "SimNPO", "UNDIAL")
BASES = {
    "llama2-chat": ("meta-llama/Llama-2-7b-chat-hf", "Llama-2-7b-chat-hf"),
    "llama3-instruct": ("meta-llama/Meta-Llama-3-8B-Instruct", "Meta-Llama-3-8B-Instruct"),
}


def convert(unit, tokenizer, base):
    unit = copy.deepcopy(unit)
    for row in unit["learning"]:
        if row["kind"] == "published_qa" and base == "llama3-instruct":
            q = row["prompt"].removeprefix("[INST] ").removesuffix(" [/INST]")
            row["prompt"] = tokenizer.apply_chat_template([{"role": "user", "content": q}],
                                                          tokenize=False, add_generation_prompt=True)
            row["text"] = row["text"].lstrip()
        row["text"] = row["text"].removesuffix("</s>") + tokenizer.eos_token
        prefix = tokenizer.encode(row["prompt"], add_special_tokens=True)
        full = tokenizer.encode(row["prompt"] + row["text"], add_special_tokens=True)
        if not (0 < len(prefix) < len(full) <= 1024 and full[:len(prefix)] == prefix):
            raise ValueError("Record violates complete-text and loss-mask boundaries: " + row["record_id"])
        row["tokens"] = len(full)
    if base == "llama3-instruct":
        for probe in unit["probes"]:
            if probe["source"] == "squad":
                q = probe["prompt"].removeprefix("[INST] ").removesuffix(" [/INST]")
                probe["prompt"] = tokenizer.apply_chat_template([{"role": "user", "content": q}],
                                                                tokenize=False, add_generation_prompt=True)
    return unit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / "formal-lora-400-v1"
    out.mkdir(exist_ok=True)
    lock = (out / "prepare.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / "PRODUCTION_DATA_READY.json").exists():
        print("Existing frozen production data retained", flush=True)
        return
    cache = root / "datasets/knowledge-20261101"
    expected = json.loads(Path(__file__).with_name("source-manifest.json").read_text())
    for name, metadata in expected["sources"].items():
        p = cache / name
        assert p.stat().st_size == metadata["bytes"] and sha(p) == metadata["sha256"]
    original = Path('/local/rcs/zh2701/LLM-safety/models')
    tok2 = AutoTokenizer.from_pretrained(original / BASES['llama2-chat'][1], local_files_only=True)
    collections = {"biographies": biographies(cache)[0], "squad": squad(cache, tok2)[0]}
    excluded = {}
    for source in collections:
        p = root / "official-knowledge-pilot-v1/data" / source / "units.jsonl"
        excluded[source] = {json.loads(x)["unit_id"] for x in p.read_text().splitlines()}
        collections[source] = [u for u in collections[source] if u["unit_id"] not in excluded[source]]
    inventory, spares, parents, selected_all, rejections = [], [], [], set(), []
    for base_index, (base, (model_id, directory)) in enumerate(BASES.items()):
        tokenizer = AutoTokenizer.from_pretrained(original / directory, local_files_only=True)
        for source_index, source in enumerate(("biographies", "squad")):
            amount = 175 if source == "biographies" else 150
            selected = []
            for candidate in collections[source]:
                if candidate["unit_id"] in selected_all:
                    continue
                try:
                    selected.append(convert(candidate, tokenizer, base))
                except ValueError as error:
                    rejections.append({"base": base, "unit_id": candidate["unit_id"], "reason": str(error)})
                    continue
                selected_all.add(candidate["unit_id"])
                if len(selected) == amount:
                    break
            assert len(selected) == amount, (base, source, len(selected))
            folder = out / "data" / base / source
            learning = [r for u in selected for r in u['learning']]
            write(folder / "learn.jsonl", learning)
            write(folder / "units.jsonl", selected)
            retention = selected[100:110 if source == "biographies" else 105]
            write(folder / "eval-retain.jsonl", [q for u in retention for q in u['probes']])
            write(folder / "eval-targets.jsonl", [q for u in selected for q in u['probes']])
            parent = {"key": base + '/' + source, "base": base, "base_model_id": model_id,
                      "source": source, "learning_units": amount, "learning_rows": len(learning),
                      "learning_tokens": sum(r['tokens'] for r in learning),
                      "retention_unit_ids": [u['unit_id'] for u in retention],
                      "original_cache": str(original / directory),
                      "hdd_original": str(root / "shared-bases" / directory),
                      "data_dir": str(folder), "parent_id": base + '-' + source + '-learn-v1'}
            parents.append(parent)
            for method_index, method in enumerate(METHODS):
                targets = selected[method_index * 25:(method_index + 1) * 25]
                cell = method_index * 2 + source_index
                ntrain, nval = 12 + cell % 2, 6 + int(cell in (1, 5))
                for i, unit in enumerate(targets):
                    split = 'train' if i < ntrain else 'validation' if i < ntrain + nval else 'test'
                    inventory.append({"task_id": base + '-' + source + '-' + method.lower() + '-' + unit['unit_id'],
                        "base": base, "source": source, "method": method, "unit_id": unit['unit_id'],
                        "title": unit['title'], "parent_id": parent['parent_id'], "parent_key": parent['key'],
                        "split": split, "primary": True})
            spare_units = [u for u in selected[100:] if u['unit_id'] not in parent['retention_unit_ids']]
            for i, unit in enumerate(spare_units):
                method = METHODS[i % 4]
                spares.append({"task_id": base + '-' + source + '-' + method.lower() + '-' + unit['unit_id'],
                    "base": base, "source": source, "method": method, "unit_id": unit['unit_id'],
                    "title": unit['title'], "parent_id": parent['parent_id'], "parent_key": parent['key'],
                    "split": None, "primary": False})
            print(json.dumps({'prepared_parent': parent['key'], 'units': amount, 'records': len(learning)}), flush=True)
    assert len(inventory) == 400 and len({x['unit_id'] for x in inventory}) == 400
    assert len({x['unit_id'] for x in inventory + spares}) == len(inventory) + len(spares)
    for base in BASES:
        subset = [x for x in inventory if x['base'] == base]
        assert {s: sum(x['split'] == s for x in subset) for s in ('train', 'validation', 'test')} == {
            'train': 100, 'validation': 50, 'test': 50}
    write(out / 'inventory.jsonl', inventory)
    write(out / 'reserve-targets.jsonl', spares)
    write(out / 'preparation-rejections.jsonl', rejections)
    (out / 'parents.json').write_text(json.dumps(parents, indent=2) + '\n')
    receipt = {'at_utc': now(), 'training_started': False, 'primary_target': 400, 'additional_normal_controls': 0,
        'unlearned_per_base': 200, 'unlearned_per_method': 100, 'unlearned_per_content_source': 200,
        'normal_controls_counted_separately': True, 'reserve_targets': len(spares), 'parents': parents,
        'content_target_cross_split_overlap': 0, 'shared_parent_count': 4,
        'shared_parents_reused_across_classifier_splits': True,
        'all_original_pilot_pool_content_excluded': True,
        'failed_or_unqualified_tasks_count_toward_target': False,
        'data_hashes': {str(p.relative_to(out)): sha(p) for p in sorted((out / 'data').rglob('*.jsonl'))},
        'inventory_sha256': sha(out / 'inventory.jsonl'), 'reserves_sha256': sha(out / 'reserve-targets.jsonl')}
    (out / 'PRODUCTION_DATA_READY.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'primary_targets': 400, 'reserve_targets': len(spares), 'training_started': False}), flush=True)


if __name__ == '__main__':
    main()
