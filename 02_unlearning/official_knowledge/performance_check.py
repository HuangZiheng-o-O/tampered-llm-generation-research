"""Measure safe micro-batch throughput with official two-update NPO jobs."""
import json
from pathlib import Path

from omegaconf import OmegaConf

from official_knowledge.prepare_data import write
from official_knowledge.production_jobs import atomic_json
from official_knowledge.run_pilot import now


def choose_microbatch(pilot, builder):
    receipt = builder.root / 'PERFORMANCE_CHECK.json'
    if receipt.exists():
        result = json.loads(receipt.read_text())
        assert result['passed']
        return result['selected_microbatch']
    # Existing published records are repeated only inside this hardware test.
    # These adapters are never counted as accepted unlearning models.
    rows = [r for u in builder.units.values() for r in u['learning']]
    if getattr(builder, 'evaluation_protocol', None) == 'published_qa_answers_only_v2':
        rows = [r for r in rows if r['kind'] == 'published_qa']
    rows = sorted(rows, key=lambda r: r['tokens'], reverse=True)[:64]
    assert len(rows) == 64
    data = builder.root / 'performance-data/longest-64.jsonl'
    write(data, rows)
    candidates = []
    for batch in (1, 2, 4):
        name = builder.source + '_performance_b' + str(batch)
        cfg = builder.template(builder.source + '_development_NPO')
        output = builder.root / 'runs' / name
        builder.output(cfg, name, output, True)
        builder.final_only(cfg)
        small_manifest = json.loads((builder.pilot / 'configs/jobs.json').read_text())
        cfg.model.model_args.pretrained_model_name_or_path = str(
            Path(small_manifest['jobs'][builder.source + '_learn']['output']) / 'merged')
        cfg.model.tokenizer_args.pretrained_model_name_or_path = builder.parent['hdd_original']
        cfg.trainer.args.update(per_device_train_batch_size=batch, gradient_accumulation_steps=32 // batch,
                                max_steps=2, num_train_epochs=1)
        for split in ('forget', 'retain'):
            cfg.data[split].published_records.args.hf_args.data_files = str(data)
            cfg.data[split].published_records.args.hf_args.cache_dir = str(builder.root / 'cache/performance')
        builder.register(name, cfg, {'entry': 'src/train.py', 'kind': 'train', 'output': str(output),
            'requires': [], 'primary_rows': 64, 'scope': 'two_update_performance_check_only'})
        builder.manifest.setdefault('shared_files_sha256', {})[str(data)] = __import__('hashlib').sha256(data.read_bytes()).hexdigest()
        builder.save()
        pilot.manifest = builder.manifest
        try:
            pilot.run(name)
        except RuntimeError:
            tail = (pilot.logs / (name + '.log')).read_text(errors='replace')
            if 'out of memory' in tail.lower() or 'OutOfMemoryError' in tail:
                candidates.append({'microbatch': batch, 'feasible': False, 'reason': 'own_cuda_allocator_limit'})
                continue
            raise
        state = json.loads((output / 'trainer_state.json').read_text())
        runtime = float(next(row['train_runtime'] for row in reversed(state['log_history']) if 'train_runtime' in row))
        profile = json.loads((pilot.logs / (name + '.resource-limits.json')).read_text())
        peak = profile['cuda_peak_allocated_bytes']
        cap = profile['cuda_allocator_cap_bytes']
        candidates.append({'microbatch': batch, 'feasible': peak <= cap * .95, 'train_runtime_seconds': runtime,
            'row_pairs_per_second': 64 / runtime, 'peak_allocated_bytes': peak, 'allocator_cap_bytes': cap})
    safe = [c for c in candidates if c['feasible']]
    if not safe:
        raise RuntimeError('No benchmark candidate retained the required GPU memory headroom')
    best = min(safe, key=lambda c: c['train_runtime_seconds'])['microbatch']
    eval_choice, eval_measurements = evaluation_batch_check(pilot, builder)
    atomic_json(receipt, {'passed': True, 'at_utc': now(), 'proof_kind': 'hardware_performance_check_only',
        'accepted_unlearning_models': 0, 'selected_microbatch': best, 'effective_batch': 32,
        'selected_eval_batch': eval_choice, 'evaluation_measurements': eval_measurements,
        'record_count': 64, 'min_tokens': min(r['tokens'] for r in rows), 'max_tokens': max(r['tokens'] for r in rows),
        'candidate_measurements': candidates})
    return best


def evaluation_batch_check(pilot, builder):
    small = json.loads((builder.pilot / 'configs/jobs.json').read_text())
    model = str(Path(small['jobs'][builder.source + '_learn']['output']) / 'merged')
    outcomes, times = {}, {}
    for batch in (1, 2):
        name = builder.source + '_performance_eval_b' + str(batch)
        cfg = builder.template(builder.source + '_learn_mmlu_validation')
        output = builder.root / 'evals' / name
        builder.output(cfg, name, output, False)
        cfg.model.model_args.pretrained_model_name_or_path = model
        cfg.model.tokenizer_args.pretrained_model_name_or_path = builder.parent['hdd_original']
        if cfg.model.get('lora'):
            del cfg.model.lora
        cfg.eval.lm_eval.simple_evaluate_args.update(batch_size=batch, max_batch_size=2, limit=1)
        builder.register(name, cfg, {'entry': 'src/eval.py', 'kind': 'eval', 'output': str(output), 'model': model,
            'source': builder.source, 'assessment': 'mmlu_validation', 'expected_mmlu_count': 57, 'requires': []})
        builder.save()
        pilot.manifest = builder.manifest
        try:
            pilot.run(name)
        except RuntimeError:
            log = (pilot.logs / (name + '.log')).read_text(errors='replace')
            if batch == 2 and ('out of memory' in log.lower() or 'OutOfMemoryError' in log):
                outcomes[2] = {'feasible': False}
                continue
            raise
        logs = json.loads((output / 'LMEval_EVAL.json').read_text())['pilot_mmlu_validation']
        predictions = {}
        for task, samples in logs.items():
            for index, sample in enumerate(samples):
                predictions[task + ':' + str(sample.get('doc_id', index))] = max(
                    range(len(sample['filtered_resps'])), key=lambda j: sample['filtered_resps'][j][0])
        assert len(predictions) == 57
        profile = json.loads((pilot.logs / (name + '.resource-limits.json')).read_text())
        times[batch] = profile['runtime_seconds']
        outcomes[batch] = {'feasible': True, 'predictions': predictions,
            'runtime_seconds': profile['runtime_seconds'], 'peak_allocated_bytes': profile['cuda_peak_allocated_bytes']}
    matched = outcomes.get(2, {}).get('predictions') == outcomes[1]['predictions']
    faster = outcomes.get(2, {}).get('feasible') and times[2] < times[1]
    # Automatic detection is capped at two and can select one for longer inputs.
    selected = 'auto' if matched and faster else 1
    return selected, {'questions_per_batch_candidate': 57, 'prediction_match': matched,
                      'candidates': outcomes, 'auto_max_batch': 2}
