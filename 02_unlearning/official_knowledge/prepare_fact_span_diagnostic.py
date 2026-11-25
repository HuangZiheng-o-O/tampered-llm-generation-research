"""Mask published biography prefixes; apply forgetting loss to fact values only."""
import argparse
import copy
import fcntl
import json
from pathlib import Path

from omegaconf import OmegaConf
from transformers import AutoTokenizer

from official_knowledge.prepare_data import biographies, write
from official_knowledge.production_jobs import atomic_json
from official_knowledge.run_pilot import now, sha
from official_knowledge.fact_values import fact_span, validated_rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--base', choices=('llama2-chat', 'llama3-instruct'), required=True)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--methods', nargs='+', choices=('NPO', 'GradDiff', 'SimNPO', 'UNDIAL'), default=['NPO'])
    args = p.parse_args()
    import sys
    sys.path.insert(0, str(args.upstream / 'src'))
    from data.utils import preprocess_pretraining_instance
    suite = args.root / ('official-lora-pilot-v1-' + args.base)
    units, _ = biographies(args.root / 'datasets/knowledge-20261101')
    units = {u['unit_id']: u for u in units}
    parent = next(p for p in json.loads((args.root / 'formal-lora-400-v1/parents.json').read_text())
                  if p['base'] == args.base and p['source'] == 'biographies')
    tokenizer = AutoTokenizer.from_pretrained(parent['hdd_original'], local_files_only=True)
    evidence = {}
    with (suite / 'configs/recipe-preparation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest_path = suite / 'configs/jobs.json'
        manifest = json.loads(manifest_path.read_text())
        for phase in ('development', 'confirmation'):
            original = suite / 'data/biographies' / (phase + '-forget.jsonl')
            rows = [json.loads(line) for line in original.read_text().splitlines()]
            converted = validated_rows(rows, units, tokenizer, preprocess_pretraining_instance)
            output_data = suite / 'data/fact-span-diagnostic' / (phase + '-forget.jsonl')
            if output_data.exists():
                assert [json.loads(line) for line in output_data.read_text().splitlines()] == converted
            else:
                write(output_data, converted)
            manifest.setdefault('shared_files_sha256', {})[str(output_data)] = sha(output_data)
            for method in args.methods:
                old_name = 'biographies_' + phase + '_' + method
                name = old_name + '_factspan'
                for tail in ('', '_knowledge', '_mmlu_validation'):
                    new = name + tail
                    if new in manifest['jobs']:
                        manifest['jobs'][new]['forget_data_protocol'] = 'published_fact_value_tokens_v1'
                        continue
                    cfg = OmegaConf.load(suite / 'configs' / (old_name + tail + '.yaml'))
                    job = copy.deepcopy(manifest['jobs'][old_name + tail])
                    output = suite / ('evals' if tail else 'runs') / new
                    cfg.task_name, cfg.paths.output_dir, cfg.hydra.run.dir = new, str(output), str(output)
                    job['output'] = str(output)
                    job['forget_data_protocol'] = 'published_fact_value_tokens_v1'
                    job['requires'] = [n.replace(old_name, name) for n in job['requires']]
                    if tail:
                        cfg.eval.lm_eval.output_dir = str(output)
                        cfg.model.lora.adapter_path = str(suite / 'runs' / name)
                        job['stage'] = phase + '_' + method + '_factspan'
                    else:
                        cfg.trainer.args.output_dir = str(output)
                        cfg.trainer.args.logging_dir = str(output / 'logs')
                        cfg.trainer.args.save_strategy = 'no'
                        cfg.trainer.args.save_only_model = True
                        cfg.checkpoint_first_step = False
                        cfg.native_checkpoint_io = False
                        cfg.data.forget.published_records.args.hf_args.data_files = str(output_data)
                        job['scope'] = 'fact_span_loss_diagnostic_only'
                    file = suite / 'configs' / (new + '.yaml')
                    OmegaConf.save(cfg, file)
                    manifest['jobs'][new] = job
                    manifest['config_sha256'][file.name] = sha(file)
            evidence[phase] = {'records': len(converted), 'fact_groups': len({r['fact_id'] for r in converted}),
                'original_data_sha256': sha(original), 'converted_data_sha256': sha(output_data),
                'supervised_tokens_min': min(r['supervised_tokens'] for r in converted),
                'supervised_tokens_max': max(r['supervised_tokens'] for r in converted),
                'example': {k: converted[0][k] for k in ('prompt', 'text', 'answer')}}
        atomic_json(manifest_path, manifest)
    receipt = {'at_utc': now(), 'passed': True, 'proof_kind': 'data_and_official_tokenization_validation_only',
        'optimizer_steps': 0, 'official_loss_changed': False, 'new_facts_or_questions_created': False,
        'original_learning_and_evaluation_data_changed': False, 'loss_on_subject_or_prefix': False,
        'loss_on_eos': False, 'formal_production_authorized': False, 'phases': evidence}
    receipt['prepared_methods'] = args.methods
    name = 'FACTSPAN_PREPARATION.json' if args.methods == ['NPO'] else 'FACTSPAN_FALLBACK_PREPARATION.json'
    atomic_json(suite / name, receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
