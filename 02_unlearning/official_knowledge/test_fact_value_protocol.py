"""Validate every formal biography span and the production case data path on CPU."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from transformers import AutoTokenizer

from official_knowledge.fact_values import PROTOCOL, validated_rows
from official_knowledge.prepare_data import write
from official_knowledge.production_jobs import ProductionJobs, atomic_json, records
from official_knowledge.run_pilot import now, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    from data.utils import preprocess_pretraining_instance
    parents = [p for p in json.loads((a.root / 'formal-lora-400-v1/parents.json').read_text())
               if p['source'] == 'biographies']
    counts, folders = {}, []
    for parent in parents:
        units = {u['unit_id']: u for u in records(Path(parent['data_dir']) / 'units.jsonl')}
        tokenizer = AutoTokenizer.from_pretrained(parent['hdd_original'], local_files_only=True)
        rows = [r for u in units.values() for r in u['learning']]
        converted = validated_rows(rows, units, tokenizer, preprocess_pretraining_instance)
        assert len(converted) == 21000 and len({r['unit_id'] for r in converted}) == 175
        counts[parent['base']] = {'records': len(converted), 'people': len(units),
            'supervised_tokens_min': min(r['supervised_tokens'] for r in converted),
            'supervised_tokens_max': max(r['supervised_tokens'] for r in converted)}
        root = Path(tempfile.mkdtemp(prefix='fact-value-case-software-', dir='/local/rcs/zh2701/LLM-safety/tmp'))
        folders.append(str(root))
        (root / 'PROOF_KIND.txt').write_text('SOFTWARE FIXTURE ONLY. No model loaded or trained.\n')
        builder = object.__new__(ProductionJobs)
        builder.root, builder.project, builder.parent = root, a.root, parent
        builder.source, builder.pilot = 'biographies', a.root / ('official-lora-pilot-v1-' + parent['base'])
        builder.configs, builder.tasks = root / 'configs', root / 'tasks'
        builder.configs.mkdir()
        builder.tasks.mkdir()
        builder.units = dict(list(units.items())[:2])
        builder.manifest = copy.deepcopy(json.loads((builder.pilot / 'configs/jobs.json').read_text()))
        builder.manifest.update(jobs={}, config_sha256={}, task_sha256={})
        unit = next(iter(builder.units.values()))
        other = list(builder.units.values())[1]
        write(root / 'data/eval-retain.jsonl', other['probes'])
        task = {'task_id': 'software-only-case', 'unit_id': unit['unit_id'], 'method': 'NPO'}
        with patch.object(builder, 'data_protocol', return_value='full_published_records_v1'):
            name, _ = builder.case(task)
        old = root / 'case-data/software-only-case/forget.jsonl'
        old_sha = sha(old)
        with patch.object(builder, 'data_protocol', return_value=PROTOCOL):
            builder.case(task)
        folder = root / 'case-data-protocols' / PROTOCOL / task['task_id']
        actual = records(folder / 'forget.jsonl')
        assert len(actual) == 120 and all(r['loss_protocol'] == PROTOCOL for r in actual)
        assert sha(old) == old_sha
        cfg = builder.manifest['jobs'][name]
        assert cfg['primary_rows'] == 120 and cfg['forget_data_protocol'] == PROTOCOL
        from omegaconf import OmegaConf
        training = OmegaConf.load(builder.configs / (name + '.yaml'))
        assert training.data.forget.published_records.args.hf_args.data_files == str(folder / 'forget.jsonl')
        assert training.trainer.args.save_strategy == 'no' and not training.checkpoint_first_step
        out = Path(cfg['output'])
        out.mkdir(parents=True)
        atomic_json(out / 'prepared-job.json', {'proof_kind': 'software_fixture_only'})
        with patch.object(builder, 'data_protocol', return_value='full_published_records_v1'):
            try:
                builder.case(task)
            except ValueError as error:
                assert 'cannot change' in str(error)
            else:
                raise AssertionError('A started case adopted a different data protocol')
    receipt = {'at_utc': now(), 'passed': True, 'proof_kind': 'data_and_software_validation_only',
        'model_loaded': False, 'optimizer_steps': 0, 'formal_production_authorized': False,
        'all_formal_biography_rows': counts, 'fixtures': folders,
        'checks': ['exact_literal_answer_tokens_only', 'all_prefix_tokens_masked', 'no_eos_loss',
                   'actual_production_case_uses_validated_span_records',
                   'full_record_cache_preserved_and_separate', 'started_case_protocol_change_rejected']}
    atomic_json(a.root / 'FACT_VALUE_PROTOCOL_CHECK.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
